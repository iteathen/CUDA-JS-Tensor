import assert from 'node:assert/strict';
import test from 'node:test';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { createBackendProfileRequest, createCudaJsTensorBackend, lowerSimtPlan, resolveTensorPlanWithAdapter } from '../testing.mjs';

const SHA256 = '1'.repeat(64);

function codedError(code, category = 'unsupported') {
  return Object.assign(new Error(code), { code, category, details: {} });
}

function fakePublicRuntime({ openError = null, planError = null, prepareError = null, workspaceBytes = 256 } = {}) {
  const observation = { adapterOpens: 0, adapterCloses: 0, planCreates: [], planCloses: 0, prepared: [], submissions: [], views: [] };
  let sequence = 0;
  let runtimeClosed = false;

  function stateful(kind, extra, onClose = null) {
    let state = 'open';
    return Object.freeze({
      kind,
      ...extra,
      get state() { return state; },
      async status() { return { state }; },
      async close() { if (state === 'open') { state = 'closed'; await onClose?.(); } return { state }; },
    });
  }

  const runtime = {
    observation,
    async describe() { return { package: { name: 'cuda-js', version: '0.1.0-alpha.16', publicApiSchema: 1 }, state: 'open', profile: 'tensor-cublaslt-test', device: { architecture: { class: 'compute_75' } } }; },
    async allocateDevice({ byteLength }) {
      const views = [];
      const memory = stateful('device-memory', {
        byteLength,
        async view(options) {
          const view = stateful('device-view', { ...options, byteLength: options.elementCount * 4 });
          views.push(view);
          observation.views.push(options);
          return view;
        },
      });
      return memory;
    },
    async compile(request) {
      observation.compile = request;
      return Object.freeze({ artifact: Object.freeze({ format: 'ptx', bytes: new Uint8Array([1]), byteLength: 1, sha256: SHA256, architecture: 'compute_75' }) });
    },
    async link() { throw codedError('UNEXPECTED_LINK', 'internal'); },
    async loadModule() {
      return stateful('module', {
        sha256: SHA256,
        format: 'ptx',
        async getFunction({ name, parameters }) { return stateful('function', { name, parameters }); },
      });
    },
    async prepareOperationDag(value) {
      if (prepareError) throw codedError(prepareError, 'validation');
      const nodes = Array.isArray(value) ? value : value.nodes;
      observation.prepared.push(nodes);
      const bindingNames = new Set();
      for (const node of nodes) {
        if (node.kind === 'cublaslt-f32-matmul') {
          for (const key of ['a', 'b', 'c', 'd', 'workspace']) if (node[key]?.binding) bindingNames.add(node[key].binding);
        } else {
          for (const argument of node.arguments) if (argument?.binding) bindingNames.add(argument.binding);
        }
      }
      return stateful('prepared-operation-dag', {
        contract: nodes.some((node) => node.kind === 'cublaslt-f32-matmul')
          ? 'SPEC-0020-prepared-kernel-dag-v1+SPEC-0031-prepared-cublaslt-f32-matmul-node-v1'
          : 'SPEC-0020-prepared-kernel-dag-v1',
        sha256: SHA256,
        nodeCount: nodes.length,
        edgeCount: nodes.reduce((count, node) => count + (node.after?.length ?? 0), 0),
        bindings: Object.freeze([...bindingNames].map((name) => Object.freeze({ name, kind: 'device-memory' }))),
        realization: 'semantic-single-stream',
        async submit({ bindings }) {
          observation.submissions.push(bindings);
          let state = 'pending';
          return Object.freeze({
            kind: 'operation',
            get state() { return state; },
            async wait() { state = 'completed'; return { status: 'completed', operationSequence: ++sequence, preparedSha256: SHA256 }; },
            async close() { state = 'closed'; return { state }; },
          });
        },
      });
    },
    async openCublasLt() {
      observation.adapterOpens += 1;
      if (openError) throw codedError(openError);
      return stateful('cublaslt-adapter', {
        profile: 'cublaslt-f32-row-major-matmul-v1',
        provider: Object.freeze({ name: 'cuBLASLt', version: 'mock-13.3', qualification: 'portable-contract-only' }),
        async createF32MatmulPlan(options) {
          observation.planCreates.push(options);
          if (planError) throw codedError(planError);
          const selectedWorkspace = Math.min(workspaceBytes, options.maxWorkspaceBytes);
          return stateful('cublaslt-matmul-plan', {
            contract: 'SPEC-0029-cublaslt-f32-row-major-matmul-v1',
            ...options,
            workspaceBytes: selectedWorkspace,
            requirements: Object.freeze({ a: options.m * options.k, b: options.k * options.n, c: options.m * options.n, d: options.m * options.n }),
          }, async () => { observation.planCloses += 1; });
        },
      }, async () => { observation.adapterCloses += 1; });
    },
    async close() { runtimeClosed = true; return { graceful: true }; },
    get closed() { return runtimeClosed; },
  };
  return runtime;
}

async function fakeCompileDeviceProgram(runtime, request) {
  const compiler = await runtime.compile({ source: request.source });
  const kernels = request.functions.filter((entry) => entry.kind === 'kernel').map((entry) => Object.freeze({
    name: entry.name,
    functionName: entry.name,
    parameters: Object.freeze(entry.parameters.map(() => Object.freeze({ kind: 'device-memory' }))),
  }));
  return Object.freeze({
    compiler: Object.freeze({ ...compiler, headerProfile: null }),
    deviceProgram: Object.freeze({ contract: 'tensor-cublaslt-test-device-program-v1', sha256: SHA256, kernels: Object.freeze(kernels) }),
  });
}

function resolveWithFakeCompiler(session, planOrProgram, options) {
  return resolveTensorPlanWithAdapter(session, planOrProgram, options, (runtime, lowering, request, normalized) => createCudaJsTensorBackend(
    runtime,
    lowering,
    request,
    normalized,
    Object.freeze({ compileDeviceProgram: fakeCompileDeviceProgram }),
  ));
}

function mixedProgram() {
  return TensorProgram.define((graph) => {
    const left = graph.unary('abs', graph.input('left', { dtype: 'f32', capacityShape: [2, 3], access: 'read' }));
    const right = graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read' });
    return graph.unary('neg', graph.matmul(left, right));
  });
}

test('backend profile fixes exact eligibility and strict rejection before adapter work', () => {
  const program = mixedProgram();
  const plan = TensorPlan.create(program);
  const lowering = lowerSimtPlan(plan);
  const request = createBackendProfileRequest(plan, lowering, { backend: 'prefer-cublaslt' });
  assert.equal(request.candidates.length, 1);
  assert.deepEqual({ m: request.candidates[0].m, n: request.candidates[0].n, k: request.candidates[0].k }, { m: 2, n: 2, k: 3 });

  const transposed = TensorProgram.define((graph) => graph.matmul(
    graph.input('left', { dtype: 'f32', capacityShape: [3, 2], access: 'read' }),
    graph.input('right', { dtype: 'f32', capacityShape: [2, 3], access: 'read' }),
    { transposeA: true, transposeB: true },
  ));
  const transposedPlan = TensorPlan.create(transposed);
  const transposedRequest = createBackendProfileRequest(transposedPlan, lowerSimtPlan(transposedPlan), { backend: 'cublaslt' });
  assert.deepEqual({ m: transposedRequest.candidates[0].m, n: transposedRequest.candidates[0].n, k: transposedRequest.candidates[0].k, transposeA: transposedRequest.candidates[0].transposeA, transposeB: transposedRequest.candidates[0].transposeB }, { m: 2, n: 2, k: 3, transposeA: true, transposeB: true });

  const batched = TensorProgram.define((graph) => graph.matmul(
    graph.input('left', { dtype: 'f32', capacityShape: [2, 2, 3], access: 'read' }),
    graph.input('right', { dtype: 'f32', capacityShape: [2, 3, 2], access: 'read' }),
  ));
  const batchedPlan = TensorPlan.create(batched);
  const batchedLowering = lowerSimtPlan(batchedPlan);
  assert.throws(() => createBackendProfileRequest(batchedPlan, batchedLowering, { backend: 'cublaslt' }), (error) => error.code === 'TENSOR_CUBLASLT_NODE_INELIGIBLE' && error.details.nodes[0].reasons.includes('rank-not-2'));
});

test('preference exposes dtype, batch, layout, active-extent, derived-offset, and empty fallbacks', () => {
  const cases = [
    {
      reason: 'dtype-not-f32',
      program: TensorProgram.define((graph) => graph.matmul(
        graph.input('left', { dtype: 'f64', capacityShape: [2, 3], access: 'read' }),
        graph.input('right', { dtype: 'f64', capacityShape: [3, 2], access: 'read' }),
      )),
    },
    {
      reason: 'rank-not-2',
      program: TensorProgram.define((graph) => graph.matmul(
        graph.input('left', { dtype: 'f32', capacityShape: [1, 2, 3], access: 'read' }),
        graph.input('right', { dtype: 'f32', capacityShape: [1, 3, 2], access: 'read' }),
      )),
    },
    {
      reason: 'layout-not-row-major-contiguous',
      program: TensorProgram.define((graph) => graph.matmul(
        graph.input('left', { dtype: 'f32', capacityShape: [2, 3], strides: [4, 1], access: 'read' }),
        graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read' }),
      )),
    },
    {
      reason: 'active-extent',
      program: TensorProgram.define((graph) => graph.matmul(
        graph.input('left', { dtype: 'f32', capacityShape: [2, 3], activeAxis0: { maximum: 2, extent: 1 }, access: 'read' }),
        graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read' }),
      )),
    },
    {
      reason: 'derived-byte-offset',
      program: TensorProgram.define((graph) => {
        const sliced = graph.slice(graph.input('left', { dtype: 'f32', capacityShape: [3, 3], access: 'read' }), [{ start: 1, length: 2 }, null]);
        return graph.matmul(sliced, graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read' }));
      }),
    },
    {
      reason: 'empty-matrix',
      program: TensorProgram.define((graph) => graph.matmul(
        graph.input('left', { dtype: 'f32', capacityShape: [0, 3], access: 'read' }),
        graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read' }),
      )),
    },
  ];
  for (const entry of cases) {
    const plan = TensorPlan.create(entry.program);
    const request = createBackendProfileRequest(plan, lowerSimtPlan(plan), { backend: 'prefer-cublaslt' });
    assert.equal(request.candidates.length, 0);
    assert.equal(request.matmuls[0].reasons.includes(entry.reason), true, entry.reason);
    assert.deepEqual(request.canonical.alignment, { operandByteOffset: 4, workspaceByteOffset: 256 });
  }
});

test('explicit preference builds one mixed prepared DAG and owns exact workspace', async () => {
  const runtime = fakePublicRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const left = await session.allocate({ dtype: 'f32', capacityShape: [2, 3], access: 'read' });
  const right = await session.allocate({ dtype: 'f32', capacityShape: [3, 2], access: 'read' });
  const resolved = await resolveWithFakeCompiler(session, mixedProgram(), { backend: 'prefer-cublaslt', maxWorkspaceBytes: 1024 });
  assert.equal(resolved.backendPolicy, 'prefer-cublaslt');
  assert.equal(resolved.backend, 'mixed');
  assert.equal(resolved.kernelCount, 2);
  assert.equal(resolved.cublasLtNodeCount, 1);
  assert.equal(resolved.workspaceBytes, 256);
  assert.equal(resolved.canonical.backendProfile.provider.identity.qualification, 'portable-contract-only');
  const nodes = runtime.observation.prepared[0];
  assert.deepEqual(nodes.map((node) => node.kind ?? 'kernel'), ['kernel', 'cublaslt-f32-matmul', 'kernel']);
  assert.equal(nodes[1].c.binding, nodes[1].d.binding);
  assert.equal(nodes[1].workspace.binding, 'b5');

  const result = await resolved.run({ left, right });
  assert.equal(runtime.observation.submissions.length, 1);
  assert.equal(runtime.observation.views.some((view) => view.dtype === 'u32' && view.elementCount === 64 && view.byteOffset === 0), true);
  await result.close();
  await resolved.close();
  assert.equal(runtime.observation.planCloses, 1);
  assert.equal(runtime.observation.adapterCloses, 1);
  await left.close();
  await right.close();
  assert.equal((await session.close()).graceful, true);
});

test('resolved plans share one session/runtime adapter and release it after the last plan', async () => {
  const runtime = fakePublicRuntime({ workspaceBytes: 0 });
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const program = mixedProgram();
  const first = await resolveWithFakeCompiler(session, program, { backend: 'cublaslt' });
  const second = await resolveWithFakeCompiler(session, TensorPlan.create(program), { backend: 'cublaslt' });
  assert.equal(runtime.observation.adapterOpens, 1);
  assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
  await first.close();
  assert.equal(runtime.observation.adapterCloses, 0);
  await second.close();
  assert.equal(runtime.observation.adapterCloses, 1);
  assert.equal((await session.close()).graceful, true);
});

test('preference records only admitted fallback outcomes while strict and identity failures remain hard', async () => {
  const unavailable = fakePublicRuntime({ openError: 'CUBLASLT_PROFILE_UNAVAILABLE' });
  const unavailableSession = await TensorSession.open({ runtime: unavailable, runtimeOwnership: 'owned' });
  const preferred = await resolveWithFakeCompiler(unavailableSession, mixedProgram(), { backend: 'prefer-cublaslt' });
  assert.equal(preferred.backend, 'simt');
  assert.equal(preferred.canonical.backendProfile.matmuls[0].reasons.includes('profile-unavailable:CUBLASLT_PROFILE_UNAVAILABLE'), true);
  await preferred.close();
  await assert.rejects(resolveWithFakeCompiler(unavailableSession, mixedProgram(), { backend: 'cublaslt' }), { code: 'CUBLASLT_PROFILE_UNAVAILABLE' });
  assert.equal((await unavailableSession.close()).graceful, true);

  const wrongIdentity = fakePublicRuntime({ openError: 'CUBLASLT_PROVIDER_IDENTITY' });
  const identitySession = await TensorSession.open({ runtime: wrongIdentity, runtimeOwnership: 'owned' });
  await assert.rejects(resolveWithFakeCompiler(identitySession, mixedProgram(), { backend: 'prefer-cublaslt' }), { code: 'CUBLASLT_PROVIDER_IDENTITY' });
  assert.equal((await identitySession.close()).graceful, true);

  const occupied = fakePublicRuntime({ openError: 'CUBLASLT_ADAPTER_ALREADY_OPEN' });
  const occupiedSession = await TensorSession.open({ runtime: occupied, runtimeOwnership: 'owned' });
  const occupiedFallback = await resolveWithFakeCompiler(occupiedSession, mixedProgram(), { backend: 'prefer-cublaslt' });
  assert.equal(occupiedFallback.backend, 'simt');
  assert.equal(occupiedFallback.canonical.backendProfile.matmuls[0].reasons.includes('profile-unavailable:CUBLASLT_ADAPTER_ALREADY_OPEN'), true);
  await occupiedFallback.close();
  assert.equal((await occupiedSession.close()).graceful, true);
});

test('algorithm preference fallback and post-plan preparation rollback close every acquired child', async () => {
  const unavailableAlgorithm = fakePublicRuntime({ planError: 'CUBLASLT_ALGORITHM_UNAVAILABLE' });
  const fallbackSession = await TensorSession.open({ runtime: unavailableAlgorithm, runtimeOwnership: 'owned' });
  const fallback = await resolveWithFakeCompiler(fallbackSession, mixedProgram(), { backend: 'prefer-cublaslt' });
  assert.equal(fallback.backend, 'simt');
  assert.equal(fallback.canonical.backendProfile.matmuls[0].reasons.includes('algorithm-unavailable:CUBLASLT_ALGORITHM_UNAVAILABLE'), true);
  assert.equal(unavailableAlgorithm.observation.adapterCloses, 1);
  await fallback.close();
  assert.equal((await fallbackSession.close()).graceful, true);

  const preparationFailure = fakePublicRuntime({ prepareError: 'PREPARED_REJECTED', workspaceBytes: 0 });
  const failureSession = await TensorSession.open({ runtime: preparationFailure, runtimeOwnership: 'owned' });
  await assert.rejects(resolveWithFakeCompiler(failureSession, mixedProgram(), { backend: 'cublaslt' }), { code: 'PREPARED_REJECTED' });
  assert.equal(preparationFailure.observation.planCloses, 1);
  assert.equal(preparationFailure.observation.adapterCloses, 1);
  assert.deepEqual((await failureSession.status()).accounting, { liveTensors: 0, reservedBytes: 0, pendingCreates: 0, resolvedPlans: 0, pendingResolutions: 0, unprovedResources: 0 });
  assert.equal((await failureSession.close()).graceful, true);
});

test('preference falls back at the resolved accelerator workspace-count gate while strict mode rejects', async () => {
  const runtime = fakePublicRuntime({ workspaceBytes: 256 });
  const session = await TensorSession.open({
    runtime,
    runtimeOwnership: 'owned',
    limits: { maxLiveTensors: 3 },
  });
  const preferred = await resolveWithFakeCompiler(session, mixedProgram(), { backend: 'prefer-cublaslt' });
  assert.equal(preferred.backend, 'simt');
  assert.equal(preferred.canonical.resourceLimits.maxAcceleratorWorkspaceCount, 0);
  assert.equal(preferred.canonical.backendProfile.matmuls[0].reasons.includes('resource-unavailable:TENSOR_CUBLASLT_WORKSPACE_COUNT_LIMIT'), true);
  assert.equal(runtime.observation.planCloses, 1);
  assert.equal(runtime.observation.adapterCloses, 1);
  await preferred.close();
  await assert.rejects(resolveWithFakeCompiler(session, mixedProgram(), { backend: 'cublaslt' }), { code: 'TENSOR_CUBLASLT_WORKSPACE_COUNT_LIMIT' });
  assert.equal(runtime.observation.planCloses, 2);
  assert.equal(runtime.observation.adapterCloses, 2);
  assert.equal((await session.close()).graceful, true);
});
