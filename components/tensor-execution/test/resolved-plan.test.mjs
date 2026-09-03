import assert from 'node:assert/strict';
import test from 'node:test';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { lowerSimtPlan, resolveTensorPlanWithAdapter } from '../testing.mjs';

function fakeRuntime() {
  let closed = false;
  let sequence = 0;
  const allocations = [];
  return {
    allocations,
    async describe() { return { package: { name: 'cuda-js', version: '0.1.0-alpha.17', publicApiSchema: 1 }, state: 'open', profile: 'tensor-simt-test', device: null }; },
    async allocateDevice({ byteLength }) {
      const memory = {
        kind: 'device-memory', state: 'open', byteLength, id: ++sequence, views: [],
        async view(options) {
          const view = {
            kind: 'device-view', state: 'open', ...options,
            async status() { return { state: this.state }; },
            async close() { this.state = 'closed'; return { state: 'closed' }; },
          };
          this.views.push(view);
          return view;
        },
        async close() { this.state = 'closed'; return { state: 'closed' }; },
      };
      allocations.push(memory);
      return memory;
    },
    async close() { closed = true; return { graceful: true }; },
    get closed() { return closed; },
  };
}

function fakeAdapter(observation = {}) {
  return async (_runtime, lowering) => ({
    identity: `fake-simt:${lowering.compatibilityIdentity}`,
    descriptor: Object.freeze({ realization: 'portable-test', lowering: lowering.compatibilityIdentity }),
    async execute(bindings) {
      observation.executions = (observation.executions ?? 0) + 1;
      observation.bindingNames = Object.keys(bindings);
      return Object.freeze({ status: 'completed', realization: 'portable-test' });
    },
    async close() { observation.closed = true; return Object.freeze({ graceful: true, failures: Object.freeze([]) }); },
  });
}

test('SIMT lowering covers the accepted dense operation families deterministically', () => {
  const program = TensorProgram.define((graph) => {
    const matrix = graph.input('matrix', { dtype: 'f32', capacityShape: [2, 3], access: 'read' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read' });
    const filled = graph.fill({ dtype: 'f16', capacityShape: [2, 3] }, 1.5);
    const copied = graph.copy(matrix);
    const cast = graph.cast(copied, 'f64');
    const reshaped = graph.reshape(cast, [3, 2]);
    const permuted = graph.permute(reshaped, [1, 0]);
    const sliced = graph.slice(permuted, [{ start: 0, length: 2 }, { start: 0, length: 1 }]);
    const broadcast = graph.broadcast(sliced, [2, 3]);
    const contiguous = graph.contiguous(broadcast);
    const unary = graph.unary('sqrt', contiguous);
    const binary = graph.binary('maximum', unary, unary);
    const reduced = graph.reduce('sum', binary, { axes: [1], order: 'fixed-tree-v1' });
    const matmul = graph.matmul(matrix, right);
    return { filled, reduced, matmul };
  });
  const plan = TensorPlan.create(program);
  const first = lowerSimtPlan(plan);
  const second = lowerSimtPlan(plan);
  assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
  assert.equal(first.source, second.source);
  assert.equal(first.kernels.length, 10);
  assert(first.totalWorkspaceBytes > 0);
  assert.match(first.source, /gpu\.math\.maximum/u);
  assert.match(first.source, /gpu\.cast\.f64/u);
  assert.match(first.source, /for \(let k = gpu\.u64/u);
  assert.doesNotMatch(first.source, /#include|__global__|cuda[A-Z]/u);
  assert.equal(first.canonical.kernels.every((entry) => entry.after.length <= 1), true);
});

test('resolved plan and result normalize convenience bindings and own only run-created tensors', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const input = await session.allocate({ dtype: 'f32', capacityShape: [4], access: 'read' });
  const program = TensorProgram.define((graph) => graph.unary('neg', graph.input('values', { dtype: 'f32', capacityShape: [4], access: 'read' })));
  const observation = {};
  const resolved = await resolveTensorPlanWithAdapter(session, program, undefined, fakeAdapter(observation));
  assert.equal(resolved.backend, 'simt');
  assert.equal(resolved.kernelCount, 1);
  assert.equal(resolved.describe().backendDescriptor.realization, 'portable-test');

  const result = await resolved.run(input);
  assert.equal(result.output, result.get('output'));
  assert.equal(result.output.dtype, 'f32');
  assert.equal(observation.executions, 1);
  assert.deepEqual(observation.bindingNames, ['b0', 'b1']);
  assert.equal((await result.close()).graceful, true);
  assert.equal(input.state, 'open');
  assert.equal((await resolved.close()).graceful, true);
  assert.equal(observation.closed, true);
  await input.close();
  assert.equal((await session.close()).graceful, true);
  assert.equal(runtime.closed, true);
});

test('named and positional bindings share exact validation and declared alias truth', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const spec = { dtype: 'f32', capacityShape: [2], access: 'read', aliasGroup: 'same' };
  const program = TensorProgram.define((graph) => {
    const left = graph.input('left', spec);
    const right = graph.input('right', spec);
    return graph.binary('add', left, right);
  });
  const owner = await session.allocate(spec);
  const alias = await owner.view();
  const resolved = await resolveTensorPlanWithAdapter(session, TensorPlan.create(program), {}, fakeAdapter());
  const first = await resolved.run([owner, alias]);
  await first.close();
  const second = await resolved.run({ left: owner, right: alias });
  await second.close();
  const distinct = await session.allocate(spec);
  await assert.rejects(resolved.run([owner, distinct]), { code: 'TENSOR_EXECUTION_ALIAS_MISMATCH' });
  await distinct.close();
  await resolved.close();
  await alias.close();
  await owner.close();
  assert.equal((await session.close()).graceful, true);
});

test('fixed-tree workspace, prepared limits, and zero-work programs fail or resolve explicitly', async () => {
  const reduction = TensorProgram.define((graph) => graph.reduce('sum', graph.input('values', { dtype: 'f32', capacityShape: [8], access: 'read' })));
  assert.throws(() => lowerSimtPlan(TensorPlan.create(reduction), { maxWorkspaceBytes: 1 }), { code: 'TENSOR_SIMT_WORKSPACE_LIMIT' });

  const tooMany = TensorProgram.define((graph) => {
    let value = graph.input('values', { dtype: 'f32', capacityShape: [1], access: 'read' });
    for (let index = 0; index < 33; index += 1) value = graph.copy(value);
    return value;
  });
  assert.throws(() => lowerSimtPlan(TensorPlan.create(tooMany)), { code: 'TENSOR_SIMT_KERNEL_LIMIT' });

  const runtime = fakeRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const empty = TensorProgram.define((graph) => graph.fill({ dtype: 'bf16', capacityShape: [0] }, 0));
  const observation = {};
  const resolved = await resolveTensorPlanWithAdapter(session, empty, {}, fakeAdapter(observation));
  assert.equal(resolved.kernelCount, 0);
  const result = await resolved.run();
  assert.equal(result.output.capacityShape[0], 0);
  assert.equal(observation.executions, 1);
  await result.close();
  await resolved.close();
  assert.equal((await session.close()).graceful, true);
});

test('resolved profile caps selected workspace and result cleanup preserves retryable child backpressure', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const program = TensorProgram.define((graph) => graph.fill({ dtype: 'f32', capacityShape: [2] }, 1));
  await assert.rejects(resolveTensorPlanWithAdapter(session, program, { maxWorkspaceBytes: 64 * 1024 * 1024 + 1 }, fakeAdapter()), { code: 'TENSOR_SIMT_WORKSPACE_LIMIT_INVALID' });
  const resolved = await resolveTensorPlanWithAdapter(session, program, {}, fakeAdapter());
  const result = await resolved.run();
  const child = await result.output.view();
  await assert.rejects(result.close(), { code: 'TENSOR_EXECUTION_RESULT_BUSY' });
  assert.equal(result.state, 'closing');
  await child.close();
  assert.equal((await result.close()).graceful, true);
  await resolved.close();
  assert.equal((await session.close()).graceful, true);
});

test('resolution proves empty-session fit and run pressure rolls back partial allocations', async () => {
  const program = TensorProgram.define((graph) => graph.copy(graph.copy(graph.input('values', { dtype: 'f32', capacityShape: [4], access: 'read' }))));

  const impossibleRuntime = fakeRuntime();
  const impossibleSession = await TensorSession.open({
    runtime: impossibleRuntime,
    runtimeOwnership: 'owned',
    limits: { maxTensorBytes: 16, maxSessionBytes: 16, maxLiveTensors: 8 },
  });
  let adapterCreated = false;
  await assert.rejects(resolveTensorPlanWithAdapter(impossibleSession, program, {}, async (...args) => {
    adapterCreated = true;
    return fakeAdapter()(...args);
  }), { code: 'TENSOR_RESOLVE_SESSION_BYTE_LIMIT' });
  assert.equal(adapterCreated, false);
  assert.equal((await impossibleSession.close()).graceful, true);

  const runtime = fakeRuntime();
  const session = await TensorSession.open({
    runtime,
    runtimeOwnership: 'owned',
    limits: { maxTensorBytes: 16, maxSessionBytes: 32, maxLiveTensors: 8 },
  });
  const input = await session.allocate({ dtype: 'f32', capacityShape: [4], access: 'read' });
  const observation = {};
  const resolved = await resolveTensorPlanWithAdapter(session, program, {}, fakeAdapter(observation));
  await assert.rejects(resolved.run(input), { code: 'TENSOR_SESSION_BYTE_LIMIT' });
  assert.equal(observation.executions ?? 0, 0);
  const accounting = (await session.status()).accounting;
  assert.deepEqual({ liveTensors: accounting.liveTensors, reservedBytes: accounting.reservedBytes }, { liveTensors: 1, reservedBytes: 16 });
  await resolved.close();
  await input.close();
  assert.equal((await session.close()).graceful, true);
});

test('session owns resolved-plan creation, cascade close, and unproved terminal accounting', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const program = TensorProgram.define((graph) => graph.fill({ dtype: 'f32', capacityShape: [1] }, 1));
  const cascade = {};
  const resolved = await resolveTensorPlanWithAdapter(session, program, {}, fakeAdapter(cascade));
  const result = await resolved.run();
  assert.equal((await session.status()).accounting.resolvedPlans, 1);
  const terminal = await session.close();
  assert.equal(terminal.graceful, true);
  assert.equal(resolved.state, 'closed');
  assert.equal(result.state, 'closed');
  assert.equal(cascade.closed, true);

  const failedRuntime = fakeRuntime();
  const failedSession = await TensorSession.open({ runtime: failedRuntime, runtimeOwnership: 'owned' });
  const failed = await resolveTensorPlanWithAdapter(failedSession, program, {}, async (_runtime, lowering) => ({
    identity: `unproved:${lowering.compatibilityIdentity}`,
    descriptor: Object.freeze({ realization: 'portable-unproved' }),
    async execute() { return Object.freeze({ status: 'completed' }); },
    async close() { return Object.freeze({ graceful: false, failures: Object.freeze([{ code: 'BACKEND_CLOSE_UNPROVED' }]) }); },
  }));
  assert.equal((await failed.close()).graceful, false);
  assert.equal((await failedSession.status()).accounting.unprovedResources, 1);
  const failedTerminal = await failedSession.close();
  assert.equal(failedTerminal.graceful, false);
  assert.equal(failedTerminal.state, 'orphaned');
});

test('session close backpressures while resolved-plan construction is pending', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned' });
  const program = TensorProgram.define((graph) => graph.fill({ dtype: 'f32', capacityShape: [1] }, 1));
  let finish;
  const adapter = new Promise((resolve) => { finish = resolve; });
  const resolving = resolveTensorPlanWithAdapter(session, program, {}, async () => adapter);
  await Promise.resolve();
  assert.equal((await session.status()).accounting.pendingResolutions, 1);
  await assert.rejects(session.close(), { code: 'TENSOR_SESSION_BUSY' });
  finish({
    identity: 'pending-adapter',
    descriptor: Object.freeze({ realization: 'portable-pending' }),
    async execute() { return Object.freeze({ status: 'completed' }); },
    async close() { return Object.freeze({ graceful: true, failures: Object.freeze([]) }); },
  });
  const resolved = await resolving;
  assert.equal((await session.status()).accounting.resolvedPlans, 1);
  await resolved.close();
  assert.equal((await session.close()).graceful, true);
});
