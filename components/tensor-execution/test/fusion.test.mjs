import assert from 'node:assert/strict';
import test from 'node:test';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { createBackendProfileRequest, createFusionProfile, lowerSimtPlan, realizeBackendProfile, resolveTensorPlanWithAdapter } from '../testing.mjs';

function plan(callback) {
  return TensorPlan.create(TensorProgram.define(callback));
}

function fakeRuntime() {
  let sequence = 0;
  const allocations = [];
  return {
    allocations,
    async describe() { return { package: { name: 'cuda-js', version: '0.1.0-alpha.16', publicApiSchema: 1 }, state: 'open', profile: 'tensor-fusion-test', device: null }; },
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
    async close() { return { graceful: true }; },
  };
}

function fakeAdapter(observation = {}) {
  return async (_runtime, lowering, request) => {
    observation.lowering = lowering;
    observation.request = request;
    return {
      identity: `fake-fusion:${lowering.compatibilityIdentity}`,
      descriptor: Object.freeze({ realization: 'portable-fusion-test', lowering: lowering.compatibilityIdentity }),
      async execute(bindings) {
        observation.bindings = Object.keys(bindings);
        return Object.freeze({ status: 'completed', realization: 'portable-fusion-test' });
      },
      async close() { return Object.freeze({ graceful: true, failures: Object.freeze([]) }); },
    };
  };
}

test('fusion profile selects deterministic maximal exact chains and preserves boundaries', () => {
  const tensorPlan = plan((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4] });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4] });
    const normalized = graph.unary('abs', left);
    const combined = graph.binary('add', normalized, right);
    return graph.unary('sqrt', combined);
  });
  const none = createFusionProfile(tensorPlan);
  const selected = createFusionProfile(tensorPlan, 'exact-elementwise');
  assert.equal(none.eligibleRegionCount, 1);
  assert.equal(none.regions.length, 0);
  assert.equal(selected.regions.length, 1);
  assert.equal(selected.fusedNodeCount, 3);
  assert.equal(selected.removedMaterialCount, 2);
  assert.deepEqual(selected.regions[0].nodeIds, ['node:0', 'node:1', 'node:2']);
  assert.deepEqual(selected.regions[0].externalInputs, ['input:left', 'input:right']);
  assert.deepEqual(selected.regions[0].removedMaterials, ['node:0', 'node:1']);
  assert.equal(selected.compatibilityIdentity, createFusionProfile(tensorPlan, 'exact-elementwise').compatibilityIdentity);

  const observable = plan((graph) => {
    const root = graph.unary('abs', graph.input('input', { dtype: 'f32', capacityShape: [4] }));
    const middle = graph.unary('sqrt', root);
    const tail = graph.unary('neg', middle);
    return { root, tail };
  });
  assert.deepEqual(createFusionProfile(observable, 'exact-elementwise').regions[0].nodeIds, ['node:1', 'node:2']);

  const fanout = plan((graph) => {
    const root = graph.unary('abs', graph.input('input', { dtype: 'f32', capacityShape: [4] }));
    return { left: graph.unary('sqrt', root), right: graph.unary('neg', root) };
  });
  assert.equal(createFusionProfile(fanout, 'exact-elementwise').regions.length, 0);

  const duplicate = plan((graph) => {
    const root = graph.unary('abs', graph.input('input', { dtype: 'f32', capacityShape: [4] }));
    return graph.binary('add', root, root);
  });
  assert.deepEqual(createFusionProfile(duplicate, 'exact-elementwise').regions[0].nodeIds, ['node:0', 'node:1']);

  const shapeChange = plan((graph) => {
    const narrow = graph.unary('abs', graph.input('narrow', { dtype: 'f32', capacityShape: [1, 3] }));
    const wide = graph.input('wide', { dtype: 'f32', capacityShape: [2, 3] });
    return graph.unary('neg', graph.binary('add', narrow, wide));
  });
  assert.deepEqual(createFusionProfile(shapeChange, 'exact-elementwise').regions[0].nodeIds, ['node:1', 'node:2']);
});

test('exact fusion emits one kernel and deletes only internal material bindings', () => {
  const tensorPlan = plan((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4] });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4] });
    const normalized = graph.unary('abs', left);
    const combined = graph.binary('add', normalized, right);
    return graph.unary('sqrt', combined);
  });
  const unfused = lowerSimtPlan(tensorPlan);
  const fused = lowerSimtPlan(tensorPlan, { fusion: 'exact-elementwise' });
  assert.equal(unfused.kernels.length, 3);
  assert.equal(unfused.materials.length, 3);
  assert.equal(fused.kernels.length, 1);
  assert.equal(fused.materials.length, 1);
  assert.equal(fused.bindings.length, 3);
  assert.equal(fused.materialBytes, tensorPlan.program.nodes.at(-1).outputSpec.byteLength);
  assert.deepEqual(fused.kernels[0].semanticNodes, ['node:0', 'node:1', 'node:2']);
  assert.equal(fused.kernels[0].fusionRegion, 'fusion:0');
  assert.match(fused.source, /let fusedValue0 = gpu\.math\.abs/u);
  assert.match(fused.source, /let fusedValue1 = \(fusedValue0 \+ p1\[f1Input1\]\)/u);
  assert.match(fused.source, /let fusedValue2 = gpu\.math\.sqrt\(fusedValue1\)/u);
  assert.equal((fused.source.match(/function tensorKernel/gu) ?? []).length, 1);
});

test('exact fusion keeps dtype transitions and broadcast side-input addressing explicit', () => {
  const typed = plan((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [2, 3] });
    return graph.unary('sqrt', graph.cast(input, 'f64'));
  });
  const typedLowering = lowerSimtPlan(typed, { fusion: 'exact-elementwise' });
  assert.equal(typedLowering.kernels.length, 1);
  assert.deepEqual(typedLowering.kernels[0].semanticNodes, ['node:0', 'node:1']);
  assert.match(typedLowering.source, /let fusedValue0 = gpu\.cast\.f64\(p0\[f0Input0\]\)/u);
  assert.match(typedLowering.source, /let fusedValue1 = gpu\.math\.sqrt\(fusedValue0\)/u);

  const broadcast = plan((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [2, 3] });
    const side = graph.input('side', { dtype: 'f32', capacityShape: [1, 3] });
    return graph.unary('neg', graph.binary('add', graph.unary('abs', left), side));
  });
  const broadcastLowering = lowerSimtPlan(broadcast, { fusion: 'exact-elementwise' });
  assert.equal(broadcastLowering.kernels.length, 1);
  assert.deepEqual(broadcastLowering.fusionProfile.regions[0].externalInputs, ['input:left', 'input:side']);
  assert.match(broadcastLowering.source, /let f1Input1 = gpu\.u64\(0n\);/u);
  assert.match(broadcastLowering.source, /f1Input1 \+= c1 \* gpu\.u64\(1n\);/u);
  assert.doesNotMatch(broadcastLowering.source, /f1Input1 \+= c0/u);
});

test('fusion composes with an adjacent cuBLASLt candidate without changing backend ownership', () => {
  const tensorPlan = plan((graph) => {
    const dense = graph.matmul(
      graph.input('left', { dtype: 'f32', capacityShape: [2, 3] }),
      graph.input('right', { dtype: 'f32', capacityShape: [3, 4] }),
    );
    return graph.unary('sqrt', graph.unary('abs', dense));
  });
  const lowering = lowerSimtPlan(tensorPlan, { fusion: 'exact-elementwise' });
  assert.equal(lowering.kernels.length, 2);
  assert.equal(lowering.kernels[0].semanticNode, 'node:0');
  assert.deepEqual(lowering.kernels[1].semanticNodes, ['node:1', 'node:2']);
  assert.deepEqual(lowering.kernels[1].after, ['kernel0']);
  const request = createBackendProfileRequest(tensorPlan, lowering, { backend: 'prefer-cublaslt' });
  assert.equal(request.candidates.length, 1);
  assert.equal(request.candidates[0].executionNodeId, 'kernel0');
  const profile = realizeBackendProfile(request, lowering, [{ semanticNode: 'node:0', backend: 'cublaslt' }]);
  assert.equal(profile.cublasLtNodeCount, 1);
  assert.equal(profile.simtNodeCount, 1);
});

test('a fused region is emitted at its final node after an independently produced side input', () => {
  const tensorPlan = plan((graph) => {
    const left = graph.unary('abs', graph.input('left', { dtype: 'f32', capacityShape: [4] }));
    const side = graph.unary('sqrt', graph.input('side', { dtype: 'f32', capacityShape: [4] }));
    return graph.binary('add', left, side);
  });
  const lowering = lowerSimtPlan(tensorPlan, { fusion: 'exact-elementwise' });
  assert.equal(lowering.fusionProfile.regions.length, 1);
  assert.deepEqual(lowering.fusionProfile.regions[0].nodeIds, ['node:0', 'node:2']);
  assert.equal(lowering.kernels.length, 2);
  assert.equal(lowering.kernels[0].semanticNode, 'node:1');
  assert.deepEqual(lowering.kernels[1].semanticNodes, ['node:0', 'node:2']);
  assert.deepEqual(lowering.kernels[1].after, ['kernel0']);
});

test('resolved fusion normalizes publicly and session admission uses realized materials', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open({
    runtime,
    runtimeOwnership: 'owned',
    limits: { maxTensorBytes: 16, maxSessionBytes: 32, maxLiveTensors: 2 },
  });
  const input = await session.allocate({ dtype: 'f32', capacityShape: [4], access: 'read' });
  const program = TensorProgram.define((graph) => {
    let value = graph.input('input', { dtype: 'f32', capacityShape: [4], access: 'read' });
    value = graph.unary('abs', value);
    value = graph.unary('sqrt', value);
    return graph.unary('neg', value);
  });

  await assert.rejects(resolveTensorPlanWithAdapter(session, program, {}, fakeAdapter()), { code: 'TENSOR_RESOLVE_SESSION_BYTE_LIMIT' });
  await assert.rejects(resolveTensorPlanWithAdapter(session, program, { fusion: 'unknown' }, fakeAdapter()), { code: 'TENSOR_FUSION_POLICY_UNSUPPORTED' });

  const observation = {};
  const resolved = await resolveTensorPlanWithAdapter(session, program, { fusion: 'exact-elementwise' }, fakeAdapter(observation));
  assert.equal(resolved.fusionPolicy, 'exact-elementwise');
  assert.equal(resolved.fusionRegionCount, 1);
  assert.equal(resolved.fusedNodeCount, 3);
  assert.equal(resolved.kernelCount, 1);
  assert.equal(resolved.bindingCount, 2);
  assert.equal(resolved.canonical.materialCount, 1);
  assert.equal(resolved.canonical.materialBytes, 16);
  assert.equal(resolved.canonical.options.fusion, 'exact-elementwise');

  const result = await resolved.run(input);
  assert.deepEqual(observation.bindings, ['b0', 'b1']);
  assert.equal(runtime.allocations.filter((allocation) => allocation.state === 'open').length, 2);
  await result.close();
  await resolved.close();
  await input.close();
  assert.equal((await session.close()).graceful, true);
});
