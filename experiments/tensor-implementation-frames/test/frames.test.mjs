import assert from 'node:assert/strict';
import test from 'node:test';

import { TensorPlan, TensorProgram } from '../../../components/public-api/index.mjs';
import {
  frameArenaReuse,
  frameBatchedMatmul,
  frameDeviceCallableDense,
  frameElementwiseFusion,
  framePrecisionMatmul,
} from '../src/index.mjs';
import { TENSOR_IMPLEMENTATION_FRAME_CONTRACT } from '../src/frame-contract.mjs';

function plan(callback) {
  return TensorPlan.create(TensorProgram.define(callback));
}

test('all frames reject non-plans and return deterministic deeply immutable analysis records', () => {
  const functions = [
    frameBatchedMatmul,
    framePrecisionMatmul,
    frameDeviceCallableDense,
    frameElementwiseFusion,
    frameArenaReuse,
  ];
  for (const frame of functions) assert.throws(() => frame({}), TypeError);

  const tensorPlan = plan((graph) => graph.input('x', { dtype: 'f32', capacityShape: [2] }));
  for (const frame of functions) {
    const first = frame(tensorPlan);
    const second = frame(tensorPlan);
    assert.equal(first.contract, TENSOR_IMPLEMENTATION_FRAME_CONTRACT);
    assert.equal(first.status, 'analysis-only');
    assert.equal(first.planIdentity, tensorPlan.compatibilityIdentity);
    assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
    assert.match(first.compatibilityIdentity, /^tensor-implementation-frame-v1:[0-9a-f]{64}$/u);
    assert(Object.isFrozen(first));
    assert(Object.isFrozen(first.candidates));
    assert(Object.isFrozen(first.blockers));
    assert(Object.isFrozen(first.completion));
  }
});

test('batched matmul reports exact broadcast and transpose dimensions without false blockers', () => {
  const tensorPlan = plan((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [2, 3, 4] });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [1, 5, 4] });
    return graph.matmul(left, right, { transposeB: true });
  });

  const result = frameBatchedMatmul(tensorPlan);
  assert.equal(result.candidates.length, 1);
  assert.deepEqual(result.candidates[0].dimensions, {
    batchA: 2,
    batchB: 1,
    batch: 2,
    m: 3,
    k: 4,
    n: 5,
  });
  assert.deepEqual(result.candidates[0].reasons, []);
  assert.equal(result.candidates[0].eligibleForCurrentSIMT, true);
  assert.equal(result.candidates[0].eligibleForProposedCublasLtProfile, true);
});

test('batched matmul classifies active extents, derived layouts, and empty output deterministically', () => {
  const tensorPlan = plan((graph) => {
    const source = graph.input('source', {
      dtype: 'f32',
      capacityShape: [3, 4, 2],
      activeAxis0: 2,
      strides: [8, 1, 4],
      access: 'read',
    });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [1, 2, 0] });
    return graph.matmul(source, right);
  });

  const candidate = frameBatchedMatmul(tensorPlan).candidates[0];
  assert.deepEqual(candidate.reasons, ['layout-not-contiguous', 'active-batch-extent', 'empty-output']);
  assert.equal(candidate.eligibleForCurrentSIMT, false);
  assert.equal(candidate.eligibleForProposedCublasLtProfile, false);
});

test('precision matmul distinguishes supported candidate semantics from provider implementation gaps', () => {
  for (const dtype of ['f16', 'bf16', 'f64']) {
    const tensorPlan = plan((graph) => graph.matmul(
      graph.input('left', { dtype, capacityShape: [2, 3] }),
      graph.input('right', { dtype, capacityShape: [3, 4] }),
    ));
    const candidate = framePrecisionMatmul(tensorPlan).candidates[0];
    assert.equal(candidate.dtype, dtype);
    assert.equal(candidate.rank, 2);
    assert.equal(candidate.rankCompatible, true);
    assert.deepEqual(candidate.reasons, []);
    assert.equal(candidate.contractReady, true);
    assert.match(candidate.implementationGap, /public CUDA-JS typed matmul/u);
  }
});

test('precision frame ignores current f32 work and records widening accumulation explicitly', () => {
  const current = plan((graph) => graph.matmul(
    graph.input('left', { dtype: 'f32', capacityShape: [2, 3] }),
    graph.input('right', { dtype: 'f32', capacityShape: [3, 4] }),
  ));
  assert.deepEqual(framePrecisionMatmul(current).candidates, []);

  const widened = plan((graph) => graph.matmul(
    graph.input('left', { dtype: 'f16', capacityShape: [2, 3] }),
    graph.input('right', { dtype: 'f16', capacityShape: [3, 4] }),
    { accumulatorDtype: 'f64' },
  ));
  const candidate = framePrecisionMatmul(widened).candidates[0];
  assert.equal(candidate.accumulatorDtype, 'f64');
  assert.deepEqual(candidate.reasons, []);
});

test('device-callable analysis finds maximal dense regions and exact external boundaries', () => {
  const tensorPlan = plan((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [2, 3] });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [3, 4] });
    const dense = graph.matmul(left, right);
    const activated = graph.unary('abs', dense);
    return graph.reduce('sum', activated, { axes: [1] });
  });
  const regions = frameDeviceCallableDense(tensorPlan).candidates;
  assert.equal(regions.length, 1);
  assert.equal(regions[0].candidateSize, 3);
  assert.deepEqual(regions[0].nodes.map(({ op }) => op), ['matmul', 'unary', 'reduce']);
  assert.deepEqual(regions[0].boundaryInputs, ['input:left', 'input:right']);
  assert.deepEqual(regions[0].boundaryOutputs, ['node:2']);
});

test('device-callable analysis does not combine across a fan-out boundary', () => {
  const tensorPlan = plan((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [4] });
    const root = graph.unary('abs', input);
    return { left: graph.unary('sqrt', root), right: graph.unary('neg', root) };
  });
  assert.deepEqual(frameDeviceCallableDense(tensorPlan).candidates, []);
});

test('fusion includes a binary side input while preserving both external boundaries', () => {
  const tensorPlan = plan((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4] });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4] });
    const normalized = graph.unary('abs', left);
    const combined = graph.binary('add', normalized, right);
    return graph.unary('sqrt', combined);
  });
  const chains = frameElementwiseFusion(tensorPlan).candidates;
  assert.equal(chains.length, 1);
  assert.equal(chains[0].candidateSize, 3);
  assert.equal(chains[0].hasBinary, true);
  assert.deepEqual(chains[0].nodes.map(({ op }) => op), ['unary', 'binary', 'unary']);
  assert.deepEqual(chains[0].boundaryInputs, ['input:left', 'input:right']);
  assert.deepEqual(chains[0].boundaryOutputs, ['node:2']);
});

test('fusion stops at observable intermediates and fan-out boundaries', () => {
  const tensorPlan = plan((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [4] });
    const observable = graph.unary('abs', input);
    const next = graph.unary('sqrt', observable);
    const tail = graph.unary('neg', next);
    return { observable, tail };
  });
  const chains = frameElementwiseFusion(tensorPlan).candidates;
  assert.equal(chains.length, 1);
  assert.deepEqual(chains[0].nodes.map(({ id }) => id), ['node:1', 'node:2']);
  assert.deepEqual(chains[0].boundaryInputs, ['node:0']);
  assert.deepEqual(chains[0].boundaryOutputs, ['node:2']);
});

test('arena first-fit reuses only non-overlapping lifetimes and reports exact savings', () => {
  const tensorPlan = plan((graph) => {
    const source = graph.fill({ dtype: 'f32', capacityShape: [4], access: 'read-write' }, 1);
    const middle = graph.unary('abs', source);
    return graph.unary('sqrt', middle);
  });
  const candidate = frameArenaReuse(tensorPlan).candidates[0];
  assert.equal(candidate.totalMaterialBytes, 48);
  assert.equal(candidate.arenaBytes, 32);
  assert.equal(candidate.potentialSavingBytes, 16);
  assert.equal(candidate.utilization, 1.5);
  assert.deepEqual(candidate.assignments.map(({ slotId }) => slotId), ['slot:0', 'slot:1', 'slot:0']);
  assert.deepEqual(candidate.slots, [
    { id: 'slot:0', offset: 0, byteLength: 16 },
    { id: 'slot:1', offset: 16, byteLength: 16 },
  ]);
});

test('arena analysis remains finite and immutable for an allocation-free plan', () => {
  const tensorPlan = plan((graph) => graph.input('input', { dtype: 'f32', capacityShape: [4] }));
  const candidate = frameArenaReuse(tensorPlan).candidates[0];
  assert.deepEqual(candidate.ordered, []);
  assert.deepEqual(candidate.slots, []);
  assert.deepEqual(candidate.assignments, []);
  assert.equal(candidate.arenaBytes, 0);
  assert.equal(candidate.potentialSavingBytes, 0);
  assert.equal(candidate.utilization, 1);
  assert(Object.isFrozen(candidate));
});
