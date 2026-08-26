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
  for (const frame of functions) {
    assert.throws(
      () => frame({}),
      (error) => error instanceof TypeError && /requires an accepted TensorPlan/u.test(error.message),
    );
  }

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
  assert.equal(result.candidates[0].coveredByCurrentSIMTSemantics, true);
  assert.equal(result.candidates[0].supportedByCurrentCublasLtProfile, false);
  assert.equal(result.candidates[0].structurallyEligibleForProposedCublasLtProfile, true);
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
  assert.equal(candidate.dimensions.batch, 3);
  assert.equal(candidate.coveredByCurrentSIMTSemantics, true);
  assert.equal(candidate.supportedByCurrentCublasLtProfile, false);
  assert.equal(candidate.structurallyEligibleForProposedCublasLtProfile, false);
});

test('batched matmul keeps widened accumulation and an empty contraction out of the proposed profile', () => {
  const widened = plan((graph) => graph.matmul(
    graph.input('left', { dtype: 'f32', capacityShape: [2, 3, 4] }),
    graph.input('right', { dtype: 'f32', capacityShape: [1, 4, 5] }),
    { accumulatorDtype: 'f64' },
  ));
  const widenedCandidate = frameBatchedMatmul(widened).candidates[0];
  assert.deepEqual(widenedCandidate.reasons, ['accumulator-not-f32']);
  assert.equal(widenedCandidate.coveredByCurrentSIMTSemantics, true);
  assert.equal(widenedCandidate.supportedByCurrentCublasLtProfile, false);
  assert.equal(widenedCandidate.structurallyEligibleForProposedCublasLtProfile, false);

  const emptyContraction = plan((graph) => graph.matmul(
    graph.input('left', { dtype: 'f32', capacityShape: [2, 3, 0] }),
    graph.input('right', { dtype: 'f32', capacityShape: [1, 0, 5] }),
  ));
  const emptyCandidate = frameBatchedMatmul(emptyContraction).candidates[0];
  assert.deepEqual(emptyCandidate.reasons, ['empty-contraction']);
  assert.equal(emptyCandidate.coveredByCurrentSIMTSemantics, true);
  assert.equal(emptyCandidate.supportedByCurrentCublasLtProfile, false);
  assert.equal(emptyCandidate.structurallyEligibleForProposedCublasLtProfile, false);
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
    assert.equal(candidate.coveredByCurrentSIMTSemantics, true);
    assert.equal(candidate.supportedByCurrentCublasLtProfile, false);
    assert.equal(candidate.structurallyEligibleForProposedProfile, true);
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
  assert.equal(candidate.structurallyEligibleForProposedProfile, true);
});

test('precision frame distinguishes SIMT support from a structurally empty proposed profile', () => {
  const tensorPlan = plan((graph) => graph.matmul(
    graph.input('left', { dtype: 'f16', capacityShape: [2, 0] }),
    graph.input('right', { dtype: 'f16', capacityShape: [0, 4] }),
  ));
  const candidate = framePrecisionMatmul(tensorPlan).candidates[0];
  assert.deepEqual(candidate.reasons, ['empty-contraction']);
  assert.equal(candidate.coveredByCurrentSIMTSemantics, true);
  assert.equal(candidate.supportedByCurrentCublasLtProfile, false);
  assert.equal(candidate.structurallyEligibleForProposedProfile, false);
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
  assert.equal(regions[0].matmulCount, 1);
  assert.deepEqual(regions[0].nodes.map(({ op }) => op), ['matmul', 'unary', 'reduce']);
  assert.deepEqual(regions[0].boundaryInputs, ['input:left', 'input:right']);
  assert.deepEqual(regions[0].boundaryOutputs, ['node:2']);
});

test('device-callable analysis excludes elementwise-only regions', () => {
  const tensorPlan = plan((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [4] });
    const root = graph.unary('abs', input);
    return { left: graph.unary('sqrt', root), right: graph.unary('neg', root) };
  });
  assert.deepEqual(frameDeviceCallableDense(tensorPlan).candidates, []);
});

test('device-callable analysis finds the complete connected matmul region across fan-out and join', () => {
  const tensorPlan = plan((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [2, 3] });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [3, 4] });
    const dense = graph.matmul(left, right);
    const positive = graph.unary('abs', dense);
    const negative = graph.unary('neg', dense);
    return graph.binary('add', positive, negative);
  });
  const regions = frameDeviceCallableDense(tensorPlan).candidates;
  assert.equal(regions.length, 1);
  assert.equal(regions[0].candidateSize, 4);
  assert.equal(regions[0].matmulCount, 1);
  assert.deepEqual(regions[0].nodes.map(({ id }) => id), ['node:0', 'node:1', 'node:2', 'node:3']);
  assert.deepEqual(regions[0].boundaryInputs, ['input:left', 'input:right']);
  assert.deepEqual(regions[0].boundaryOutputs, ['node:3']);
});

test('device-callable analysis retains a standalone matmul as an explicit candidate', () => {
  const tensorPlan = plan((graph) => graph.matmul(
    graph.input('left', { dtype: 'f32', capacityShape: [2, 3] }),
    graph.input('right', { dtype: 'f32', capacityShape: [3, 4] }),
  ));
  const candidate = frameDeviceCallableDense(tensorPlan).candidates[0];
  assert.equal(candidate.candidateSize, 1);
  assert.equal(candidate.matmulCount, 1);
  assert.deepEqual(candidate.boundaryInputs, ['input:left', 'input:right']);
  assert.deepEqual(candidate.boundaryOutputs, ['node:0']);
});

test('device-callable analysis exposes every observable output inside one connected region', () => {
  const tensorPlan = plan((graph) => {
    const dense = graph.matmul(
      graph.input('left', { dtype: 'f32', capacityShape: [2, 3] }),
      graph.input('right', { dtype: 'f32', capacityShape: [3, 4] }),
    );
    const activated = graph.unary('abs', dense);
    return { dense, activated };
  });
  const candidate = frameDeviceCallableDense(tensorPlan).candidates[0];
  assert.deepEqual(candidate.boundaryOutputs, ['node:0', 'node:1']);
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

test('fusion treats duplicate input edges to one node as one consumer', () => {
  const tensorPlan = plan((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [4] });
    const normalized = graph.unary('abs', input);
    return graph.binary('add', normalized, normalized);
  });
  const candidate = frameElementwiseFusion(tensorPlan).candidates[0];
  assert.equal(candidate.candidateSize, 2);
  assert.deepEqual(candidate.nodes.map(({ id }) => id), ['node:0', 'node:1']);
  assert.deepEqual(candidate.boundaryInputs, ['input:input']);
  assert.deepEqual(candidate.boundaryOutputs, ['node:1']);
});

test('arena first-fit reuses only non-overlapping lifetimes and reports exact savings', () => {
  const tensorPlan = plan((graph) => {
    const source = graph.fill({ dtype: 'f32', capacityShape: [4], access: 'read-write' }, 1);
    const middle = graph.unary('abs', source);
    return graph.unary('sqrt', middle);
  });
  const candidate = frameArenaReuse(tensorPlan).candidates[0];
  assert.equal(candidate.distinctMaterialBytes, 48);
  assert.equal(candidate.arenaBytes, 32);
  assert.equal(candidate.alignmentPaddingBytes, 0);
  assert.equal(candidate.netByteDelta, 16);
  assert.equal(candidate.reuseFactor, 1.5);
  assert.deepEqual(candidate.assignments.map(({ slotId }) => slotId), ['slot:0', 'slot:1', 'slot:0']);
  assert.deepEqual(candidate.assignments.map(({ reused }) => reused), [false, false, true]);
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
  assert.equal(candidate.distinctMaterialBytes, 0);
  assert.equal(candidate.arenaBytes, 0);
  assert.equal(candidate.alignmentPaddingBytes, 0);
  assert.equal(candidate.netByteDelta, 0);
  assert.equal(candidate.reuseFactor, 1);
  assert(Object.isFrozen(candidate));
});

test('arena closes material lifetimes over views before considering reuse', () => {
  const tensorPlan = plan((graph) => {
    const source = graph.fill({ dtype: 'f32', capacityShape: [4], access: 'read-write' }, 1);
    const view = graph.reshape(source, [4]);
    const later = graph.fill({ dtype: 'f32', capacityShape: [4], access: 'read-write' }, 2);
    return { view, later };
  });
  const candidate = frameArenaReuse(tensorPlan).candidates[0];
  assert.deepEqual(candidate.assignments.map(({ slotId }) => slotId), ['slot:0', 'slot:1']);
  assert.equal(candidate.assignments[0].directLastUse, 1);
  assert.equal(candidate.assignments[0].effectiveLastUse, 3);
  assert.equal(candidate.assignments[0].aliasClass, 'material:node:0');
  assert.equal(candidate.arenaBytes, 32);
  assert.equal(candidate.netByteDelta, 0);
});

test('arena reports alignment overhead as a signed net delta rather than a negative saving', () => {
  const tensorPlan = plan((graph) => ({
    narrow: graph.fill({ dtype: 'f16', capacityShape: [], access: 'read-write' }, 1),
    wide: graph.fill({ dtype: 'f64', capacityShape: [], access: 'read-write' }, 1),
  }));
  const candidate = frameArenaReuse(tensorPlan).candidates[0];
  assert.equal(candidate.distinctMaterialBytes, 10);
  assert.equal(candidate.alignmentPaddingBytes, 6);
  assert.equal(candidate.arenaBytes, 16);
  assert.equal(candidate.netByteDelta, -6);
  assert.equal(candidate.reuseFactor, 0.625);
});

test('arena rejects a safe TensorPlan whose aligned aggregate would exceed the safe integer range', () => {
  const maximumF64ElementsAfterScalar = Math.floor((Number.MAX_SAFE_INTEGER - 2) / 8);
  const tensorPlan = plan((graph) => ({
    narrow: graph.fill({ dtype: 'f16', capacityShape: [], access: 'read-write' }, 1),
    wide: graph.fill({
      dtype: 'f64',
      capacityShape: [maximumF64ElementsAfterScalar],
      access: 'read-write',
    }, 1),
  }));
  assert.throws(
    () => frameArenaReuse(tensorPlan),
    (error) => error instanceof RangeError && /arena byte length exceeds the safe integer range/u.test(error.message),
  );
});
