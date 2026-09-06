import assert from 'node:assert/strict';
import test from 'node:test';

import {
  TENSOR_PROGRAM_CONTRACT,
  TENSOR_PROGRAM_SPEC0010_CONTRACT,
  TensorProgram,
} from '../index.mjs';

function code(expected) { return (error) => error?.code === expected; }

function extensionProgram() {
  return TensorProgram.define((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [2, 5], access: 'read' });
    return graph.gather(input, 1, [4, 1, 4, 0]);
  });
}

test('rehydration rejects forged base/extension contracts and extension limits', () => {
  const extension = JSON.parse(JSON.stringify(extensionProgram().canonical));
  assert.equal(extension.contract, TENSOR_PROGRAM_SPEC0010_CONTRACT);

  const forgedBase = structuredClone(extension);
  forgedBase.contract = TENSOR_PROGRAM_CONTRACT;
  forgedBase.limits = { maxInputs: 256, maxNodes: 4096, maxOutputs: 256 };
  assert.throws(() => TensorProgram.create(forgedBase), code('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));

  const forgedLimits = structuredClone(extension);
  forgedLimits.limits.maxStaticGatherIndices = 22;
  assert.throws(() => TensorProgram.create(forgedLimits), code('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));

  const legacy = TensorProgram.define((graph) => graph.copy(graph.input('x', { dtype: 'f32', capacityShape: [2] })));
  const forgedExtension = JSON.parse(JSON.stringify(legacy.canonical));
  forgedExtension.contract = TENSOR_PROGRAM_SPEC0010_CONTRACT;
  forgedExtension.limits = { ...forgedExtension.limits, maxStaticGatherIndices: 65536, maxConcatInputs: 256 };
  assert.throws(() => TensorProgram.create(forgedExtension), code('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));
});

test('gather rejects unknown options, invalid axes and semantic metadata pressure before resolution', () => {
  assert.throws(() => TensorProgram.define((graph) => {
    const input = graph.input('x', { dtype: 'f32', capacityShape: [2, 3] });
    return graph.node('gather', [input], { axis: 1, indices: [0], unexpected: true });
  }), code('TENSOR_PROGRAM_OPERATION_OPTIONS_INVALID'));

  for (const axis of [-1, 1.5, 2]) {
    assert.throws(() => TensorProgram.define((graph) => graph.gather(
      graph.input('x', { dtype: 'f32', capacityShape: [2, 3] }),
      axis,
      [0],
    )), code('TENSOR_PROGRAM_AXIS_INVALID'));
  }

  for (const index of [-1, 1.5, Number.MAX_SAFE_INTEGER]) {
    assert.throws(() => TensorProgram.define((graph) => graph.gather(
      graph.input('x', { dtype: 'f32', capacityShape: [2, 3] }),
      1,
      [index],
    )), code('TENSOR_PROGRAM_GATHER_INDEX_INVALID'));
  }

  const tooMany = Array.from({ length: 65_537 }, () => 0);
  assert.throws(() => TensorProgram.define((graph) => graph.gather(
    graph.input('x', { dtype: 'f32', capacityShape: [1] }),
    0,
    tooMany,
  )), (error) => error?.code === 'TENSOR_PROGRAM_GATHER_INDICES_INVALID' && error?.category === 'pressure');
});

test('concat enforces its semantic arity but leaves narrower execution pressure downstream', () => {
  assert.throws(() => TensorProgram.define((graph) => {
    const input = graph.input('x', { dtype: 'f32', capacityShape: [1] });
    return graph.concat([input], 0);
  }), code('TENSOR_PROGRAM_OPERATION_INPUTS_INVALID'));

  assert.throws(() => TensorProgram.define((graph) => {
    const input = graph.input('x', { dtype: 'f32', capacityShape: [1] });
    return graph.concat(Array.from({ length: 257 }, () => input), 0);
  }), (error) => error?.code === 'TENSOR_PROGRAM_OPERATION_INPUTS_INVALID' && error?.category === 'pressure');
});

test('concat input order is semantic and changes canonical identity', () => {
  function make(reverse) {
    return TensorProgram.define((graph) => {
      const left = graph.input('left', { dtype: 'f32', capacityShape: [1, 2] });
      const right = graph.input('right', { dtype: 'f32', capacityShape: [1, 3] });
      return graph.concat(reverse ? [right, left] : [left, right], 1);
    });
  }
  const forward = make(false);
  const reverse = make(true);
  assert.notEqual(forward.compatibilityIdentity, reverse.compatibilityIdentity);
  assert.deepEqual(forward.nodes[0].inputIds, ['input:left', 'input:right']);
  assert.deepEqual(reverse.nodes[0].inputIds, ['input:right', 'input:left']);
});
