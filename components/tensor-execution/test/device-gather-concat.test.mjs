import assert from 'node:assert/strict';
import test from 'node:test';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { createDeviceItemProfile } from '../testing.mjs';

function profile(program, itemInputs) {
  return createDeviceItemProfile(TensorPlan.create(program), { itemCapacity: 4, itemInputs });
}

test('expected failure before #37 implementation: non-axis gather is item-callable', () => {
  const program = TensorProgram.define((graph) => {
    const items = graph.input('items', { dtype: 'f32', capacityShape: [4, 3, 4], access: 'read' });
    return graph.gather(items, 1, [2, 0, 2]);
  });
  const result = profile(program, ['items']);
  assert.match(result.canonical.contract, /SPEC-0009-gather-concat-v1$/u);
});

test('expected failure before #37 implementation: non-axis concat is item-callable', () => {
  const program = TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4, 3], access: 'read' });
    return graph.concat([left, right], 1);
  });
  const result = profile(program, ['left', 'right']);
  assert.match(result.canonical.contract, /SPEC-0009-gather-concat-v1$/u);
});
