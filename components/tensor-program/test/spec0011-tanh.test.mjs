import assert from 'node:assert/strict';
import test from 'node:test';

import { TensorProgram } from '../../public-api/index.mjs';

const TANH_CONTRACT = 'SPEC-0004-tensor-program-v1+SPEC-0011-tanh-v1';

test('SPEC-0011 f32 unary:tanh selects its additive TensorProgram contract', () => {
  const program = TensorProgram.define((graph) => graph.unary('tanh', graph.input('x', { dtype: 'f32', capacityShape: [2, 3] })));
  assert.equal(program.contract, TANH_CONTRACT);
  assert.deepEqual(program.canonical.limits, { maxInputs: 256, maxNodes: 4096, maxOutputs: 256 });
  assert.equal(program.outputs[0].spec.dtype, 'f32');
  assert.deepEqual(program.outputs[0].spec.capacityShape, [2, 3]);
  assert.equal(program.nodes[0].materialization, 'materialize');
});
