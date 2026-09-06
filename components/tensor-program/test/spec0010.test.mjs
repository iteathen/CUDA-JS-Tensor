import assert from 'node:assert/strict';
import test from 'node:test';

import {
  TensorPlan,
  TensorProgram,
} from '../../public-api/index.mjs';
import {
  TENSOR_PROGRAM_CONTRACT,
  TENSOR_PROGRAM_SPEC0010_CONTRACT,
  TENSOR_PROGRAM_SPEC0010_LIMITS,
} from '../src/program.mjs';

function expectCode(code) { return (error) => error?.code === code; }

test('legacy programs retain the base contract and exact pre-extension limit projection', () => {
  const legacy = TensorProgram.define((graph) => graph.unary('sqrt', graph.input('x', { dtype: 'f32', capacityShape: [2, 3] })));
  assert.equal(legacy.contract, TENSOR_PROGRAM_CONTRACT);
  assert.deepEqual(legacy.canonical.limits, { maxInputs: 256, maxNodes: 4096, maxOutputs: 256 });
  assert.equal(Object.hasOwn(legacy.canonical.limits, 'maxStaticGatherIndices'), false);
  assert.equal(legacy.compatibilityIdentity, 'tensor-program-v1:a114d0722e8245b86c6284da0225766d9ba65fa072eaa629b278e684efd2aaaf');
  assert.equal(TensorProgram.create(JSON.parse(JSON.stringify(legacy.canonical))).compatibilityIdentity, legacy.compatibilityIdentity);
});

test('unary erf admits only same-kind f32/f64 and selects the extension contract', () => {
  for (const dtype of ['f32', 'f64']) {
    const program = TensorProgram.define((graph) => graph.unary('erf', graph.input('x', { dtype, capacityShape: [2, 3] })));
    assert.equal(program.contract, TENSOR_PROGRAM_SPEC0010_CONTRACT);
    assert.deepEqual(program.canonical.limits, { ...TENSOR_PROGRAM_SPEC0010_LIMITS });
    assert.equal(program.outputs[0].spec.dtype, dtype);
    assert.deepEqual(program.outputs[0].spec.capacityShape, [2, 3]);
    assert.equal(program.nodes[0].materialization, 'materialize');
    assert.equal(TensorProgram.create(JSON.parse(JSON.stringify(program.canonical))).compatibilityIdentity, program.compatibilityIdentity);
  }
  for (const dtype of ['u32', 'i32', 'u64', 'f16', 'bf16']) {
    assert.throws(() => TensorProgram.define((graph) => graph.unary('erf', graph.input('x', { dtype, capacityShape: [1] }))), expectCode('TENSOR_PROGRAM_DTYPE_UNSUPPORTED'));
  }
});

test('static gather preserves order, duplicates, empties and active non-axis-0 extent', () => {
  const program = TensorProgram.define((graph) => {
    const input = graph.input('x', { dtype: 'f32', capacityShape: [8, 5, 3], activeAxis0: 6 });
    return {
      duplicate: graph.gather(input, 1, [4, 1, 4, 0]),
      empty: graph.gather(input, 2, []),
    };
  });
  assert.deepEqual(program.nodes[0].options, { axis: 1, indices: [4, 1, 4, 0] });
  assert.deepEqual(program.outputs[0].spec.capacityShape, [8, 4, 3]);
  assert.deepEqual(program.outputs[0].spec.activeAxis0, { extent: 6, maximum: 8 });
  assert.deepEqual(program.outputs[1].spec.capacityShape, [8, 5, 0]);
  assert.equal(program.nodes[0].materialization, 'materialize');

  assert.throws(() => TensorProgram.define((graph) => graph.gather(graph.input('x', { dtype: 'f32', capacityShape: [2, 3] }), 1, [3])), expectCode('TENSOR_PROGRAM_GATHER_INDEX_INVALID'));
  assert.throws(() => TensorProgram.define((graph) => graph.gather(graph.input('x', { dtype: 'f32', capacityShape: [2, 3], activeAxis0: 1 }), 0, [0])), expectCode('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE'));
});

test('ordered concat sums only its selected axis and preserves matching active extent', () => {
  const program = TensorProgram.define((graph) => {
    const a = graph.input('a', { dtype: 'f32', capacityShape: [8, 2, 3], activeAxis0: 5 });
    const b = graph.input('b', { dtype: 'f32', capacityShape: [8, 4, 3], activeAxis0: 5 });
    const c = graph.input('c', { dtype: 'f32', capacityShape: [8, 1, 3], activeAxis0: 5 });
    return graph.concat([a, b, c], 1);
  });
  assert.deepEqual(program.outputs[0].spec.capacityShape, [8, 7, 3]);
  assert.deepEqual(program.outputs[0].spec.activeAxis0, { extent: 5, maximum: 8 });
  assert.deepEqual(program.nodes[0].options, { axis: 1 });
  assert.equal(program.nodes[0].materialization, 'materialize');

  assert.throws(() => TensorProgram.define((graph) => graph.concat([
    graph.input('a', { dtype: 'f32', capacityShape: [2, 3] }),
    graph.input('b', { dtype: 'f64', capacityShape: [2, 3] }),
  ], 1)), expectCode('TENSOR_PROGRAM_DTYPE_MISMATCH'));
  assert.throws(() => TensorProgram.define((graph) => graph.concat([
    graph.input('a', { dtype: 'f32', capacityShape: [2, 3] }),
    graph.input('b', { dtype: 'f32', capacityShape: [3, 3] }),
  ], 1)), expectCode('TENSOR_PROGRAM_CONCAT_SHAPE_INVALID'));
  assert.throws(() => TensorProgram.define((graph) => graph.concat([
    graph.input('a', { dtype: 'f32', capacityShape: [8, 2], activeAxis0: 4 }),
    graph.input('b', { dtype: 'f32', capacityShape: [8, 3], activeAxis0: 5 }),
  ], 1)), expectCode('TENSOR_PROGRAM_ACTIVE_AXIS_MISMATCH'));
  assert.throws(() => TensorProgram.define((graph) => graph.concat([
    graph.input('a', { dtype: 'f32', capacityShape: [8, 2], activeAxis0: 4 }),
    graph.input('b', { dtype: 'f32', capacityShape: [8, 2], activeAxis0: 4 }),
  ], 0)), expectCode('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE'));
});

test('TensorPlan treats SPEC-0010 outputs as ordinary bounded material allocations', () => {
  const program = TensorProgram.define((graph) => {
    const source = graph.input('source', { dtype: 'f32', capacityShape: [2, 4] });
    const gathered = graph.gather(source, 1, [3, 1, 1]);
    const gaussian = graph.unary('erf', gathered);
    return graph.concat([gaussian, gaussian], 1);
  });
  const plan = TensorPlan.create(program);
  assert.equal(plan.allocations.length, 3);
  assert.deepEqual(plan.operations.map((entry) => entry.op), ['gather', 'unary', 'concat']);
  assert.deepEqual(plan.allocations.map((entry) => entry.byteLength), [24, 24, 48]);
  assert.equal(plan.totalDistinctBytes, 96);
  assert(plan.allocations.every((entry) => entry.reuse === 'none'));
});
