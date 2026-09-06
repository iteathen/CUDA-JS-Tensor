import assert from 'node:assert/strict';
import test from 'node:test';

import {
  TENSOR_PROGRAM_SPEC0010_CONTRACT,
  TENSOR_PROGRAM_SPEC0010_SPEC0011_CONTRACT,
  TENSOR_PROGRAM_SPEC0011_CONTRACT,
  TensorPlan,
  TensorProgram,
} from '../../public-api/index.mjs';

const BASE = 'SPEC-0004-tensor-program-v1';
const BASE_LIMITS = { maxInputs: 256, maxNodes: 4096, maxOutputs: 256 };
const SPEC0010_LIMITS = { ...BASE_LIMITS, maxStaticGatherIndices: 65536, maxConcatInputs: 256 };

function code(expected) {
  return (error) => error?.code === expected;
}

test('SPEC-0011 f32 unary:tanh selects its additive TensorProgram contract and exact normalized semantics', () => {
  const program = TensorProgram.define((graph) => graph.unary('tanh', graph.input('x', { dtype: 'f32', capacityShape: [4, 3], activeAxis0: { maximum: 4, current: 2 } })));
  assert.equal(program.contract, TENSOR_PROGRAM_SPEC0011_CONTRACT);
  assert.deepEqual(program.canonical.limits, BASE_LIMITS);
  assert.deepEqual(program.nodes[0].options, {
    operator: 'tanh',
    arithmetic: 'round-to-output-dtype-v1',
    specialValues: 'ieee-nan-propagate-signed-zero-v1',
  });
  assert.equal(program.outputs[0].spec.dtype, 'f32');
  assert.deepEqual(program.outputs[0].spec.capacityShape, [4, 3]);
  assert.deepEqual(program.outputs[0].spec.activeAxis0, { maximum: 4, current: 2 });
  assert.equal(program.outputs[0].spec.layout, 'row-major-contiguous');
  assert.equal(program.nodes[0].materialization, 'materialize');
  const roundTrip = TensorProgram.create(program.canonical);
  assert.deepEqual(roundTrip.canonical, program.canonical);
  assert.equal(roundTrip.compatibilityIdentity, program.compatibilityIdentity);
});

test('SPEC-0011 admits f64 but rejects f16, bf16 and integer tanh', () => {
  const f64 = TensorProgram.define((graph) => graph.unary('tanh', graph.input('x', { dtype: 'f64', capacityShape: [2, 2] })));
  assert.equal(f64.contract, TENSOR_PROGRAM_SPEC0011_CONTRACT);
  assert.equal(f64.outputs[0].spec.dtype, 'f64');
  for (const dtype of ['f16', 'bf16', 'u32', 'u64', 'i32']) {
    assert.throws(() => TensorProgram.define((graph) => graph.unary('tanh', graph.input('x', { dtype, capacityShape: [2] }))), code('TENSOR_PROGRAM_DTYPE_UNSUPPORTED'), dtype);
  }
});

test('pre-tanh base and SPEC-0010-only identities remain exact while mixed use selects canonical SPEC-0010 then SPEC-0011 order', () => {
  const base = TensorProgram.define((graph) => graph.unary('sqrt', graph.input('x', { dtype: 'f32', capacityShape: [2] })));
  assert.equal(base.contract, BASE);
  assert.deepEqual(base.canonical.limits, BASE_LIMITS);
  assert.doesNotMatch(JSON.stringify(base.canonical), /SPEC-0011-tanh-v1/u);

  const spec0010 = TensorProgram.define((graph) => graph.unary('erf', graph.input('x', { dtype: 'f32', capacityShape: [2] })));
  assert.equal(spec0010.contract, TENSOR_PROGRAM_SPEC0010_CONTRACT);
  assert.deepEqual(spec0010.canonical.limits, SPEC0010_LIMITS);
  assert.doesNotMatch(spec0010.contract, /SPEC-0011/u);

  const mixed = TensorProgram.define((graph) => {
    const x = graph.input('x', { dtype: 'f32', capacityShape: [2] });
    return graph.unary('tanh', graph.unary('erf', x));
  });
  assert.equal(mixed.contract, TENSOR_PROGRAM_SPEC0010_SPEC0011_CONTRACT);
  assert.equal(mixed.contract, 'SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1+SPEC-0011-tanh-v1');
  assert.deepEqual(mixed.canonical.limits, SPEC0010_LIMITS);
  assert.equal(TensorProgram.create(mixed.canonical).compatibilityIdentity, mixed.compatibilityIdentity);
});

test('SPEC-0011 contract and limits are derived and forged, omitted, reordered or unnecessary children fail closed', () => {
  const tanh = TensorProgram.define((graph) => graph.unary('tanh', graph.input('x', { dtype: 'f32', capacityShape: [2] })));
  assert.throws(() => TensorProgram.create({ ...tanh.canonical, contract: BASE }), code('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));
  assert.throws(() => TensorProgram.create({ ...tanh.canonical, limits: SPEC0010_LIMITS }), code('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));

  const mixed = TensorProgram.define((graph) => {
    const x = graph.input('x', { dtype: 'f32', capacityShape: [2] });
    return graph.unary('tanh', graph.unary('erf', x));
  });
  assert.throws(() => TensorProgram.create({ ...mixed.canonical, contract: TENSOR_PROGRAM_SPEC0010_CONTRACT }), code('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));
  assert.throws(() => TensorProgram.create({ ...mixed.canonical, contract: 'SPEC-0004-tensor-program-v1+SPEC-0011-tanh-v1+SPEC-0010-erf-gather-concat-v1' }), code('TENSOR_PROGRAM_OPTIONS_INVALID'));

  const base = TensorProgram.define((graph) => graph.copy(graph.input('x', { dtype: 'f32', capacityShape: [2] })));
  assert.throws(() => TensorProgram.create({ ...base.canonical, contract: TENSOR_PROGRAM_SPEC0011_CONTRACT }), code('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));
});

test('SPEC-0011 tanh remains an ordinary material in TensorPlan with no new resource owner', () => {
  const program = TensorProgram.define((graph) => graph.unary('tanh', graph.input('x', { dtype: 'f32', capacityShape: [2, 3] })));
  const plan = TensorPlan.create(program);
  assert.equal(plan.operations.length, 1);
  assert.equal(plan.operations[0].op, 'unary');
  assert.equal(plan.allocations.length, 1);
  assert.equal(plan.totalDistinctBytes, 24);
});
