import assert from 'node:assert/strict';
import test from 'node:test';
import { TensorPlan, TensorProgram } from '../../public-api/index.mjs';

function expectCode(code) { return (error) => error?.code === code; }

test('static plan exposes exact liveness, alias inheritance and distinct allocation bounds', () => {
  const program = TensorProgram.define((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [2, 3], aliasGroup: 'caller-declared' });
    const view = graph.permute(input, [1, 0]);
    const dense = graph.contiguous(view);
    const result = graph.unary('abs', dense);
    return { view, result };
  });
  const plan = TensorPlan.create(program);
  assert.equal(plan.executable, false);
  assert.equal(plan.allocationPolicy, 'distinct-v1');
  assert.deepEqual(plan.liveness, [
    { value: 'input:input', definition: 'input', definedAt: -1, lastUse: 0, observable: false },
    { value: 'node:0', definition: 'permute', definedAt: 0, lastUse: 3, observable: true },
    { value: 'node:1', definition: 'contiguous', definedAt: 1, lastUse: 2, observable: false },
    { value: 'node:2', definition: 'unary', definedAt: 2, lastUse: 3, observable: true },
  ]);
  assert.equal(plan.aliases[0].aliasClass, 'declared:caller-declared');
  assert.equal(plan.aliases[1].aliasClass, plan.aliases[0].aliasClass);
  assert.equal(plan.aliases[2].aliasClass, 'material:node:1');
  assert.equal(plan.aliases[3].aliasClass, 'material:node:2');
  assert.deepEqual(plan.allocations.map((entry) => ({ value: entry.value, bytes: entry.byteLength, lifetime: entry.lifetime })), [
    { value: 'node:1', bytes: 24, lifetime: { definedAt: 1, lastUse: 2 } },
    { value: 'node:2', bytes: 24, lifetime: { definedAt: 2, lastUse: 3 } },
  ]);
  assert.equal(plan.totalDistinctBytes, 48);
  assert(plan.unresolved.includes('backend-selection'));
  assert(plan.unresolved.includes('runtime-input-aliasing'));
  assert(Object.isFrozen(plan.canonical));
  assert.deepEqual(Object.keys(plan), []);
});

test('empty material values retain explicit dtype-width physical allocation bounds', () => {
  const program = TensorProgram.define((graph) => graph.fill({ dtype: 'f64', capacityShape: [0], access: 'read-write' }, 0));
  const plan = TensorPlan.create(program);
  assert.equal(program.outputs[0].spec.byteLength, 0);
  assert.equal(plan.allocations[0].byteLength, 8);
  assert.equal(plan.totalDistinctBytes, 8);
});

test('input-only programs remain finite and defer runtime alias proof', () => {
  const program = TensorProgram.define((graph) => graph.input('input', { dtype: 'u32', capacityShape: [4] }));
  const plan = TensorPlan.create(program);
  assert.equal(plan.operations.length, 0);
  assert.equal(plan.allocations.length, 0);
  assert.equal(plan.totalDistinctBytes, 0);
  assert.equal(plan.aliases[0].runtimeInputAliasingUnresolved, true);
  assert.equal(plan.outputs[0].aliasClass, 'unresolved-input:input');
});

test('plan identity is deterministic and unsupported optimization policy rejects', () => {
  const make = () => TensorProgram.define((graph) => graph.copy(graph.input('x', { dtype: 'f32', capacityShape: [4] })));
  const left = TensorPlan.create(make());
  const right = TensorPlan.create(make(), { allocationPolicy: 'distinct-v1' });
  assert.equal(left.compatibilityIdentity, right.compatibilityIdentity);
  assert.throws(() => TensorPlan.create(make(), { allocationPolicy: 'arena-reuse' }), expectCode('TENSOR_PLAN_ALLOCATION_POLICY_UNSUPPORTED'));
  assert.throws(() => TensorPlan.create({}, {}), expectCode('TENSOR_PLAN_PROGRAM_INVALID'));
});
