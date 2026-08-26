import assert from 'node:assert/strict';
import test from 'node:test';
import { TensorProgram, TensorSpec } from '../../public-api/index.mjs';

function expectCode(code) { return (error) => error?.code === code; }

test('canonical record and callback builder normalize to one immutable program identity', () => {
  const leftSpec = TensorSpec.create('f32', [2, 3]);
  const rightSpec = TensorSpec.create('f32', [1, 3]);
  const built = TensorProgram.define((graph) => {
    const left = graph.input('left', leftSpec);
    const right = graph.input('right', rightSpec);
    return { sum: graph.binary('add', left, right) };
  });
  const canonical = TensorProgram.create({
    inputs: [{ name: 'left', spec: leftSpec }, { name: 'right', spec: rightSpec }],
    nodes: [{ op: 'binary', inputs: ['input:left', 'input:right'], options: { operator: 'add' } }],
    outputs: [{ name: 'sum', value: 'node:0' }],
  });
  assert.equal(built.compatibilityIdentity, canonical.compatibilityIdentity);
  assert.equal(TensorProgram.create(JSON.parse(JSON.stringify(built.canonical))).compatibilityIdentity, built.compatibilityIdentity);
  assert.deepEqual(built.outputs[0].spec.capacityShape, [2, 3]);
  assert.equal(built.nodes[0].materialization, 'materialize');
  assert.equal(built.valueSpec('node:0'), built.outputs[0].spec);
  assert.deepEqual(Object.keys(built), []);
  assert(Object.isFrozen(built.canonical));
  assert(Object.isFrozen(built.canonical.nodes));
  const corrupted = JSON.parse(JSON.stringify(built.canonical));
  corrupted.nodes[0].outputSpec.capacityShape[0] = 99;
  assert.throws(() => TensorProgram.create(corrupted), expectCode('TENSOR_PROGRAM_CANONICAL_DIVERGENCE'));
});

test('view operations infer exact shape, stride, offset, access and alias declarations', () => {
  const program = TensorProgram.define((graph) => {
    const input = graph.input('input', { dtype: 'f32', capacityShape: [4, 6], aliasGroup: 'source' });
    return {
      reshaped: graph.reshape(input, [3, 8]),
      transposed: graph.permute(input, [1, 0]),
      sliced: graph.slice(input, [{ start: 1, length: 2 }, { start: 0, length: 3, step: 2 }]),
      broadcast: graph.broadcast(graph.reshape(input, [1, 24]), [5, 24]),
    };
  });
  const [reshaped, transposed, sliced, broadcast] = program.outputs.map((entry) => entry.spec);
  assert.deepEqual(reshaped.capacityShape, [3, 8]);
  assert.deepEqual(reshaped.strides, [8, 1]);
  assert.deepEqual(transposed.capacityShape, [6, 4]);
  assert.deepEqual(transposed.strides, [1, 6]);
  assert.deepEqual(sliced.capacityShape, [2, 3]);
  assert.deepEqual(sliced.strides, [6, 2]);
  assert.equal(sliced.byteOffset, 24);
  assert.deepEqual(broadcast.capacityShape, [5, 24]);
  assert.deepEqual(broadcast.strides, [0, 1]);
  assert.equal(broadcast.access, 'read');
  assert(program.nodes.slice(0, 4).every((node) => node.materialization === 'view'));

  const emptySlice = TensorProgram.define((graph) => graph.slice(
    graph.input('empty', { dtype: 'f32', capacityShape: [0, 4] }),
    [null, { start: 4, length: 0 }],
  ));
  assert.equal(emptySlice.outputs[0].spec.byteOffset, 0);
  assert.equal(emptySlice.outputs[0].spec.byteLength, 0);
});

test('elementwise semantics use explicit dtypes, finite broadcasting and active-axis compatibility', () => {
  const program = TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [2, 1, 4], activeAxis0: 1 });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [1, 3, 4] });
    const sum = graph.binary('add', left, right);
    return { root: graph.unary('sqrt', sum) };
  });
  assert.deepEqual(program.outputs[0].spec.capacityShape, [2, 3, 4]);
  assert.deepEqual(program.outputs[0].spec.activeAxis0, { extent: 1, maximum: 2 });
  assert.equal(program.nodes[0].options.arithmetic, 'round-to-output-dtype-v1');

  const empty = TensorProgram.define((graph) => {
    const a = graph.input('a', { dtype: 'i32', capacityShape: [0, 3] });
    const b = graph.input('b', { dtype: 'i32', capacityShape: [1, 3] });
    return graph.binary('mul', a, b);
  });
  assert.deepEqual(empty.outputs[0].spec.capacityShape, [0, 3]);

  assert.throws(() => TensorProgram.define((graph) => graph.binary('add', graph.input('a', { dtype: 'f32', capacityShape: [2] }), graph.input('b', { dtype: 'i32', capacityShape: [2] }))), expectCode('TENSOR_PROGRAM_DTYPE_MISMATCH'));
  assert.throws(() => TensorProgram.define((graph) => graph.binary('div', graph.input('a', { dtype: 'i32', capacityShape: [2] }), graph.input('b', { dtype: 'i32', capacityShape: [2] }))), expectCode('TENSOR_PROGRAM_DTYPE_UNSUPPORTED'));
  assert.throws(() => TensorProgram.define((graph) => graph.unary('sqrt', graph.input('a', { dtype: 'u32', capacityShape: [2] }))), expectCode('TENSOR_PROGRAM_DTYPE_UNSUPPORTED'));
});

test('reductions normalize axes, active extent, accumulator, identity and order', () => {
  const program = TensorProgram.define((graph) => {
    const input = graph.input('input', { dtype: 'f16', capacityShape: [8, 4, 2], activeAxis0: 5 });
    return {
      sum: graph.reduce('sum', input, { axes: [2, 1] }),
      minimum: graph.reduce('minimum', input, { axes: [0], keepDimensions: true, order: 'backend-defined' }),
    };
  });
  const sum = program.outputs[0].spec;
  assert.equal(sum.dtype, 'f32');
  assert.deepEqual(sum.capacityShape, [8]);
  assert.deepEqual(sum.activeAxis0, { extent: 5, maximum: 8 });
  assert.deepEqual(program.nodes[0].options.axes, [1, 2]);
  assert.deepEqual(program.nodes[0].options.identity, { dtype: 'f32', value: 0 });
  assert.deepEqual(program.outputs[1].spec.capacityShape, [1, 4, 2]);
  assert.equal(program.outputs[1].spec.activeAxis0, null);
  assert.deepEqual(program.nodes[1].options.identity, { dtype: 'f32', value: 'Infinity' });
  assert.equal(program.nodes[1].options.order, 'backend-defined');

  assert.throws(() => TensorProgram.define((graph) => graph.reduce('sum', graph.input('x', { dtype: 'f32', capacityShape: [2, 3] }), { axes: [1, 1] })), expectCode('TENSOR_PROGRAM_REDUCTION_AXES_INVALID'));
  assert.throws(() => TensorProgram.define((graph) => graph.reduce('sum', graph.input('x', { dtype: 'i32', capacityShape: [2] }), { accumulatorDtype: 'f32' })), expectCode('TENSOR_PROGRAM_ACCUMULATOR_INVALID'));
});

test('rank-2 and rank-3 matmul infer transpose, batch broadcast and active batch bounds', () => {
  const rank2 = TensorProgram.define((graph) => {
    const a = graph.input('a', { dtype: 'f32', capacityShape: [5, 3], activeAxis0: 2 });
    const b = graph.input('b', { dtype: 'f32', capacityShape: [3, 7] });
    return graph.matmul(a, b);
  });
  assert.deepEqual(rank2.outputs[0].spec.capacityShape, [5, 7]);
  assert.deepEqual(rank2.outputs[0].spec.activeAxis0, { extent: 2, maximum: 5 });

  const rank3 = TensorProgram.define((graph) => {
    const a = graph.input('a', { dtype: 'f16', capacityShape: [8, 4, 3], activeAxis0: 6 });
    const b = graph.input('b', { dtype: 'f16', capacityShape: [1, 5, 3] });
    return graph.matmul(a, b, { transposeB: true });
  });
  assert.deepEqual(rank3.outputs[0].spec.capacityShape, [8, 4, 5]);
  assert.deepEqual(rank3.outputs[0].spec.activeAxis0, { extent: 6, maximum: 8 });
  assert.equal(rank3.nodes[0].options.accumulatorDtype, 'f32');

  const emptyBatch = TensorProgram.define((graph) => graph.matmul(
    graph.input('a', { dtype: 'f32', capacityShape: [0, 2, 3] }),
    graph.input('b', { dtype: 'f32', capacityShape: [1, 3, 4] }),
  ));
  assert.deepEqual(emptyBatch.outputs[0].spec.capacityShape, [0, 2, 4]);
  assert.throws(() => TensorProgram.define((graph) => graph.matmul(graph.input('a', { dtype: 'i32', capacityShape: [2, 3] }), graph.input('b', { dtype: 'i32', capacityShape: [3, 4] }))), expectCode('TENSOR_PROGRAM_DTYPE_UNSUPPORTED'));
  assert.throws(() => TensorProgram.define((graph) => graph.matmul(graph.input('a', { dtype: 'f32', capacityShape: [2, 3] }), graph.input('b', { dtype: 'f32', capacityShape: [5, 4] }))), expectCode('TENSOR_PROGRAM_MATMUL_SHAPE_INVALID'));
});

test('fill/cast/copy/contiguous are material operations with canonical scalar and dtype facts', () => {
  const program = TensorProgram.define((graph) => {
    const filled = graph.fill({ dtype: 'u64', capacityShape: [2], access: 'read-write' }, 9n);
    const copied = graph.copy(filled);
    const cast = graph.cast(copied, 'f32');
    return graph.contiguous(cast);
  });
  assert.deepEqual(program.nodes[0].options.value, { dtype: 'u64', value: '9' });
  assert.equal(TensorProgram.create(JSON.parse(JSON.stringify(program.canonical))).compatibilityIdentity, program.compatibilityIdentity);
  assert.equal(program.outputs[0].spec.dtype, 'f32');
  assert(program.nodes.every((node) => node.materialization === 'materialize'));
  const negativeZero = TensorProgram.define((graph) => graph.fill({ dtype: 'f32', capacityShape: [], access: 'read-write' }, -0));
  assert.deepEqual(negativeZero.nodes[0].options.value, { dtype: 'f32', value: '-0' });
  assert.throws(() => TensorProgram.define((graph) => graph.fill({ dtype: 'f32', capacityShape: [1], access: 'read' }, 1)), expectCode('TENSOR_PROGRAM_FILL_ACCESS_INVALID'));
});

test('invalid references, duplicate IDs, foreign values, unknown fields and async builders fail closed', () => {
  assert.throws(() => TensorProgram.create({ inputs: [], nodes: [{ op: 'copy', inputs: ['later'] }], outputs: [] }), expectCode('TENSOR_PROGRAM_REFERENCE_INVALID'));
  assert.throws(() => TensorProgram.create({ inputs: [{ name: 'x', spec: { dtype: 'f32', capacityShape: [1] } }], nodes: [{ id: 'input:x', op: 'copy', inputs: ['input:x'] }], outputs: [{ name: 'x', value: 'input:x' }] }), expectCode('TENSOR_PROGRAM_VALUE_DUPLICATE'));
  let foreign;
  TensorProgram.define((graph) => { foreign = graph.input('foreign', { dtype: 'f32', capacityShape: [1] }); return foreign; });
  assert.throws(() => TensorProgram.define((graph) => graph.copy(foreign)), expectCode('TENSOR_PROGRAM_VALUE_INVALID'));
  assert.throws(() => TensorProgram.create({ inputs: [], nodes: [], outputs: [], backend: 'simt' }), expectCode('TENSOR_PROGRAM_OPTIONS_INVALID'));
  assert.throws(() => TensorProgram.define(async () => ({})), expectCode('TENSOR_PROGRAM_BUILDER_ASYNC'));
});

test('caller-owned shapes, axes, node arrays and output records are copied before identity', () => {
  const shape = [2, 3];
  const axes = [1, 0];
  const nodeInputs = ['input:x'];
  const outputs = [{ name: 'result', value: 'transpose' }];
  const program = TensorProgram.create({
    inputs: [{ name: 'x', spec: { dtype: 'f32', capacityShape: shape } }],
    nodes: [{ id: 'transpose', op: 'permute', inputs: nodeInputs, options: { axes } }],
    outputs,
  });
  const identity = program.compatibilityIdentity;
  shape[0] = 99;
  axes.reverse();
  nodeInputs[0] = 'missing';
  outputs[0].value = 'missing';
  assert.deepEqual(program.outputs[0].spec.capacityShape, [3, 2]);
  assert.equal(program.compatibilityIdentity, identity);
  assert.throws(() => { program.canonical.nodes[0].inputs[0] = 'changed'; }, TypeError);
});
