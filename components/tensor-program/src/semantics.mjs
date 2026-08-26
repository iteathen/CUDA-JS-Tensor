import { createTensorSpec, TensorSpec, tensorDtypeWidth, TENSOR_DTYPES } from '../../tensor-value/index.mjs';
import { checkedAdd, checkedMultiply, deepFreeze, exactRecord, fail, plainObject } from './contract.mjs';

const FLOAT_DTYPES = new Set(['f16', 'bf16', 'f32', 'f64']);
const SIGNED_DTYPES = new Set(['i32', ...FLOAT_DTYPES]);
const UNARY_OPERATORS = new Set(['neg', 'abs', 'exp', 'log', 'sqrt']);
const BINARY_OPERATORS = new Set(['add', 'sub', 'mul', 'div', 'minimum', 'maximum']);
const REDUCTION_OPERATORS = new Set(['sum', 'product', 'minimum', 'maximum']);
const OP_FIELDS = Object.freeze({
  fill: new Set(['spec', 'value']),
  copy: new Set(),
  cast: new Set(['dtype']),
  reshape: new Set(['capacityShape']),
  permute: new Set(['axes']),
  slice: new Set(['slices']),
  broadcast: new Set(['capacityShape']),
  contiguous: new Set(),
  unary: new Set(['operator']),
  binary: new Set(['operator']),
  reduce: new Set(['operator', 'axes', 'keepDimensions', 'accumulatorDtype', 'order', 'identity']),
  matmul: new Set(['transposeA', 'transposeB', 'accumulatorDtype']),
});
const MATERIAL_OPS = new Set(['fill', 'copy', 'cast', 'contiguous', 'unary', 'binary', 'reduce', 'matmul']);
const VIEW_OPS = new Set(['reshape', 'permute', 'slice', 'broadcast']);
const U32_MAX = 0xffff_ffff;
const I32_MIN = -0x8000_0000;
const I32_MAX = 0x7fff_ffff;
const U64_MAX = (1n << 64n) - 1n;

function requireSpec(value, field) {
  if (!(value instanceof TensorSpec)) fail('TENSOR_PROGRAM_SPEC_INVALID', 'validation', `${field} must be a TensorSpec.`);
  return value;
}

function requireReadable(spec, field) {
  if (spec.access === 'write') fail('TENSOR_PROGRAM_INPUT_NOT_READABLE', 'validation', `${field} has write-only access.`);
}

function normalizeShape(value, field) {
  if (!Array.isArray(value) || value.length > 16 || value.some((entry) => !Number.isSafeInteger(entry) || entry < 0)) {
    fail('TENSOR_PROGRAM_SHAPE_INVALID', 'validation', `${field} must be a rank 0-16 nonnegative safe-integer shape.`, { field });
  }
  return Object.freeze([...value]);
}

function sameShape(left, right) {
  return left.length === right.length && left.every((entry, index) => entry === right[index]);
}

function materialSpec(input, dtype = input.dtype) {
  return createTensorSpec({
    dtype,
    capacityShape: input.capacityShape,
    activeAxis0: input.activeAxis0,
    access: 'read-write',
  });
}

function assertViewWithin(input, output) {
  if (output.byteOffset < input.byteOffset || output.requiredByteEnd > input.requiredByteEnd) {
    fail('TENSOR_PROGRAM_VIEW_RANGE_INVALID', 'validation', 'Derived view exceeds its input byte authority.', { inputStart: input.byteOffset, inputEnd: input.requiredByteEnd, outputStart: output.byteOffset, outputEnd: output.requiredByteEnd });
  }
  return output;
}

function canonicalScalar(dtype, value, { allowSpecial = false } = {}) {
  if (dtype === 'u64') {
    if (typeof value !== 'bigint' || value < 0n || value > U64_MAX) fail('TENSOR_PROGRAM_SCALAR_INVALID', 'validation', 'u64 scalar must be a bigint in range.');
    return Object.freeze({ dtype, value: value.toString(10) });
  }
  if (dtype === 'u32') {
    if (!Number.isSafeInteger(value) || value < 0 || value > U32_MAX) fail('TENSOR_PROGRAM_SCALAR_INVALID', 'validation', 'u32 scalar is out of range.');
    return Object.freeze({ dtype, value });
  }
  if (dtype === 'i32') {
    if (!Number.isSafeInteger(value) || value < I32_MIN || value > I32_MAX) fail('TENSOR_PROGRAM_SCALAR_INVALID', 'validation', 'i32 scalar is out of range.');
    return Object.freeze({ dtype, value });
  }
  if (typeof value !== 'number' || (!Number.isFinite(value) && !(allowSpecial && (value === Infinity || value === -Infinity)))) {
    fail('TENSOR_PROGRAM_SCALAR_INVALID', 'validation', `${dtype} scalar must be ${allowSpecial ? 'numeric' : 'finite'}.`);
  }
  return Object.freeze({ dtype, value: value === Infinity ? 'Infinity' : (value === -Infinity ? '-Infinity' : (Object.is(value, -0) ? '-0' : value)) });
}

function defaultAccumulator(dtype) {
  return dtype === 'f16' || dtype === 'bf16' ? 'f32' : dtype;
}

function normalizeAccumulator(input, requested) {
  const result = requested ?? defaultAccumulator(input);
  const allowed = input === result
    || (input === 'u32' && result === 'u64')
    || ((input === 'f16' || input === 'bf16') && (result === 'f32' || result === 'f64'))
    || (input === 'f32' && result === 'f64');
  if (!allowed) fail('TENSOR_PROGRAM_ACCUMULATOR_INVALID', 'validation', 'accumulatorDtype is not an accepted exact/widening profile.', { inputDtype: input, accumulatorDtype: result });
  return result;
}

function mergeActiveForElementwise(left, right, outputShape) {
  const active = [left, right].filter((spec) => spec.activeAxis0 !== null);
  if (active.length === 0) return null;
  for (const spec of active) {
    if (spec.rank !== outputShape.length || spec.capacityShape[0] !== outputShape[0]) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'Active axis 0 cannot shift under elementwise broadcasting.');
  }
  if (active.length === 2 && (active[0].activeAxis0.maximum !== active[1].activeAxis0.maximum || active[0].activeAxis0.extent !== active[1].activeAxis0.extent)) {
    fail('TENSOR_PROGRAM_ACTIVE_AXIS_MISMATCH', 'validation', 'Elementwise active extents must match exactly.');
  }
  return active[0].activeAxis0;
}

function broadcastShape(left, right) {
  const rank = Math.max(left.length, right.length);
  const output = new Array(rank);
  for (let offset = 1; offset <= rank; offset += 1) {
    const a = left[left.length - offset] ?? 1;
    const b = right[right.length - offset] ?? 1;
    if (a !== b && a !== 1 && b !== 1) fail('TENSOR_PROGRAM_BROADCAST_MISMATCH', 'validation', 'Shapes are not broadcast-compatible.', { left: a, right: b });
    output[rank - offset] = a === b ? a : (a === 1 ? b : a);
  }
  return Object.freeze(output);
}

function normalizeAxes(value, rank) {
  const axes = value === undefined ? Array.from({ length: rank }, (_, index) => index) : value;
  if (!Array.isArray(axes) || axes.some((axis) => !Number.isSafeInteger(axis) || axis < 0 || axis >= rank)) fail('TENSOR_PROGRAM_REDUCTION_AXES_INVALID', 'validation', 'Reduction axes are out of range.');
  const sorted = [...axes].sort((a, b) => a - b);
  if (sorted.some((axis, index) => index > 0 && axis === sorted[index - 1])) fail('TENSOR_PROGRAM_REDUCTION_AXES_INVALID', 'validation', 'Reduction axes must be unique.');
  return Object.freeze(sorted);
}

function reductionIdentity(operator, dtype, value) {
  if (value !== undefined) return canonicalScalar(dtype, value, { allowSpecial: true });
  if (operator === 'sum') return canonicalScalar(dtype, dtype === 'u64' ? 0n : 0);
  if (operator === 'product') return canonicalScalar(dtype, dtype === 'u64' ? 1n : 1);
  if (operator === 'minimum') {
    if (dtype === 'u32') return canonicalScalar(dtype, U32_MAX);
    if (dtype === 'u64') return canonicalScalar(dtype, U64_MAX);
    if (dtype === 'i32') return canonicalScalar(dtype, I32_MAX);
    return canonicalScalar(dtype, Infinity, { allowSpecial: true });
  }
  if (dtype === 'u32') return canonicalScalar(dtype, 0);
  if (dtype === 'u64') return canonicalScalar(dtype, 0n);
  if (dtype === 'i32') return canonicalScalar(dtype, I32_MIN);
  return canonicalScalar(dtype, -Infinity, { allowSpecial: true });
}

function inferFill(options) {
  const spec = requireSpec(options.spec, 'fill.spec');
  if (spec.access !== 'read-write') fail('TENSOR_PROGRAM_FILL_ACCESS_INVALID', 'validation', 'fill output spec must be read-write.');
  return { options: deepFreeze({ spec: spec.canonical, value: canonicalScalar(spec.dtype, options.value) }), outputSpec: spec };
}

function inferReshape(input, options) {
  requireReadable(input, 'reshape input');
  const capacityShape = normalizeShape(options.capacityShape, 'reshape.capacityShape');
  let count = 1;
  for (const dimension of capacityShape) count = checkedMultiply(count, dimension, 'reshape.capacityElementCount');
  if (count !== input.capacityElementCount || input.layout !== 'row-major-contiguous' || input.hasBroadcastAliasing) fail('TENSOR_PROGRAM_RESHAPE_INVALID', 'validation', 'reshape requires equal element count and contiguous non-broadcast order.');
  if (input.activeAxis0 && !sameShape(capacityShape, input.capacityShape)) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'A dynamic active extent cannot be remapped by reshape in v1.');
  const outputSpec = assertViewWithin(input, createTensorSpec({ dtype: input.dtype, capacityShape, activeAxis0: input.activeAxis0, byteOffset: input.byteOffset, access: input.access, aliasGroup: input.aliasGroup }));
  return { options: deepFreeze({ capacityShape: [...capacityShape] }), outputSpec };
}

function inferPermute(input, options) {
  requireReadable(input, 'permute input');
  const axes = options.axes;
  if (!Array.isArray(axes) || axes.length !== input.rank || axes.some((axis) => !Number.isSafeInteger(axis) || axis < 0 || axis >= input.rank) || new Set(axes).size !== input.rank) {
    fail('TENSOR_PROGRAM_PERMUTATION_INVALID', 'validation', 'axes must be an exact permutation.');
  }
  if (input.activeAxis0 && axes[0] !== 0) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'Active axis 0 must remain output axis 0.');
  const outputSpec = assertViewWithin(input, createTensorSpec({
    dtype: input.dtype,
    capacityShape: axes.map((axis) => input.capacityShape[axis]),
    activeAxis0: input.activeAxis0,
    strides: axes.map((axis) => input.strides[axis]),
    byteOffset: input.byteOffset,
    access: input.access,
    aliasGroup: input.aliasGroup,
  }));
  return { options: deepFreeze({ axes: [...axes] }), outputSpec };
}

function inferSlice(input, options) {
  requireReadable(input, 'slice input');
  if (!Array.isArray(options.slices) || options.slices.length !== input.rank) fail('TENSOR_PROGRAM_SLICE_INVALID', 'validation', 'slices must contain one entry per axis.');
  const slices = options.slices.map((entry, axis) => {
    const dimension = input.capacityShape[axis];
    if (entry === null) return Object.freeze({ start: 0, length: dimension, step: 1 });
    exactRecord(entry, new Set(['start', 'length', 'step']), 'TENSOR_PROGRAM_SLICE_INVALID', 'Each slice must be null or an exact start/length/step record.');
    const start = entry.start;
    const length = entry.length;
    const step = entry.step ?? 1;
    if (![start, length, step].every(Number.isSafeInteger) || start < 0 || length < 0 || step < 1 || start > dimension) fail('TENSOR_PROGRAM_SLICE_INVALID', 'validation', 'Slice values are outside finite positive-step bounds.');
    if (length > 0) {
      const reach = checkedMultiply(length - 1, step, 'slice.reach');
      if (checkedAdd(start, reach, 'slice.last') >= dimension) fail('TENSOR_PROGRAM_SLICE_INVALID', 'validation', 'Slice reaches outside its capacity dimension.');
    }
    return Object.freeze({ start, length, step });
  });
  if (input.activeAxis0) {
    const axis0 = slices[0];
    if (axis0.start !== 0 || axis0.length !== input.capacityShape[0] || axis0.step !== 1) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'Active axis 0 requires a full-capacity slice.');
  }
  let elementOffset = 0;
  if (slices.every((entry) => entry.length > 0)) {
    for (let axis = 0; axis < input.rank; axis += 1) elementOffset = checkedAdd(elementOffset, checkedMultiply(slices[axis].start, input.strides[axis], 'slice.byteOffset'), 'slice.byteOffset');
  }
  const byteOffset = checkedAdd(input.byteOffset, checkedMultiply(elementOffset, input.dtypeWidth, 'slice.byteOffset'), 'slice.byteOffset');
  const outputSpec = assertViewWithin(input, createTensorSpec({
    dtype: input.dtype,
    capacityShape: slices.map((entry) => entry.length),
    activeAxis0: input.activeAxis0,
    strides: slices.map((entry, axis) => checkedMultiply(input.strides[axis], entry.step, 'slice.stride')),
    byteOffset,
    access: input.access,
    aliasGroup: input.aliasGroup,
  }));
  return { options: deepFreeze({ slices: slices.map((entry) => ({ ...entry })) }), outputSpec };
}

function inferBroadcast(input, options) {
  requireReadable(input, 'broadcast input');
  const target = normalizeShape(options.capacityShape, 'broadcast.capacityShape');
  if (target.length < input.rank) fail('TENSOR_PROGRAM_BROADCAST_MISMATCH', 'validation', 'Broadcast target rank cannot shrink.');
  if (input.activeAxis0 && (target.length !== input.rank || target[0] !== input.capacityShape[0])) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'Active axis 0 cannot shift or change capacity under broadcast.');
  const strides = new Array(target.length).fill(0);
  let aliased = false;
  for (let offset = 1; offset <= target.length; offset += 1) {
    const sourceAxis = input.rank - offset;
    const targetAxis = target.length - offset;
    if (sourceAxis < 0) { if (target[targetAxis] > 1) aliased = true; continue; }
    const source = input.capacityShape[sourceAxis];
    if (source !== target[targetAxis] && source !== 1) fail('TENSOR_PROGRAM_BROADCAST_MISMATCH', 'validation', 'Broadcast target dimension is incompatible.', { source, target: target[targetAxis] });
    if (source === 1 && target[targetAxis] > 1) aliased = true;
    else strides[targetAxis] = input.strides[sourceAxis];
  }
  const outputSpec = assertViewWithin(input, createTensorSpec({ dtype: input.dtype, capacityShape: target, activeAxis0: input.activeAxis0, strides, byteOffset: input.byteOffset, access: aliased ? 'read' : input.access, aliasGroup: input.aliasGroup }));
  return { options: deepFreeze({ capacityShape: [...target] }), outputSpec };
}

function inferUnary(input, options) {
  requireReadable(input, 'unary input');
  const operator = options.operator;
  if (!UNARY_OPERATORS.has(operator)) fail('TENSOR_PROGRAM_UNARY_OPERATOR_INVALID', 'validation', 'Unary operator is unsupported.', { operator: operator ?? null });
  if (operator === 'neg' && !SIGNED_DTYPES.has(input.dtype)) fail('TENSOR_PROGRAM_DTYPE_UNSUPPORTED', 'unsupported', 'neg requires a signed or floating dtype.');
  if (['exp', 'log', 'sqrt'].includes(operator) && !FLOAT_DTYPES.has(input.dtype)) fail('TENSOR_PROGRAM_DTYPE_UNSUPPORTED', 'unsupported', `${operator} requires a floating dtype.`);
  return { options: deepFreeze({ operator, arithmetic: FLOAT_DTYPES.has(input.dtype) ? 'round-to-output-dtype-v1' : 'modulo-width-v1', specialValues: FLOAT_DTYPES.has(input.dtype) ? 'ieee-nan-propagate-signed-zero-v1' : null }), outputSpec: materialSpec(input) };
}

function inferBinary(left, right, options) {
  requireReadable(left, 'binary left input');
  requireReadable(right, 'binary right input');
  const operator = options.operator;
  if (!BINARY_OPERATORS.has(operator)) fail('TENSOR_PROGRAM_BINARY_OPERATOR_INVALID', 'validation', 'Binary operator is unsupported.', { operator: operator ?? null });
  if (left.dtype !== right.dtype) fail('TENSOR_PROGRAM_DTYPE_MISMATCH', 'validation', 'Binary operations require identical dtypes; cast explicitly.');
  if (operator === 'div' && !FLOAT_DTYPES.has(left.dtype)) fail('TENSOR_PROGRAM_DTYPE_UNSUPPORTED', 'unsupported', 'Integer division is not in the first dense profile.');
  const capacityShape = broadcastShape(left.capacityShape, right.capacityShape);
  const activeAxis0 = mergeActiveForElementwise(left, right, capacityShape);
  const outputSpec = createTensorSpec({ dtype: left.dtype, capacityShape, activeAxis0, access: 'read-write' });
  const arithmetic = FLOAT_DTYPES.has(left.dtype) ? 'round-to-output-dtype-v1' : (['minimum', 'maximum'].includes(operator) ? 'integer-exact-ordered-v1' : 'modulo-width-v1');
  return { options: deepFreeze({ operator, arithmetic, specialValues: FLOAT_DTYPES.has(left.dtype) ? 'ieee-nan-propagate-signed-zero-v1' : null }), outputSpec };
}

function inferReduce(input, options) {
  requireReadable(input, 'reduce input');
  const operator = options.operator;
  if (!REDUCTION_OPERATORS.has(operator)) fail('TENSOR_PROGRAM_REDUCTION_OPERATOR_INVALID', 'validation', 'Reduction operator is unsupported.', { operator: operator ?? null });
  const axes = normalizeAxes(options.axes, input.rank);
  const keepDimensions = options.keepDimensions ?? false;
  if (typeof keepDimensions !== 'boolean') fail('TENSOR_PROGRAM_REDUCTION_OPTIONS_INVALID', 'validation', 'keepDimensions must be boolean.');
  const accumulatorDtype = normalizeAccumulator(input.dtype, options.accumulatorDtype);
  const order = options.order ?? 'fixed-tree-v1';
  if (!['fixed-tree-v1', 'backend-defined'].includes(order)) fail('TENSOR_PROGRAM_REDUCTION_OPTIONS_INVALID', 'validation', 'Reduction order is unsupported.');
  const axisSet = new Set(axes);
  const capacityShape = keepDimensions
    ? input.capacityShape.map((dimension, axis) => axisSet.has(axis) ? 1 : dimension)
    : input.capacityShape.filter((_, axis) => !axisSet.has(axis));
  const activeAxis0 = input.activeAxis0 && !axisSet.has(0) ? input.activeAxis0 : null;
  const outputSpec = createTensorSpec({ dtype: accumulatorDtype, capacityShape, activeAxis0, access: 'read-write' });
  return {
    options: deepFreeze({ operator, axes: [...axes], keepDimensions, accumulatorDtype, order, identity: reductionIdentity(operator, accumulatorDtype, options.identity), specialValues: FLOAT_DTYPES.has(accumulatorDtype) ? 'ieee-nan-propagate-signed-zero-v1' : null }),
    outputSpec,
  };
}

function matmulDimensions(spec, transpose) {
  const offset = spec.rank - 2;
  const first = spec.capacityShape[offset];
  const second = spec.capacityShape[offset + 1];
  return transpose ? { rows: second, columns: first } : { rows: first, columns: second };
}

function inferMatmul(left, right, options) {
  requireReadable(left, 'matmul left input');
  requireReadable(right, 'matmul right input');
  if (left.dtype !== right.dtype || !FLOAT_DTYPES.has(left.dtype)) fail('TENSOR_PROGRAM_DTYPE_UNSUPPORTED', 'unsupported', 'matmul requires matching floating dtypes.');
  if (left.rank !== right.rank || ![2, 3].includes(left.rank)) fail('TENSOR_PROGRAM_MATMUL_RANK_INVALID', 'validation', 'matmul requires both inputs to have rank 2 or both rank 3.');
  const transposeA = options.transposeA ?? false;
  const transposeB = options.transposeB ?? false;
  if (typeof transposeA !== 'boolean' || typeof transposeB !== 'boolean') fail('TENSOR_PROGRAM_MATMUL_OPTIONS_INVALID', 'validation', 'transpose flags must be boolean.');
  const a = matmulDimensions(left, transposeA);
  const b = matmulDimensions(right, transposeB);
  if (a.columns !== b.rows) fail('TENSOR_PROGRAM_MATMUL_SHAPE_INVALID', 'validation', 'matmul contracting capacity dimensions do not match.', { left: a.columns, right: b.rows });
  const accumulatorDtype = normalizeAccumulator(left.dtype, options.accumulatorDtype);
  let capacityShape;
  let activeAxis0 = null;
  if (left.rank === 2) {
    if (right.activeAxis0 || (left.activeAxis0 && transposeA)) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'Rank-2 matmul admits active extent only on non-transposed A rows.');
    capacityShape = [a.rows, b.columns];
    activeAxis0 = left.activeAxis0;
  } else {
    const leftBatch = left.capacityShape[0];
    const rightBatch = right.capacityShape[0];
    if (leftBatch !== rightBatch && leftBatch !== 1 && rightBatch !== 1) fail('TENSOR_PROGRAM_MATMUL_BATCH_INVALID', 'validation', 'Batched matmul capacities are not broadcast-compatible.');
    const batch = leftBatch === rightBatch ? leftBatch : (leftBatch === 1 ? rightBatch : leftBatch);
    capacityShape = [batch, a.rows, b.columns];
    const active = [left, right].filter((spec) => spec.activeAxis0 !== null);
    if (active.some((spec) => spec.capacityShape[0] !== batch)) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'Active batch capacity must equal output batch capacity.');
    if (active.length === 2 && (active[0].activeAxis0.maximum !== active[1].activeAxis0.maximum || active[0].activeAxis0.extent !== active[1].activeAxis0.extent)) fail('TENSOR_PROGRAM_ACTIVE_AXIS_MISMATCH', 'validation', 'Batched matmul active extents must match exactly.');
    activeAxis0 = active[0]?.activeAxis0 ?? null;
  }
  const outputSpec = createTensorSpec({ dtype: left.dtype, capacityShape, activeAxis0, access: 'read-write' });
  return { options: deepFreeze({ transposeA, transposeB, accumulatorDtype, outputDtype: left.dtype, arithmetic: 'accumulate-then-round-output-v1', specialValues: 'ieee-nan-propagate-signed-zero-v1' }), outputSpec };
}

export function inferOperation(op, inputs, rawOptions) {
  if (!Object.hasOwn(OP_FIELDS, op)) fail('TENSOR_PROGRAM_OPERATION_INVALID', 'validation', 'Operation family is unsupported.', { op: op ?? null });
  const options = rawOptions ?? {};
  exactRecord(options, OP_FIELDS[op], 'TENSOR_PROGRAM_OPERATION_OPTIONS_INVALID', `${op} options contain unknown fields.`);
  const expectedInputs = op === 'fill' ? 0 : (op === 'binary' || op === 'matmul' ? 2 : 1);
  if (!Array.isArray(inputs) || inputs.length !== expectedInputs || inputs.some((spec) => !(spec instanceof TensorSpec))) fail('TENSOR_PROGRAM_OPERATION_INPUTS_INVALID', 'validation', `${op} requires exactly ${expectedInputs} typed input value(s).`);

  let result;
  if (op === 'fill') result = inferFill(options);
  else if (op === 'copy') { requireReadable(inputs[0], 'copy input'); result = { options: Object.freeze({}), outputSpec: materialSpec(inputs[0]) }; }
  else if (op === 'cast') {
    requireReadable(inputs[0], 'cast input');
    if (!TENSOR_DTYPES.includes(options.dtype)) fail('TENSOR_PROGRAM_DTYPE_UNSUPPORTED', 'unsupported', 'cast dtype is unsupported.', { dtype: options.dtype ?? null });
    result = { options: deepFreeze({ dtype: options.dtype, conversion: 'explicit-cast-v1' }), outputSpec: materialSpec(inputs[0], options.dtype) };
  } else if (op === 'reshape') result = inferReshape(inputs[0], options);
  else if (op === 'permute') result = inferPermute(inputs[0], options);
  else if (op === 'slice') result = inferSlice(inputs[0], options);
  else if (op === 'broadcast') result = inferBroadcast(inputs[0], options);
  else if (op === 'contiguous') { requireReadable(inputs[0], 'contiguous input'); result = { options: Object.freeze({}), outputSpec: materialSpec(inputs[0]) }; }
  else if (op === 'unary') result = inferUnary(inputs[0], options);
  else if (op === 'binary') result = inferBinary(inputs[0], inputs[1], options);
  else if (op === 'reduce') result = inferReduce(inputs[0], options);
  else result = inferMatmul(inputs[0], inputs[1], options);

  return Object.freeze({ ...result, materialization: MATERIAL_OPS.has(op) ? 'materialize' : (VIEW_OPS.has(op) ? 'view' : fail('TENSOR_PROGRAM_OPERATION_INVALID', 'internal', 'Operation materialization is unowned.')) });
}

export function physicalBytes(spec) {
  requireSpec(spec, 'physicalBytes.spec');
  return Math.max(spec.requiredByteEnd, tensorDtypeWidth(spec.dtype));
}
