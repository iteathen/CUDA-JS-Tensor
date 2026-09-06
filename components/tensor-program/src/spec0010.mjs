import { createTensorSpec, TensorSpec } from '../../tensor-value/index.mjs';
import { checkedAdd, deepFreeze, fail } from './contract.mjs';

export const SPEC0010_PROGRAM_CONTRACT = 'SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1';
export const SPEC0010_LIMITS = Object.freeze({
  maxInputs: 256,
  maxNodes: 4_096,
  maxOutputs: 256,
  maxStaticGatherIndices: 65_536,
  maxConcatInputs: 256,
});

export function isSpec0010Operation(op, options = {}) {
  return op === 'gather' || op === 'concat' || (op === 'unary' && options?.operator === 'erf');
}

function requireSpec(value, field) {
  if (!(value instanceof TensorSpec)) fail('TENSOR_PROGRAM_SPEC_INVALID', 'validation', `${field} must be a TensorSpec.`);
  return value;
}

function requireReadable(spec, field) {
  if (spec.access === 'write') fail('TENSOR_PROGRAM_INPUT_NOT_READABLE', 'validation', `${field} has write-only access.`);
}

function exactOptions(options, fields, op) {
  if (options === null || typeof options !== 'object' || Array.isArray(options)) {
    fail('TENSOR_PROGRAM_OPERATION_OPTIONS_INVALID', 'validation', `${op} options must be an ordinary record.`);
  }
  const allowed = new Set(fields);
  if (Object.keys(options).some((key) => !allowed.has(key))) {
    fail('TENSOR_PROGRAM_OPERATION_OPTIONS_INVALID', 'validation', `${op} options contain unknown fields.`);
  }
}

function normalizeAxis(value, rank, op) {
  if (!Number.isSafeInteger(value) || value < 0 || value >= rank) {
    fail('TENSOR_PROGRAM_AXIS_INVALID', 'validation', `${op}.axis must be a nonnegative axis within rank.`, { axis: value ?? null, rank });
  }
  return value;
}

function sameActive(left, right) {
  if (left === null || right === null) return left === right;
  return left.maximum === right.maximum && left.extent === right.extent;
}

function inferErf(input, options) {
  exactOptions(options, ['operator'], 'unary');
  requireSpec(input, 'unary input');
  requireReadable(input, 'unary input');
  if (options.operator !== 'erf') fail('TENSOR_PROGRAM_UNARY_OPERATOR_INVALID', 'validation', 'Unary operator is unsupported.', { operator: options.operator ?? null });
  if (!['f32', 'f64'].includes(input.dtype)) {
    fail('TENSOR_PROGRAM_DTYPE_UNSUPPORTED', 'unsupported', 'erf requires f32 or f64.', { dtype: input.dtype });
  }
  const outputSpec = createTensorSpec({
    dtype: input.dtype,
    capacityShape: input.capacityShape,
    activeAxis0: input.activeAxis0,
    access: 'read-write',
  });
  return Object.freeze({
    options: deepFreeze({
      operator: 'erf',
      arithmetic: 'round-to-output-dtype-v1',
      specialValues: 'ieee-nan-propagate-signed-zero-v1',
    }),
    outputSpec,
    materialization: 'materialize',
  });
}

function inferGather(input, options) {
  exactOptions(options, ['axis', 'indices'], 'gather');
  requireSpec(input, 'gather input');
  requireReadable(input, 'gather input');
  const axis = normalizeAxis(options.axis, input.rank, 'gather');
  if (!Array.isArray(options.indices) || options.indices.length > SPEC0010_LIMITS.maxStaticGatherIndices) {
    fail('TENSOR_PROGRAM_GATHER_INDICES_INVALID', options.indices?.length > SPEC0010_LIMITS.maxStaticGatherIndices ? 'pressure' : 'validation', 'gather.indices must be a bounded static array.', { maximum: SPEC0010_LIMITS.maxStaticGatherIndices });
  }
  const dimension = input.capacityShape[axis];
  const indices = options.indices.map((index, position) => {
    if (!Number.isSafeInteger(index) || index < 0 || index >= dimension) {
      fail('TENSOR_PROGRAM_GATHER_INDEX_INVALID', 'validation', 'Static gather index is outside source capacity.', { position, index: Number.isSafeInteger(index) ? index : null, dimension });
    }
    return index;
  });
  if (input.activeAxis0 !== null && axis === 0) {
    fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'Static gather on active axis 0 is unsupported in SPEC-0010.');
  }
  const capacityShape = [...input.capacityShape];
  capacityShape[axis] = indices.length;
  const outputSpec = createTensorSpec({
    dtype: input.dtype,
    capacityShape,
    activeAxis0: input.activeAxis0,
    access: 'read-write',
  });
  return Object.freeze({
    options: deepFreeze({ axis, indices: [...indices] }),
    outputSpec,
    materialization: 'materialize',
  });
}

function inferConcat(inputs, options) {
  exactOptions(options, ['axis'], 'concat');
  if (!Array.isArray(inputs) || inputs.length < 2 || inputs.length > SPEC0010_LIMITS.maxConcatInputs) {
    fail('TENSOR_PROGRAM_OPERATION_INPUTS_INVALID', inputs?.length > SPEC0010_LIMITS.maxConcatInputs ? 'pressure' : 'validation', `concat requires 2-${SPEC0010_LIMITS.maxConcatInputs} typed inputs.`);
  }
  inputs.forEach((input, index) => {
    requireSpec(input, `concat input ${index}`);
    requireReadable(input, `concat input ${index}`);
  });
  const first = inputs[0];
  const axis = normalizeAxis(options.axis, first.rank, 'concat');
  let axisLength = 0;
  for (let index = 0; index < inputs.length; index += 1) {
    const input = inputs[index];
    if (input.dtype !== first.dtype) fail('TENSOR_PROGRAM_DTYPE_MISMATCH', 'validation', 'concat inputs require identical dtypes.', { input: index });
    if (input.rank !== first.rank) fail('TENSOR_PROGRAM_CONCAT_SHAPE_INVALID', 'validation', 'concat inputs require identical rank.', { input: index });
    for (let dimension = 0; dimension < first.rank; dimension += 1) {
      if (dimension !== axis && input.capacityShape[dimension] !== first.capacityShape[dimension]) {
        fail('TENSOR_PROGRAM_CONCAT_SHAPE_INVALID', 'validation', 'concat non-axis capacity dimensions must match.', { input: index, axis: dimension });
      }
    }
    axisLength = checkedAdd(axisLength, input.capacityShape[axis], 'concat.capacityShape');
  }

  const active = inputs.filter((input) => input.activeAxis0 !== null);
  let activeAxis0 = null;
  if (active.length > 0) {
    if (active.length !== inputs.length) fail('TENSOR_PROGRAM_ACTIVE_AXIS_MISMATCH', 'validation', 'concat cannot mix active and inactive inputs.');
    if (axis === 0) fail('TENSOR_PROGRAM_ACTIVE_AXIS_UNREPRESENTABLE', 'unsupported', 'concat on active axis 0 is unsupported in SPEC-0010.');
    if (active.some((input) => !sameActive(input.activeAxis0, first.activeAxis0))) {
      fail('TENSOR_PROGRAM_ACTIVE_AXIS_MISMATCH', 'validation', 'concat active extents must match exactly.');
    }
    activeAxis0 = first.activeAxis0;
  }

  const capacityShape = [...first.capacityShape];
  capacityShape[axis] = axisLength;
  const outputSpec = createTensorSpec({
    dtype: first.dtype,
    capacityShape,
    activeAxis0,
    access: 'read-write',
  });
  return Object.freeze({
    options: deepFreeze({ axis }),
    outputSpec,
    materialization: 'materialize',
  });
}

export function inferSpec0010Operation(op, inputs, rawOptions) {
  const options = rawOptions ?? {};
  if (op === 'unary' && options.operator === 'erf') {
    if (!Array.isArray(inputs) || inputs.length !== 1) fail('TENSOR_PROGRAM_OPERATION_INPUTS_INVALID', 'validation', 'unary requires exactly 1 typed input value.');
    return inferErf(inputs[0], options);
  }
  if (op === 'gather') {
    if (!Array.isArray(inputs) || inputs.length !== 1) fail('TENSOR_PROGRAM_OPERATION_INPUTS_INVALID', 'validation', 'gather requires exactly 1 typed input value.');
    return inferGather(inputs[0], options);
  }
  if (op === 'concat') return inferConcat(inputs, options);
  fail('TENSOR_PROGRAM_OPERATION_INVALID', 'internal', 'SPEC-0010 dispatcher received an unowned operation.', { op: op ?? null });
}
