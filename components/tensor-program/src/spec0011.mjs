import { createTensorSpec, TensorSpec } from '../../tensor-value/index.mjs';
import { deepFreeze, fail } from './contract.mjs';

export const SPEC0011_CONTRACT_SUFFIX = 'SPEC-0011-tanh-v1';
export const SPEC0011_PROGRAM_CONTRACT = `SPEC-0004-tensor-program-v1+${SPEC0011_CONTRACT_SUFFIX}`;
export const SPEC0010_SPEC0011_PROGRAM_CONTRACT = `SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1+${SPEC0011_CONTRACT_SUFFIX}`;

export function isSpec0011Operation(op, options = {}) {
  return op === 'unary' && options?.operator === 'tanh';
}

function exactOptions(options) {
  if (options === null || typeof options !== 'object' || Array.isArray(options) || Object.keys(options).some((key) => key !== 'operator')) {
    fail('TENSOR_PROGRAM_OPERATION_OPTIONS_INVALID', 'validation', 'unary options contain unknown fields.');
  }
}

export function inferSpec0011Operation(op, inputs, rawOptions) {
  const options = rawOptions ?? {};
  if (!isSpec0011Operation(op, options)) {
    fail('TENSOR_PROGRAM_OPERATION_INVALID', 'internal', 'SPEC-0011 dispatcher received an unowned operation.', { op: op ?? null });
  }
  if (!Array.isArray(inputs) || inputs.length !== 1) {
    fail('TENSOR_PROGRAM_OPERATION_INPUTS_INVALID', 'validation', 'unary requires exactly 1 typed input value.');
  }
  exactOptions(options);
  const input = inputs[0];
  if (!(input instanceof TensorSpec)) fail('TENSOR_PROGRAM_SPEC_INVALID', 'validation', 'unary input must be a TensorSpec.');
  if (input.access === 'write') fail('TENSOR_PROGRAM_INPUT_NOT_READABLE', 'validation', 'unary input has write-only access.');
  if (!['f32', 'f64'].includes(input.dtype)) {
    fail('TENSOR_PROGRAM_DTYPE_UNSUPPORTED', 'unsupported', 'tanh requires f32 or f64.', { dtype: input.dtype });
  }
  const outputSpec = createTensorSpec({
    dtype: input.dtype,
    capacityShape: input.capacityShape,
    activeAxis0: input.activeAxis0,
    access: 'read-write',
  });
  return Object.freeze({
    options: deepFreeze({
      operator: 'tanh',
      arithmetic: 'round-to-output-dtype-v1',
      specialValues: 'ieee-nan-propagate-signed-zero-v1',
    }),
    outputSpec,
    materialization: 'materialize',
  });
}
