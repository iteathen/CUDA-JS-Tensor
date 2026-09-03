import { createHash } from 'node:crypto';

import { TensorError } from '../../tensor-value/index.mjs';
import { CUDA_JS_TENSOR_COMPATIBILITY } from './cuda-js-compatibility.mjs';

export const RESOLVED_TENSOR_PLAN_CONTRACT = 'SPEC-0006-resolved-dense-plan-v1+SPEC-0007-exact-elementwise-fusion-v1';
export const TENSOR_EXECUTION_RESULT_CONTRACT = 'SPEC-0005-tensor-execution-result-v1';

const tensorSimtLimits = {
  maxLogicalWorkItems: 0xffff_ffff,
  maxWorkspaceBytes: 64 * 1024 * 1024,
  cudaJsPreparedOperationDagLimits: CUDA_JS_TENSOR_COMPATIBILITY.preparedOperationDagLimits,
};
Object.defineProperties(tensorSimtLimits, {
  maxKernels: {
    enumerable: false,
    get() { return CUDA_JS_TENSOR_COMPATIBILITY.preparedOperationDagLimits.nodes; },
  },
  maxBindings: {
    enumerable: false,
    get() { return CUDA_JS_TENSOR_COMPATIBILITY.preparedOperationDagLimits.bindings; },
  },
});
export const TENSOR_SIMT_LIMITS = Object.freeze(tensorSimtLimits);

export function fail(code, category, message, details = {}, options = {}) {
  throw new TensorError(code, category, message, details, options);
}

export function plainObject(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  try { return Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null; } catch { return false; }
}

export function exactRecord(value, fields, code, message) {
  if (!plainObject(value) || Object.keys(value).some((key) => !fields.has(key))) fail(code, 'validation', message);
  return value;
}

export function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Array.isArray(value) ? value : Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

export function identity(prefix, record) {
  return `${prefix}:${createHash('sha256').update(JSON.stringify(record)).digest('hex')}`;
}

export function checkedAdd(left, right, field) {
  if (!Number.isSafeInteger(left) || !Number.isSafeInteger(right) || left < 0 || right < 0 || right > Number.MAX_SAFE_INTEGER - left) {
    fail('TENSOR_EXECUTION_RANGE_OVERFLOW', 'pressure', `${field} exceeds the safe integer range.`, { field });
  }
  return left + right;
}

export function checkedMultiply(left, right, field) {
  if (!Number.isSafeInteger(left) || !Number.isSafeInteger(right) || left < 0 || right < 0
      || (left !== 0 && right > Math.floor(Number.MAX_SAFE_INTEGER / left))) {
    fail('TENSOR_EXECUTION_RANGE_OVERFLOW', 'pressure', `${field} exceeds the safe integer range.`, { field });
  }
  return left * right;
}

export function failureSummary(error) {
  return Object.freeze({
    code: typeof error?.code === 'string' ? error.code : 'UNKNOWN',
    category: typeof error?.category === 'string' ? error.category : 'unknown',
    name: typeof error?.name === 'string' ? error.name : 'Error',
  });
}
