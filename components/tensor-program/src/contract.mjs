import { createHash } from 'node:crypto';
import { TensorError } from '../../tensor-value/index.mjs';

export function fail(code, category, message, details = {}) {
  throw new TensorError(code, category, message, details);
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

export function checkedMultiply(left, right, field) {
  if (left !== 0 && right > Math.floor(Number.MAX_SAFE_INTEGER / left)) fail('TENSOR_PROGRAM_RANGE_OVERFLOW', 'validation', `${field} exceeds the safe integer range.`, { field });
  return left * right;
}

export function checkedAdd(left, right, field) {
  if (right > Number.MAX_SAFE_INTEGER - left) fail('TENSOR_PROGRAM_RANGE_OVERFLOW', 'validation', `${field} exceeds the safe integer range.`, { field });
  return left + right;
}

export function boundedName(value, field) {
  if (typeof value !== 'string' || !/^[A-Za-z][A-Za-z0-9._:-]{0,127}$/u.test(value)) fail('TENSOR_PROGRAM_NAME_INVALID', 'validation', `${field} must be a bounded identifier.`, { field });
  return value;
}
