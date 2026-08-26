import { createHash } from 'node:crypto';
import { deepFreeze, fail } from './error.mjs';

export const TENSOR_SPEC_CONTRACT = 'SPEC-0001-tensor-spec-v1';
export const TENSOR_DTYPES = Object.freeze(['u32', 'u64', 'i32', 'f32', 'f64', 'f16', 'bf16']);
export const TENSOR_ACCESS_ROLES = Object.freeze(['read', 'write', 'read-write']);

const DTYPE_WIDTH = Object.freeze({ u32: 4, u64: 8, i32: 4, f32: 4, f64: 8, f16: 2, bf16: 2 });
const DTYPE_SET = new Set(TENSOR_DTYPES);
const ACCESS_SET = new Set(TENSOR_ACCESS_ROLES);
const SPEC_FIELDS = new Set(['dtype', 'capacityShape', 'activeAxis0', 'strides', 'byteOffset', 'alignment', 'access', 'aliasGroup']);
const SPEC_TOKEN = Symbol('TensorSpec');

function plainObject(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  try { return Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null; } catch { return false; }
}

function nonnegativeSafeInteger(value, field) {
  if (!Number.isSafeInteger(value) || value < 0) fail('TENSOR_SPEC_RANGE_INVALID', 'validation', `${field} must be a nonnegative safe integer.`, { field });
  return value;
}

function checkedMultiply(left, right, field) {
  if (left !== 0 && right > Math.floor(Number.MAX_SAFE_INTEGER / left)) {
    fail('TENSOR_SPEC_RANGE_OVERFLOW', 'validation', `${field} exceeds the safe integer range.`, { field });
  }
  return left * right;
}

function checkedAdd(left, right, field) {
  if (right > Number.MAX_SAFE_INTEGER - left) fail('TENSOR_SPEC_RANGE_OVERFLOW', 'validation', `${field} exceeds the safe integer range.`, { field });
  return left + right;
}

function normalizeShape(value, field) {
  if (!Array.isArray(value) || value.length > 16) fail('TENSOR_SPEC_SHAPE_INVALID', 'validation', `${field} must be an array with rank from 0 through 16.`, { field });
  return Object.freeze(value.map((dimension, index) => nonnegativeSafeInteger(dimension, `${field}[${index}]`)));
}

function shapeProduct(shape, field) {
  let product = 1;
  for (const dimension of shape) product = checkedMultiply(product, dimension, field);
  return product;
}

function rowMajorStrides(shape) {
  const strides = new Array(shape.length);
  let stride = 1;
  for (let index = shape.length - 1; index >= 0; index -= 1) {
    strides[index] = stride;
    stride = checkedMultiply(stride, shape[index], 'rowMajorStrides');
  }
  return Object.freeze(strides);
}

function normalizeStrides(value, shape, defaults) {
  if (value === undefined) return defaults;
  if (!Array.isArray(value) || value.length !== shape.length) {
    fail('TENSOR_SPEC_STRIDES_INVALID', 'validation', 'strides must contain one nonnegative element stride per dimension.');
  }
  return Object.freeze(value.map((stride, index) => nonnegativeSafeInteger(stride, `strides[${index}]`)));
}

function normalizeActiveAxis0(value, shape) {
  if (value === undefined || value === null) return null;
  if (shape.length === 0) fail('TENSOR_SPEC_ACTIVE_AXIS_INVALID', 'validation', 'A rank-zero tensor cannot declare activeAxis0.');
  const record = Number.isSafeInteger(value) ? { extent: value, maximum: shape[0] } : value;
  if (!plainObject(record) || Object.keys(record).some((key) => !['extent', 'maximum'].includes(key)) || !Object.hasOwn(record, 'extent') || !Object.hasOwn(record, 'maximum')) {
    fail('TENSOR_SPEC_ACTIVE_AXIS_INVALID', 'validation', 'activeAxis0 must be an extent or an exact extent/maximum record.');
  }
  const extent = nonnegativeSafeInteger(record.extent, 'activeAxis0.extent');
  const maximum = nonnegativeSafeInteger(record.maximum, 'activeAxis0.maximum');
  if (maximum !== shape[0] || extent > maximum) {
    fail('TENSOR_SPEC_ACTIVE_AXIS_INVALID', 'validation', 'activeAxis0 must stay within capacity axis 0 and declare that exact maximum.', { extent, maximum, capacity: shape[0] });
  }
  return Object.freeze({ extent, maximum });
}

function normalizeAliasGroup(value) {
  if (value === undefined || value === null) return null;
  if (typeof value !== 'string' || value.length < 1 || value.length > 128 || /[\x00-\x1f\x7f]/u.test(value)) {
    fail('TENSOR_SPEC_ALIAS_GROUP_INVALID', 'validation', 'aliasGroup must be null or a bounded printable string.');
  }
  return value;
}

function storageSpan(shape, strides, elementCount) {
  if (elementCount === 0) return 0;
  let maximumIndex = 0;
  for (let index = 0; index < shape.length; index += 1) {
    const reach = checkedMultiply(shape[index] - 1, strides[index], 'storageElementSpan');
    maximumIndex = checkedAdd(maximumIndex, reach, 'storageElementSpan');
  }
  return checkedAdd(maximumIndex, 1, 'storageElementSpan');
}

function identityFor(record) {
  return `tensor-spec-v1:${createHash('sha256').update(JSON.stringify(record)).digest('hex')}`;
}

function normalizedInput(first, second, defaults) {
  if (first instanceof TensorSpec) return first;
  if (typeof first === 'string' && Array.isArray(second)) return { dtype: first, capacityShape: second, access: defaults.access };
  if (Array.isArray(first) && second === undefined) return { dtype: defaults.dtype, capacityShape: first, access: defaults.access };
  if (!plainObject(first) || second !== undefined) fail('TENSOR_SPEC_OPTIONS_INVALID', 'validation', 'TensorSpec requires an exact record, a shape array, or a dtype/shape pair.');
  if (Object.keys(first).some((key) => !SPEC_FIELDS.has(key)) || !Object.hasOwn(first, 'capacityShape')) {
    fail('TENSOR_SPEC_OPTIONS_INVALID', 'validation', 'TensorSpec options contain unknown fields or omit capacityShape.');
  }
  return {
    ...first,
    dtype: first.dtype ?? defaults.dtype,
    access: first.access ?? defaults.access,
  };
}

export function tensorDtypeWidth(dtype) {
  return DTYPE_WIDTH[dtype] ?? null;
}

export class TensorSpec {
  constructor(token, record) {
    if (token !== SPEC_TOKEN) fail('TENSOR_SPEC_CONSTRUCTION_INVALID', 'validation', 'Use TensorSpec.create().');
    Object.assign(this, record);
    Object.freeze(this);
  }

  static create(first, second) {
    return createTensorSpec(first, second);
  }

  toJSON() {
    return this.canonical;
  }
}

export function createTensorSpec(first, second, defaults = { dtype: 'f32', access: 'read-write' }) {
  const input = normalizedInput(first, second, defaults);
  if (input instanceof TensorSpec) return input;
  if (!DTYPE_SET.has(input.dtype)) fail('TENSOR_SPEC_DTYPE_INVALID', 'validation', 'dtype is not in the accepted CUDA-JS tensor dtype registry.', { dtype: input.dtype ?? null });
  if (!ACCESS_SET.has(input.access)) fail('TENSOR_SPEC_ACCESS_INVALID', 'validation', 'access must be read, write, or read-write.', { access: input.access ?? null });

  const capacityShape = normalizeShape(input.capacityShape, 'capacityShape');
  const capacityElementCount = shapeProduct(capacityShape, 'capacityElementCount');
  const defaultStrides = rowMajorStrides(capacityShape);
  const strides = normalizeStrides(input.strides, capacityShape, defaultStrides);
  const activeAxis0 = normalizeActiveAxis0(input.activeAxis0, capacityShape);
  const logicalShape = Object.freeze(capacityShape.map((dimension, index) => (index === 0 && activeAxis0 ? activeAxis0.extent : dimension)));
  const logicalElementCount = shapeProduct(logicalShape, 'logicalElementCount');
  const width = DTYPE_WIDTH[input.dtype];
  const byteOffset = nonnegativeSafeInteger(input.byteOffset ?? 0, 'byteOffset');
  const alignment = input.alignment ?? width;
  if (alignment !== width) {
    fail('TENSOR_SPEC_ALIGNMENT_UNSUPPORTED', 'unsupported', 'V1 required alignment must equal the dtype width because no stronger public CUDA-JS guarantee exists.', { alignment, dtypeWidth: width });
  }
  if (byteOffset % alignment !== 0) fail('TENSOR_SPEC_ALIGNMENT_INVALID', 'validation', 'byteOffset does not satisfy required alignment.', { byteOffset, alignment });

  const storageElementSpan = storageSpan(capacityShape, strides, capacityElementCount);
  const byteLength = checkedMultiply(storageElementSpan, width, 'byteLength');
  const requiredByteEnd = checkedAdd(byteOffset, byteLength, 'requiredByteEnd');
  const hasBroadcastAliasing = capacityElementCount > 0 && capacityShape.some((dimension, index) => dimension > 1 && strides[index] === 0);
  if (hasBroadcastAliasing && input.access !== 'read') {
    fail('TENSOR_SPEC_BROADCAST_WRITE_UNSUPPORTED', 'unsupported', 'Zero-stride broadcast tensor views are read-only in v1.');
  }
  const layout = strides.every((stride, index) => stride === defaultStrides[index]) ? 'row-major-contiguous' : 'strided';
  const aliasGroup = normalizeAliasGroup(input.aliasGroup);
  const canonical = deepFreeze({
    contract: TENSOR_SPEC_CONTRACT,
    dtype: input.dtype,
    dtypeWidth: width,
    rank: capacityShape.length,
    capacityShape: [...capacityShape],
    activeAxis0: activeAxis0 ? { ...activeAxis0 } : null,
    logicalShape: [...logicalShape],
    strides: [...strides],
    layout,
    byteOffset,
    alignment,
    access: input.access,
    aliasGroup,
    capacityElementCount,
    logicalElementCount,
    storageElementSpan,
    byteLength,
    requiredByteEnd,
    hasBroadcastAliasing,
  });
  return new TensorSpec(SPEC_TOKEN, { ...canonical, compatibilityIdentity: identityFor(canonical), canonical });
}
