import assert from 'node:assert/strict';
import test from 'node:test';
import { createTensorSpec, TensorSpec, tensorDtypeWidth } from '../index.mjs';

function expectCode(code) {
  return (error) => error?.code === code;
}

test('convenience and expert TensorSpec forms normalize to one immutable identity', () => {
  const shape = [2, 3, 4];
  const easy = TensorSpec.create('f32', shape);
  const shorthand = createTensorSpec(shape);
  const expert = TensorSpec.create({ dtype: 'f32', capacityShape: [2, 3, 4], strides: [12, 4, 1], byteOffset: 0, alignment: 4, access: 'read-write' });
  shape[0] = 99;

  assert.equal(easy.compatibilityIdentity, shorthand.compatibilityIdentity);
  assert.equal(easy.compatibilityIdentity, expert.compatibilityIdentity);
  assert.deepEqual(easy.capacityShape, [2, 3, 4]);
  assert.deepEqual(easy.strides, [12, 4, 1]);
  assert.equal(easy.layout, 'row-major-contiguous');
  assert.equal(easy.capacityElementCount, 24);
  assert.equal(easy.storageElementSpan, 24);
  assert.equal(easy.byteLength, 96);
  assert(Object.isFrozen(easy));
  assert(Object.isFrozen(easy.capacityShape));
  assert(Object.isFrozen(easy.canonical));
});

test('rank-zero, empty dimensions, and bounded active axis zero remain distinct', () => {
  const scalar = TensorSpec.create({ dtype: 'f64', capacityShape: [] });
  assert.equal(scalar.rank, 0);
  assert.equal(scalar.capacityElementCount, 1);
  assert.equal(scalar.storageElementSpan, 1);
  assert.equal(scalar.byteLength, 8);

  const empty = TensorSpec.create({ dtype: 'f32', capacityShape: [8, 0, 3] });
  assert.equal(empty.capacityElementCount, 0);
  assert.equal(empty.storageElementSpan, 0);
  assert.equal(empty.byteLength, 0);
  assert.equal(empty.access, 'read-write');

  const active = TensorSpec.create({ dtype: 'f16', capacityShape: [16, 4], activeAxis0: { extent: 5, maximum: 16 } });
  const activeShorthand = TensorSpec.create({ dtype: 'f16', capacityShape: [16, 4], activeAxis0: 5 });
  assert.deepEqual(active.logicalShape, [5, 4]);
  assert.equal(active.logicalElementCount, 20);
  assert.equal(active.capacityElementCount, 64);
  assert.equal(active.storageElementSpan, 64);
  assert.equal(active.compatibilityIdentity, activeShorthand.compatibilityIdentity);
});

test('explicit strides determine a finite storage envelope and broadcast writes fail closed', () => {
  const transposed = TensorSpec.create({ dtype: 'i32', capacityShape: [3, 2], strides: [1, 3], byteOffset: 8, access: 'read' });
  assert.equal(transposed.layout, 'strided');
  assert.equal(transposed.storageElementSpan, 6);
  assert.equal(transposed.byteLength, 24);
  assert.equal(transposed.requiredByteEnd, 32);

  const broadcast = TensorSpec.create({ dtype: 'f32', capacityShape: [5, 7], strides: [0, 1], access: 'read' });
  assert.equal(broadcast.hasBroadcastAliasing, true);
  assert.equal(broadcast.storageElementSpan, 7);
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [5, 7], strides: [0, 1], access: 'read-write' }), expectCode('TENSOR_SPEC_BROADCAST_WRITE_UNSUPPORTED'));
});

test('dtype, rank, range, alignment, active extent, and option errors reject deterministically', () => {
  assert.equal(tensorDtypeWidth('bf16'), 2);
  assert.equal(tensorDtypeWidth('unknown'), null);
  assert.throws(() => TensorSpec.create({ dtype: 'bad', capacityShape: [1] }), expectCode('TENSOR_SPEC_DTYPE_INVALID'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: new Array(17).fill(1) }), expectCode('TENSOR_SPEC_SHAPE_INVALID'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [-1] }), expectCode('TENSOR_SPEC_RANGE_INVALID'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [Number.MAX_SAFE_INTEGER, 2] }), expectCode('TENSOR_SPEC_RANGE_OVERFLOW'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [4], byteOffset: 2 }), expectCode('TENSOR_SPEC_ALIGNMENT_INVALID'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [4], alignment: 8 }), expectCode('TENSOR_SPEC_ALIGNMENT_UNSUPPORTED'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [4], activeAxis0: 5 }), expectCode('TENSOR_SPEC_ACTIVE_AXIS_INVALID'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [], activeAxis0: 0 }), expectCode('TENSOR_SPEC_ACTIVE_AXIS_INVALID'));
  assert.throws(() => TensorSpec.create({ dtype: 'f32', capacityShape: [1], negativeStrides: true }), expectCode('TENSOR_SPEC_OPTIONS_INVALID'));
});
