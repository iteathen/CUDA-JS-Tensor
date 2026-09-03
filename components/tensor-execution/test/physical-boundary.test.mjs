import assert from 'node:assert/strict';
import test from 'node:test';

import { CUDA_JS_COMPATIBILITY } from 'cuda-js/compatibility';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TENSOR_SIMT_LIMITS } from '../src/contract.mjs';
import { CUDA_JS_TENSOR_COMPATIBILITY, requireTensorDeviceLibraryOutput } from '../src/cuda-js-compatibility.mjs';
import { createDeviceItemProfile, TENSOR_DEVICE_PROGRAM_LIMITS } from '../src/device-item-profile.mjs';
import { lowerSimtPlan } from '../src/lowering.mjs';

test('Tensor physical compatibility facts come from the supported CUDA-JS compatibility entry', () => {
  assert.deepEqual(
    CUDA_JS_TENSOR_COMPATIBILITY.preparedOperationDagLimits,
    CUDA_JS_COMPATIBILITY.capabilities.preparedOperationDagLimits,
  );
  assert.deepEqual(
    CUDA_JS_TENSOR_COMPATIBILITY.deviceJsLimits,
    CUDA_JS_COMPATIBILITY.capabilities.deviceJsLimits,
  );
  assert.deepEqual(
    CUDA_JS_TENSOR_COMPATIBILITY.compilerOutputFormats,
    CUDA_JS_COMPATIBILITY.capabilities.compilerOutputFormats,
  );
  assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.package.version, '0.1.0-alpha.17');

  assert.equal(Object.hasOwn(TENSOR_SIMT_LIMITS, 'maxKernels'), false);
  assert.equal(Object.hasOwn(TENSOR_SIMT_LIMITS, 'maxBindings'), false);
  assert.equal(Object.hasOwn(TENSOR_DEVICE_PROGRAM_LIMITS, 'maxParameters'), false);
});

test('host and device physical identities carry lower compatibility under explicit CUDA-JS records', () => {
  const hostProgram = TensorProgram.define((graph) => graph.unary(
    'neg',
    graph.input('values', { dtype: 'f32', capacityShape: [4], access: 'read' }),
  ));
  const host = lowerSimtPlan(TensorPlan.create(hostProgram));
  assert.deepEqual(
    host.canonical.limits.cudaJsPreparedOperationDagLimits,
    CUDA_JS_COMPATIBILITY.capabilities.preparedOperationDagLimits,
  );
  assert.equal(host.canonical.blockSize, 256);
  assert.deepEqual(host.kernels[0].grid, { x: 1, y: 1, z: 1 });
  assert.deepEqual(host.kernels[0].block, { x: 256, y: 1, z: 1 });

  const deviceProgram = TensorProgram.define((graph) => graph.unary(
    'neg',
    graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' }),
  ));
  const device = createDeviceItemProfile(TensorPlan.create(deviceProgram), {
    itemCapacity: 4,
    itemInputs: ['items'],
    output: 'ptx',
  });
  assert.deepEqual(device.canonical.cudaJsCompatibility.deviceJsLimits, CUDA_JS_COMPATIBILITY.capabilities.deviceJsLimits);
  assert.deepEqual(device.canonical.cudaJsCompatibility.compilerOutputFormats, CUDA_JS_COMPATIBILITY.capabilities.compilerOutputFormats);
  assert.equal(device.output, 'ptx');
  assert.equal(requireTensorDeviceLibraryOutput('lto-ir'), 'lto-ir');
  assert.throws(
    () => requireTensorDeviceLibraryOutput('cubin'),
    (error) => error.code === 'TENSOR_DEVICE_PROGRAM_OUTPUT_UNSUPPORTED_BY_CUDA_JS' && error.category === 'unsupported',
  );
});
