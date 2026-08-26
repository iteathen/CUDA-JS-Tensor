import assert from 'node:assert/strict';
import test from 'node:test';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { TensorError, TensorSession, TensorSpec } from '../index.mjs';
import { inspectTensorForSession } from '../internal.mjs';

function expectCode(code) {
  return (error) => error?.code === code;
}

function fakeRuntime({ allocationFailure = null, viewFailure = null, viewCloseFailures = [], memoryCloseFailure = null, version = '0.1.0-alpha.15' } = {}) {
  let runtimeClosed = false;
  let memoryClosed = false;
  let viewClosed = false;
  return {
    get runtimeClosed() { return runtimeClosed; },
    get memoryClosed() { return memoryClosed; },
    async describe() {
      return { package: { name: 'cuda-js', version, publicApiSchema: 1 }, state: runtimeClosed ? 'closed' : 'open', profile: 'test-double', device: { architecture: { class: 'compute_75' } } };
    },
    async allocateDevice({ byteLength }) {
      if (allocationFailure) throw allocationFailure;
      const storage = new Uint8Array(byteLength);
      return {
        byteLength,
        async write(bytes, { deviceOffset = 0 } = {}) { storage.set(Uint8Array.from(bytes), deviceOffset); return { byteLength: bytes.byteLength }; },
        async read({ deviceOffset = 0, byteLength: readLength }) { return { bytes: Uint8Array.from(storage.subarray(deviceOffset, deviceOffset + readLength)) }; },
        async view(options) {
          if (viewFailure) throw viewFailure;
          return {
            ...options,
            async status() { return { state: viewClosed ? 'closed' : 'open' }; },
            async close() {
              const failure = viewCloseFailures.shift();
              if (failure) throw failure;
              viewClosed = true;
              return { state: 'closed' };
            },
          };
        },
        async close() {
          if (memoryCloseFailure) throw memoryCloseFailure;
          memoryClosed = true;
          return { state: 'closed' };
        },
      };
    },
    async close() { runtimeClosed = true; return { graceful: true }; },
  };
}

test('borrowed session overloads normalize defaults and preserve caller runtime ownership', async () => {
  const runtime = await openCudaRuntimeForTesting();
  const session = await TensorSession.open(runtime);
  assert.equal(session.kind, 'tensor-session');
  assert.equal(session.ownershipMode, 'borrowed');
  assert.equal(session.resolvedOpenOptions.runtimeSource, 'injected');
  assert.equal(session.resolvedOpenOptions.deviceSelection, 'injected');
  assert.deepEqual(session.defaults, { dtype: 'f32', access: 'read-write' });

  const easy = await session.allocate([2, 3]);
  const explicit = await session.allocate(TensorSpec.create({ dtype: 'f32', capacityShape: [2, 3], strides: [3, 1], access: 'read-write' }));
  assert.deepEqual(Object.keys(session), []);
  assert.deepEqual(Object.keys(easy), []);
  assert.equal(JSON.stringify(easy), '{}');
  assert.equal(easy.spec.compatibilityIdentity, explicit.spec.compatibilityIdentity);
  assert.deepEqual((await session.status()).accounting, { liveTensors: 2, reservedBytes: 48, pendingCreates: 0, resolvedPlans: 0, pendingResolutions: 0, unprovedResources: 0 });

  await easy.close();
  await explicit.close();
  const terminal = await session.close();
  assert.equal(terminal.graceful, true);
  assert.equal(terminal.runtimeClosed, false);
  assert.equal((await runtime.describe()).state, 'open');
  assert.equal((await runtime.close()).graceful, true);
});

test('explicit owned injection transfers runtime close authority and session close orders tensor children', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open({ runtime, runtimeOwnership: 'owned', limits: { maxTensorBytes: 64, maxSessionBytes: 96, maxLiveTensors: 4 }, defaults: { dtype: 'i32', access: 'read-write' } });
  const root = await session.allocate([4, 4]);
  const child = await root.view({ capacityShape: [2, 4], strides: [4, 1], byteOffset: 16, access: 'read' });
  assert.equal(root.aliasGroup, child.aliasGroup);
  assert.equal((await root.status()).childCount, 1);
  await assert.rejects(root.close(), expectCode('TENSOR_HAS_CHILDREN'));

  const terminal = await session.close();
  assert.equal(terminal.graceful, true);
  assert.equal(terminal.runtimeClosed, true);
  assert.equal(runtime.runtimeClosed, true);
  assert.equal(runtime.memoryClosed, true);
  assert.equal(root.state, 'closed');
  assert.equal(child.state, 'closed');
});

test('views preserve dtype/range/access authority and cross-session use rejects before CUDA work', async () => {
  const leftRuntime = await openCudaRuntimeForTesting();
  const rightRuntime = await openCudaRuntimeForTesting();
  const left = await TensorSession.open(leftRuntime);
  const right = await TensorSession.open(rightRuntime);
  const root = await left.allocate({ dtype: 'f32', capacityShape: [4, 4], access: 'read' });

  const transpose = await root.view({ capacityShape: [4, 4], strides: [1, 4], access: 'read' });
  assert.equal(transpose.spec.layout, 'strided');
  await assert.rejects(transpose.view({ capacityShape: [2, 4], strides: [1, 4], access: 'read' }), expectCode('TENSOR_VIEW_STRIDED_PARENT_UNSUPPORTED'));
  assert.throws(() => inspectTensorForSession(right, root), expectCode('TENSOR_CROSS_SESSION'));
  await assert.rejects(root.view({ dtype: 'i32', capacityShape: [4, 4], access: 'read' }), expectCode('TENSOR_VIEW_DTYPE_MISMATCH'));
  await assert.rejects(root.view({ capacityShape: [4, 4], access: 'write' }), expectCode('TENSOR_VIEW_ACCESS_DENIED'));
  await assert.rejects(root.view({ capacityShape: [5, 4], access: 'read' }), expectCode('TENSOR_VIEW_RANGE_OUT_OF_BOUNDS'));

  const internal = inspectTensorForSession(left, transpose);
  assert.equal(internal.spec, transpose.spec);
  assert.equal(internal.deviceView.kind, 'device-view');
  await transpose.close();
  await root.close();
  await left.close();
  await right.close();
  await leftRuntime.close();
  await rightRuntime.close();
});

test('empty tensors use explicit minimal storage while logical and view byte ranges remain empty', async () => {
  const runtime = await openCudaRuntimeForTesting({ driver: { memory: { maxDeviceBytes: 8, maxAllocationBytes: 8, maxTransferBytes: 8 } } });
  const session = await TensorSession.open({ runtime, limits: { maxTensorBytes: 4, maxSessionBytes: 4, maxLiveTensors: 2 } });
  const empty = await session.allocate({ dtype: 'f32', capacityShape: [0, 8] });
  const status = await empty.status();
  assert.equal(status.spec.byteLength, 0);
  assert.equal(status.allocationBytes, 4);
  assert.equal((await session.status()).accounting.reservedBytes, 4);
  await empty.close();
  await session.close();
  await runtime.close();
});

test('session limits reserve before asynchronous work and recover after ordinary allocation failure', async () => {
  const runtime = await openCudaRuntimeForTesting({ driver: { memory: { maxDeviceBytes: 32, maxAllocationBytes: 16, maxTransferBytes: 16 } } });
  const session = await TensorSession.open({ runtime, limits: { maxTensorBytes: 16, maxSessionBytes: 16, maxLiveTensors: 1 } });
  const tensorPromise = session.allocate('f32', [4]);
  await assert.rejects(session.allocate('f32', [1]), expectCode('TENSOR_SESSION_LIVE_LIMIT'));
  await assert.rejects(session.close(), expectCode('TENSOR_SESSION_BUSY'));
  const tensor = await tensorPromise;
  await tensor.close();
  await assert.rejects(session.allocate('f64', [3]), expectCode('TENSOR_ALLOCATION_LIMIT'));
  assert.deepEqual((await session.status()).accounting, { liveTensors: 0, reservedBytes: 0, pendingCreates: 0, resolvedPlans: 0, pendingResolutions: 0, unprovedResources: 0 });
  await session.close();
  await runtime.close();
});

test('CUDA-JS allocation and view admission failures release provisional session accounting when rollback succeeds', async () => {
  const allocationFailure = Object.assign(new Error('allocation failed'), { code: 'MEMORY_ALLOCATION_FAILED', category: 'pressure' });
  const allocationRuntime = fakeRuntime({ allocationFailure });
  const allocationSession = await TensorSession.open(allocationRuntime);
  await assert.rejects(allocationSession.allocate('f32', [4]), expectCode('MEMORY_ALLOCATION_FAILED'));
  assert.deepEqual((await allocationSession.status()).accounting, { liveTensors: 0, reservedBytes: 0, pendingCreates: 0, resolvedPlans: 0, pendingResolutions: 0, unprovedResources: 0 });
  await allocationSession.close();
  await allocationRuntime.close();

  const viewFailure = Object.assign(new Error('view failed'), { code: 'MEMORY_VIEW_FAILED', category: 'validation' });
  const viewRuntime = fakeRuntime({ viewFailure });
  const viewSession = await TensorSession.open(viewRuntime);
  await assert.rejects(viewSession.allocate('f32', [4]), expectCode('MEMORY_VIEW_FAILED'));
  assert.deepEqual((await viewSession.status()).accounting, { liveTensors: 0, reservedBytes: 0, pendingCreates: 0, resolvedPlans: 0, pendingResolutions: 0, unprovedResources: 0 });
  assert.equal(viewRuntime.memoryClosed, true);
  await viewSession.close();
  await viewRuntime.close();
});

test('retriable view-close backpressure preserves the tensor for a later terminal close', async () => {
  const busy = Object.assign(new Error('view leased'), { code: 'RESOURCE_BUSY', category: 'backpressure' });
  const runtime = fakeRuntime({ viewCloseFailures: [busy] });
  const session = await TensorSession.open(runtime);
  const tensor = await session.allocate('f32', [4]);
  await assert.rejects(tensor.close(), expectCode('RESOURCE_BUSY'));
  assert.equal(tensor.state, 'open');
  assert.equal((await session.status()).accounting.reservedBytes, 16);
  assert.equal((await tensor.close()).disposition, 'closed');
  assert.equal((await session.close()).graceful, true);
  await runtime.close();
});

test('public tensor byte transfers snapshot exact contiguous logical storage and enforce access', async () => {
  const runtime = fakeRuntime();
  const session = await TensorSession.open(runtime);
  const tensor = await session.allocate({ dtype: 'u32', capacityShape: [2], access: 'read-write' });
  const input = Uint8Array.of(1, 2, 3, 4, 5, 6, 7, 8);
  assert.equal((await tensor.write(input)).byteLength, 8);
  input.fill(0);
  const first = await tensor.read();
  assert.deepEqual([...first.bytes], [1, 2, 3, 4, 5, 6, 7, 8]);
  first.bytes.fill(9);
  assert.deepEqual([...(await tensor.read()).bytes], [1, 2, 3, 4, 5, 6, 7, 8]);
  const readOnly = await tensor.view({ dtype: 'u32', capacityShape: [2], access: 'read' });
  await assert.rejects(readOnly.write(new Uint8Array(8)), expectCode('TENSOR_WRITE_ACCESS_DENIED'));
  await readOnly.close();
  const strided = await tensor.view({ dtype: 'u32', capacityShape: [1], strides: [2], access: 'read' });
  await assert.rejects(strided.read(), expectCode('TENSOR_TRANSFER_LAYOUT_UNSUPPORTED'));
  await strided.close();
  await tensor.close();
  await session.close();
  await runtime.close();
});

test('non-retriable allocation close failure remains explicit and never recovers reserved bytes', async () => {
  const closeFailure = Object.assign(new Error('free unproved'), { code: 'MEMORY_FREE_UNPROVED', category: 'restart-required' });
  const runtime = fakeRuntime({ memoryCloseFailure: closeFailure });
  const session = await TensorSession.open(runtime);
  const tensor = await session.allocate('f32', [4]);
  const tensorReport = await tensor.close();
  assert.equal(tensorReport.disposition, 'cleanup-unproved');
  assert.equal(tensor.state, 'orphaned');
  assert.deepEqual((await session.status()).accounting, { liveTensors: 0, reservedBytes: 16, pendingCreates: 0, resolvedPlans: 0, pendingResolutions: 0, unprovedResources: 1 });
  const sessionReport = await session.close();
  assert.equal(sessionReport.graceful, false);
  assert.equal(sessionReport.state, 'orphaned');
  await runtime.close();
});

test('allocation rollback retains capacity and reports cleanup as unproved', async () => {
  const viewFailure = Object.assign(new Error('view admission failed'), { code: 'MEMORY_VIEW_FAILURE', category: 'validation' });
  const closeFailure = Object.assign(new Error('free failed'), { code: 'MEMORY_FREE_FAILURE', category: 'restart-required' });
  const runtime = fakeRuntime({ viewFailure, memoryCloseFailure: closeFailure });
  const session = await TensorSession.open(runtime);
  await assert.rejects(session.allocate('f32', [4]), (error) => error instanceof TensorError && error.code === 'TENSOR_ALLOCATION_ROLLBACK_UNPROVED');
  assert.deepEqual((await session.status()).accounting, { liveTensors: 0, reservedBytes: 16, pendingCreates: 0, resolvedPlans: 0, pendingResolutions: 0, unprovedResources: 1 });
  const report = await session.close();
  assert.equal(report.graceful, false);
  assert.equal(report.state, 'orphaned');
  assert.equal(runtime.runtimeClosed, false);
  await runtime.close();
});

test('incompatible transferred runtime is closed during session-open rollback', async () => {
  const runtime = fakeRuntime({ version: '0.1.0-alpha.11' });
  await assert.rejects(TensorSession.open({ runtime, runtimeOwnership: 'owned' }), expectCode('TENSOR_CUDA_JS_RUNTIME_INCOMPATIBLE'));
  assert.equal(runtime.runtimeClosed, true);
});
