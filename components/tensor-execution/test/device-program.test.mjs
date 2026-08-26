import assert from 'node:assert/strict';
import test from 'node:test';

import { compileDeviceProgram } from 'cuda-js';
import { discoverCudaDevicesForTesting, openCudaRuntimeForTesting } from 'cuda-js/testing';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { compileTensorDeviceProgram } from '../index.mjs';
import { createDeviceItemProfile } from '../testing.mjs';

function denseProgram() {
  return TensorProgram.define((graph) => {
    const features = graph.input('features', { dtype: 'f32', capacityShape: [4, 3], access: 'read' });
    const weights = graph.input('weights', { dtype: 'f32', capacityShape: [3, 2], access: 'read' });
    const bias = graph.input('bias', { dtype: 'f32', capacityShape: [2], access: 'read' });
    const projected = graph.matmul(features, weights);
    const biased = graph.binary('add', projected, bias);
    const score = graph.reduce('sum', biased, { axes: [1], order: 'fixed-tree-v1' });
    return { values: biased, score };
  });
}

test('item-parallel profile owns exact item independence, ABI, workspace, and private source', () => {
  const plan = TensorPlan.create(denseProgram());
  const first = createDeviceItemProfile(plan, { itemCapacity: 4, itemInputs: ['features'], output: 'ptx', maxWorkspaceBytes: 1 << 20 });
  const second = createDeviceItemProfile(plan, { itemCapacity: 4, itemInputs: ['features'], output: 'ptx', maxWorkspaceBytes: 1 << 20 });

  assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
  assert.equal(first.lowering.source, second.lowering.source);
  assert.deepEqual(first.inputs.map((entry) => [entry.name, entry.itemVarying]), [
    ['features', true],
    ['weights', false],
    ['bias', false],
  ]);
  assert.deepEqual(first.outputs.map((entry) => [entry.name, entry.perItemElements]), [['values', 2], ['score', 1]]);
  assert.deepEqual(first.parameters.map((entry) => entry.role), ['item-index', 'input', 'input', 'input', 'output', 'output', 'workspace']);
  assert(first.totalWorkspaceBytes > 0);
  assert.match(first.lowering.source, /^function tensorRunItem\(itemIndex, input0, input1, input2, output0, output1, workspace0\)/u);
  assert.match(first.lowering.source, /if \(itemIndex >= gpu\.u32\(4\)\)/u);
  assert.match(first.lowering.source, /for \(let k = gpu\.u64\(0n\)/u);
  assert.match(first.lowering.source, /ScratchBase/u);
  assert.doesNotMatch(first.lowering.source, /gpu\.thread|gpu\.block|gpu\.barrier|gpu\.atomic|#include|__device__|cuda[A-Z]|MCGS|evaluator/u);
  assert.equal(first.canonical.totalWorkspaceBytes, first.totalWorkspaceBytes);
});

test('item input sets normalize in program order and caller arrays are copied', () => {
  const program = TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    return graph.binary('add', left, right);
  });
  const plan = TensorPlan.create(program);
  const names = ['right', 'left'];
  const first = createDeviceItemProfile(plan, { itemCapacity: 4, itemInputs: names });
  names.reverse();
  const second = createDeviceItemProfile(plan, { itemCapacity: 4, itemInputs: ['left', 'right'] });
  assert.deepEqual(first.itemInputs, ['left', 'right']);
  assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
});

test('item classifier rejects cross-item and ambiguous shared work before compiler dispatch', () => {
  const crossItem = TensorProgram.define((graph) => {
    const input = graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    return graph.reduce('sum', input, { axes: [0] });
  });
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(crossItem), { itemCapacity: 4, itemInputs: ['items'] }), (error) => error.code === 'TENSOR_DEVICE_CROSS_ITEM_REDUCTION');

  const sharedMaterial = TensorProgram.define((graph) => {
    const shared = graph.input('shared', { dtype: 'f32', capacityShape: [2], access: 'read' });
    return graph.unary('neg', shared);
  });
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(sharedMaterial), { itemCapacity: 4, itemInputs: ['shared'] }), (error) => error.code === 'TENSOR_DEVICE_ITEM_INPUT_INVALID');

  const active = TensorProgram.define((graph) => {
    const input = graph.input('items', { dtype: 'f32', capacityShape: [4, 2], activeAxis0: 2, access: 'read' });
    return graph.unary('neg', input);
  });
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(active), { itemCapacity: 4, itemInputs: ['items'] }), (error) => error.code === 'TENSOR_DEVICE_ACTIVE_AXIS_UNSUPPORTED');
});

test('rank-3 item matmul admits shared batch-one weights and rejects shared varying batches', () => {
  const accepted = TensorProgram.define((graph) => {
    const items = graph.input('items', { dtype: 'f32', capacityShape: [4, 2, 3], access: 'read' });
    const weights = graph.input('weights', { dtype: 'f32', capacityShape: [1, 3, 2], access: 'read' });
    return graph.matmul(items, weights);
  });
  const profile = createDeviceItemProfile(TensorPlan.create(accepted), { itemCapacity: 4, itemInputs: ['items'] });
  assert.equal(profile.outputs[0].perItemElements, 4);

  const rejected = TensorProgram.define((graph) => {
    const items = graph.input('items', { dtype: 'f32', capacityShape: [4, 2, 3], access: 'read' });
    const weights = graph.input('weights', { dtype: 'f32', capacityShape: [4, 3, 2], access: 'read' });
    return graph.matmul(items, weights);
  });
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(rejected), { itemCapacity: 4, itemInputs: ['items'] }), (error) => error.code === 'TENSOR_DEVICE_SHARED_INPUT_VARIES');
});

test('item-preserving views, materialization, casts, unary, and shared broadcasting stay one removable profile', () => {
  const program = TensorProgram.define((graph) => {
    const items = graph.input('items', { dtype: 'f32', capacityShape: [4, 2, 3], access: 'read' });
    const bias = graph.input('bias', { dtype: 'f64', capacityShape: [3], access: 'read' });
    const reshaped = graph.reshape(items, [4, 3, 2]);
    const permuted = graph.permute(reshaped, [0, 2, 1]);
    const sliced = graph.slice(permuted, [null, { start: 0, length: 1 }, null]);
    const broadcast = graph.broadcast(sliced, [4, 2, 3]);
    const contiguous = graph.contiguous(broadcast);
    const widened = graph.cast(contiguous, 'f64');
    return graph.binary('add', graph.unary('abs', widened), bias);
  });
  const profile = createDeviceItemProfile(TensorPlan.create(program), { itemCapacity: 4, itemInputs: ['items'] });
  assert.deepEqual(profile.workspace.map((entry) => entry.dtype), ['f32', 'f64']);
  assert.match(profile.lowering.source, /gpu\.cast\.f64/u);
  assert.match(profile.lowering.source, /gpu\.math\.abs/u);
  assert.match(profile.lowering.source, /input1\[/u);

  const movedAxis = TensorProgram.define((graph) => graph.permute(
    graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' }),
    [1, 0],
  ));
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(movedAxis), { itemCapacity: 4, itemInputs: ['items'] }), (error) => error.code === 'TENSOR_DEVICE_ITEM_VIEW_INVALID');
});

test('direct item outputs need no workspace and finite ABI/workspace gates fail closed', () => {
  const direct = TensorProgram.define((graph) => graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' }));
  const directProfile = createDeviceItemProfile(TensorPlan.create(direct), { itemCapacity: 4, itemInputs: ['items'], maxWorkspaceBytes: 0 });
  assert.equal(directProfile.totalWorkspaceBytes, 0);
  assert.equal(directProfile.workspace.length, 0);
  assert.deepEqual(directProfile.parameters.map((entry) => entry.role), ['item-index', 'input', 'output']);

  const material = TensorProgram.define((graph) => graph.unary('neg', graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' })));
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(material), { itemCapacity: 4, itemInputs: ['items'], maxWorkspaceBytes: 0 }), (error) => error.code === 'TENSOR_DEVICE_WORKSPACE_LIMIT');

  const tooManyOutputs = TensorProgram.define((graph) => {
    const input = graph.input('items', { dtype: 'f32', capacityShape: [4], access: 'read' });
    return Object.fromEntries(Array.from({ length: 63 }, (_, index) => [`output${index}`, input]));
  });
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(tooManyOutputs), { itemCapacity: 4, itemInputs: ['items'] }), (error) => error.code === 'TENSOR_DEVICE_PARAMETER_LIMIT');
});

test('public Tensor device program compiles and composes as one copied CUDA-JS leaf library', { timeout: 15_000 }, async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  const session = await TensorSession.open(runtime);
  try {
    const deviceProgram = await compileTensorDeviceProgram(session, denseProgram(), { itemCapacity: 4, itemInputs: ['features'] });
    assert.equal(deviceProgram.kind, 'tensor-device-program');
    assert.equal(deviceProgram.function.name, 'tensorRunItem');
    assert.equal(deviceProgram.function.returns, 'u32');
    assert.equal(deviceProgram.library.format, 'ptx');
    assert.equal(deviceProgram.parameters.length, deviceProgram.function.parameters.length);
    assert.equal(JSON.stringify(deviceProgram).includes('function tensorRunItem'), false);
    assert.equal(JSON.stringify(deviceProgram.canonical).includes('__device__'), false);
    assert.deepEqual(deviceProgram.importAs('evaluateItem'), { library: deviceProgram.library, name: 'tensorRunItem', as: 'evaluateItem' });
    assert.throws(() => deviceProgram.importAs('gpu'), (error) => error.code === 'TENSOR_DEVICE_IMPORT_ALIAS_INVALID');

    const detached = deviceProgram.library;
    detached.artifact.bytes[0] ^= 0xff;
    assert.notEqual(detached.artifact.bytes[0], deviceProgram.library.artifact.bytes[0]);

    const repeated = await compileTensorDeviceProgram(session, denseProgram(), { itemCapacity: 4, itemInputs: ['features'] });
    assert.equal(repeated.compatibilityIdentity, deviceProgram.compatibilityIdentity);
    assert.equal(repeated.library.sha256, deviceProgram.library.sha256);
    assert.deepEqual(repeated.function, deviceProgram.function);

    const lto = await compileTensorDeviceProgram(session, denseProgram(), { itemCapacity: 4, itemInputs: ['features'], output: 'lto-ir' });
    assert.equal(lto.outputFormat, 'lto-ir');
    assert.equal(lto.library.format, 'lto-ir');
    assert.notEqual(lto.compatibilityIdentity, deviceProgram.compatibilityIdentity);

    const kernelParameters = [
      ...deviceProgram.function.parameters,
      { name: 'status', type: 'ptr<u32>' },
    ];
    const callArguments = deviceProgram.function.parameters.map((entry) => entry.name).join(', ');
    const source = `function consumer(${kernelParameters.map((entry) => entry.name).join(', ')}) { status[gpu.u32(0)] = evaluateItem(${callArguments}); }`;
    const composed = await compileDeviceProgram(runtime, {
      source,
      functions: [{ name: 'consumer', kind: 'kernel', parameters: kernelParameters, returns: 'void' }],
      imports: [deviceProgram.importAs('evaluateItem')],
    });
    assert.equal(composed.deviceProgram.imports[0].exportName, 'tensorRunItem');
    assert.equal(composed.linker.artifact.format, 'cubin');
  } finally {
    assert.equal((await session.close()).graceful, true);
    assert.equal((await runtime.close()).graceful, true);
  }
});

test('independent Tensor sessions preserve each selected CUDA-JS device target', { timeout: 15_000 }, async () => {
  const snapshot = await discoverCudaDevicesForTesting([
    { nativeDevice: 0, computeCapabilityMajor: 7, computeCapabilityMinor: 5 },
    { nativeDevice: 7, computeCapabilityMajor: 8, computeCapabilityMinor: 9 },
  ]);
  const baselineRuntime = await openCudaRuntimeForTesting({ device: snapshot.devices[0].selector, compiler: true });
  const newerRuntime = await openCudaRuntimeForTesting({ device: snapshot.devices[1].selector, compiler: true });
  const baselineSession = await TensorSession.open(baselineRuntime);
  const newerSession = await TensorSession.open(newerRuntime);
  try {
    const options = { itemCapacity: 4, itemInputs: ['features'] };
    const [baseline, newer] = await Promise.all([
      compileTensorDeviceProgram(baselineSession, denseProgram(), options),
      compileTensorDeviceProgram(newerSession, denseProgram(), options),
    ]);

    assert.equal(baseline.library.architecture, 'compute_75');
    assert.equal(newer.library.architecture, 'compute_89');
    assert.equal(baseline.canonical.compiled.library.architecture, 'compute_75');
    assert.equal(newer.canonical.compiled.library.architecture, 'compute_89');
    assert.notEqual(baseline.compatibilityIdentity, newer.compatibilityIdentity);
  } finally {
    assert.equal((await baselineSession.close()).graceful, true);
    assert.equal((await newerSession.close()).graceful, true);
    assert.equal((await baselineRuntime.close()).graceful, true);
    assert.equal((await newerRuntime.close()).graceful, true);
  }
});
