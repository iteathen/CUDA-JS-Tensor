import assert from 'node:assert/strict';
import test from 'node:test';

import { compileDeviceProgram } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { compileTensorDeviceProgram, TENSOR_DEVICE_PROGRAM_CONTRACT } from '../index.mjs';
import { createDeviceItemProfile, lowerSimtPlan } from '../testing.mjs';

function tanhProgram(dtype = 'f32') {
  return TensorProgram.define((graph) => graph.unary(
    'tanh',
    graph.input('items', { dtype, capacityShape: [4, 2], access: 'read' }),
  ));
}

function mixedProgram(dtype = 'f32') {
  return TensorProgram.define((graph) => {
    const input = graph.input('items', { dtype, capacityShape: [4, 2], access: 'read' });
    return graph.unary('tanh', graph.unary('erf', input));
  });
}

test('ordinary f32/f64 tanh lowers only through public gpu.math.tanh and compiles with the exact Device-JS child', { timeout: 20_000 }, async () => {
  for (const dtype of ['f32', 'f64']) {
    const lowering = lowerSimtPlan(TensorPlan.create(tanhProgram(dtype)));
    assert.equal(lowering.kernels.length, 1);
    assert.match(lowering.source, /gpu\.math\.tanh/u);
    assert.doesNotMatch(lowering.source, /__tanhf|gpu\.math\.exp|gelu|approx|#include|__device__|cuda[A-Z]/iu);

    const runtime = await openCudaRuntimeForTesting({ compiler: true });
    try {
      const compiled = await compileDeviceProgram(runtime, { source: lowering.source, functions: lowering.functions });
      assert.match(compiled.deviceProgram.contract, /SPEC-0030-dense-numeric-v1\+SPEC-0030-tanh-v1$/u);
      assert.equal(compiled.compiler.headerProfile, 'cuda-numeric');
    } finally {
      assert.equal((await runtime.close()).graceful, true);
    }
  }
});

test('exact fusion composes erf then tanh into one kernel and one canonical Device-JS erf+tanh identity', { timeout: 20_000 }, async () => {
  const plan = TensorPlan.create(mixedProgram());
  const unfused = lowerSimtPlan(plan);
  const fused = lowerSimtPlan(plan, { fusion: 'exact-elementwise' });
  assert.equal(unfused.kernels.length, 2);
  assert.equal(fused.kernels.length, 1);
  assert.match(fused.source, /gpu\.math\.erf/u);
  assert.match(fused.source, /gpu\.math\.tanh/u);
  assert.equal(fused.fusionProfile.regions.length, 1);
  assert.deepEqual(fused.fusionProfile.regions[0].nodeIds, ['node:0', 'node:1']);
  assert.doesNotMatch(fused.source, /__tanhf|gpu\.math\.exp|gelu|approx/iu);

  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  try {
    const compiled = await compileDeviceProgram(runtime, { source: fused.source, functions: fused.functions });
    assert.match(compiled.deviceProgram.contract, /SPEC-0030-dense-numeric-v1\+SPEC-0030-erf-v1\+SPEC-0030-tanh-v1$/u);
  } finally {
    assert.equal((await runtime.close()).graceful, true);
  }
});

test('device-callable tanh reuses the base SPEC-0009 item ABI/workspace and public helper', () => {
  const profile = createDeviceItemProfile(TensorPlan.create(tanhProgram()), { itemCapacity: 4, itemInputs: ['items'] });
  assert.equal(profile.contract, TENSOR_DEVICE_PROGRAM_CONTRACT);
  assert.match(profile.lowering.source, /gpu\.math\.tanh/u);
  assert.doesNotMatch(profile.lowering.source, /__tanhf|gpu\.math\.exp|gelu|approx|#include|__device__|cuda[A-Z]/iu);
  assert.deepEqual(profile.parameters.map((entry) => entry.role), ['item-index', 'input', 'output', 'workspace']);
  assert.equal(profile.workspace.length, 1);
  assert.equal(profile.workspace[0].dtype, 'f32');
  assert.equal(profile.workspace[0].perItemElements, 2);
  assert.equal(profile.totalWorkspaceBytes, 32);
  assert.match(profile.lowering.source, /if \(itemIndex >= gpu\.u32\(4\)\) \{\n    return gpu\.u32\(1\);/u);
});

test('public Tensor device program compiles tanh as one copied CUDA-JS leaf library without a tanh-specific item child', { timeout: 20_000 }, async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  const session = await TensorSession.open(runtime);
  try {
    const callable = await compileTensorDeviceProgram(session, tanhProgram(), { itemCapacity: 4, itemInputs: ['items'] });
    assert.equal(callable.contract, TENSOR_DEVICE_PROGRAM_CONTRACT);
    assert.equal(callable.function.name, 'tensorRunItem');
    assert.equal(callable.function.returns, 'u32');
    assert.match(callable.library.contract, /SPEC-0030-dense-numeric-v1\+SPEC-0030-tanh-v1\+SPEC-0028-device-library-v1$/u);
    assert.equal(callable.outputFormat, 'ptx');
    assert.equal(JSON.stringify(callable).includes('gpu.math.tanh'), false);
  } finally {
    assert.equal((await session.close()).graceful, true);
    assert.equal((await runtime.close()).graceful, true);
  }
});
