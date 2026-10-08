import assert from 'node:assert/strict';
import test from 'node:test';
import { CUDA_JS_COMPATIBILITY } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { compileTensorDeviceProgram, TensorProgram, TensorSession } from 'cuda-js-tensor';

test('selected exact public CUDA-JS cohort opens a Tensor session and compiles the unchanged item ABI', async () => {
  assert.equal(CUDA_JS_COMPATIBILITY.package.version, '0.1.0-alpha.21');
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  let session;
  try {
    session = await TensorSession.open(runtime);
    const program = TensorProgram.define((graph) => graph.unary('tanh', graph.input('items', { dtype: 'f32', capacityShape: [2, 3], access: 'read' })));
    const callable = await compileTensorDeviceProgram(session, program, { itemCapacity: 2, itemInputs: ['items'] });
    assert.equal(callable.contract, 'SPEC-0009-item-parallel-device-tensor-program-v1');
    assert.equal(callable.itemCapacity, 2);
    assert.equal(callable.workspace[0].perItemElements, 3);
    assert.equal(callable.totalWorkspaceBytes, 24);
    assert.deepEqual(callable.function.parameters.map(({ type }) => type), ['u32', 'ptr<f32>', 'ptr<f32>', 'ptr<f32>']);
  } finally {
    if (session) assert.equal((await session.close()).graceful, true);
    assert.equal((await runtime.close()).graceful, true);
  }
});

test('selected package still rejects an older injected public runtime description', async () => {
  const runtime = await openCudaRuntimeForTesting();
  const older = {
    async describe() { const description = await runtime.describe(); return { ...description, package: { ...description.package, version: '0.1.0-alpha.18' } }; },
    allocateDevice: runtime.allocateDevice.bind(runtime),
    close: runtime.close.bind(runtime),
  };
  try { await assert.rejects(TensorSession.open(older), { code: 'TENSOR_CUDA_JS_RUNTIME_INCOMPATIBLE' }); }
  finally { assert.equal((await runtime.close()).graceful, true); }
});
