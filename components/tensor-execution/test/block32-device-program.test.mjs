import assert from 'node:assert/strict';
import test from 'node:test';
import { inspectDeviceProgram } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { compileTensorDeviceProgram } from '../index.mjs';
import { createDeviceItemProfile } from '../testing.mjs';

function mixed() { return TensorPlan.create(TensorProgram.define((graph) => { const items = graph.input('items', { dtype: 'f32', capacityShape: [2, 2, 3], access: 'read' }), weights = graph.input('weights', { dtype: 'f32', capacityShape: [1, 3, 2], access: 'read' }); const product = graph.matmul(items, weights), selected = graph.gather(product, 2, [1, 0, 1]); return { selected: graph.unary('tanh', graph.unary('erf', selected)), sum: graph.reduce('sum', product, { axes: [2] }) }; })); }
test('block32 changes only participation and keeps exact scalar ABI, material layout and workspace', () => {
  const plan = mixed(), options = { itemCapacity: 2, itemInputs: ['items'] };
  const scalar = createDeviceItemProfile(plan, options), explicit = createDeviceItemProfile(plan, { ...options, participation: 'scalar' }), cooperative = createDeviceItemProfile(plan, { ...options, participation: 'block32' });
  assert.equal(explicit.compatibilityIdentity, scalar.compatibilityIdentity); assert.equal(explicit.lowering.source, scalar.lowering.source);
  assert.notEqual(cooperative.compatibilityIdentity, scalar.compatibilityIdentity); assert.equal(cooperative.totalWorkspaceBytes, scalar.totalWorkspaceBytes);
  assert.deepEqual(cooperative.canonical.workspace, scalar.canonical.workspace); assert.deepEqual(cooperative.parameters, scalar.parameters);
  assert.match(cooperative.contract, /SPEC-0009-block32-v1$/);
  assert.match(cooperative.lowering.source, /let participant = gpu\.cast\.u64\(gpu\.thread\.x\(\)\)/);
  assert.match(cooperative.lowering.source, /index = participant; index < .*; index \+= gpu\.u64\(32n\)/);
  assert.match(cooperative.lowering.source, /for \(let k = gpu\.u64\(0n\); k < .*; k \+= gpu\.u64\(1n\)/);
  assert.match(cooperative.lowering.source, /for \(let reductionIndex = gpu\.u64\(0n\)/);
  const materialStages = plan.program.nodes.filter((node) => node.materialization === 'materialize' && node.outputSpec.logicalElementCount > 0).length;
  assert.equal(cooperative.lowering.source.split('gpu.barrier.block();').length - 1, materialStages + plan.program.outputs.length);
  const inspected = inspectDeviceProgram({ source: cooperative.lowering.source + '\nfunction consume(items, weights, out0, out1, workspace) { let status = tensorRunItem(gpu.block.x(), items, weights, out0, out1, workspace); }', functions: [cooperative.lowering.function, { name: 'consume', kind: 'kernel', parameters: cooperative.lowering.function.parameters.slice(1).map((p, i) => ({ ...p, name: ['items', 'weights', 'out0', 'out1', 'workspace'][i] })), returns: 'void' }] });
  assert(inspected.inspection.publicHelperUsage.find((entry) => entry.function === 'tensorRunItem').helpers.includes('gpu.barrier.block'));
});
test('block32 rejects unknown participation before compilation', () => { assert.throws(() => createDeviceItemProfile(mixed(), { itemCapacity: 2, itemInputs: ['items'], participation: 'warp' }), { code: 'TENSOR_DEVICE_PARTICIPATION_UNSUPPORTED' }); });
test('public callable declares and validates complete uniform 32-thread block admission', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true }), session = await TensorSession.open(runtime);
  try {
    const callable = await compileTensorDeviceProgram(session, mixed(), { itemCapacity: 2, itemInputs: ['items'], participation: 'block32' });
    assert.equal(callable.participation.requiredThreads, 32);
    assert.deepEqual(callable.participation.block, { x: 32, y: 1, z: 1 });
    assert.equal(callable.requireParticipation({ block: { x: 32, y: 1, z: 1 }, uniformItemIndex: true, uniformCall: true }).requiredThreads, 32);
    for (const request of [{ block: { x: 16, y: 1, z: 1 }, uniformItemIndex: true, uniformCall: true }, { block: { x: 32, y: 1, z: 1 }, uniformItemIndex: false, uniformCall: true }, { block: { x: 32, y: 1, z: 1 }, uniformItemIndex: true, uniformCall: false }]) assert.throws(() => callable.requireParticipation(request), { code: 'TENSOR_DEVICE_PARTICIPATION_MISMATCH' });
    assert.throws(() => callable.requireParticipation({ block: Object.assign([], { x: 32, y: 1, z: 1 }), uniformItemIndex: true, uniformCall: true }), { code: 'TENSOR_DEVICE_PARTICIPATION_INVALID' });
  } finally { await session.close(); await runtime.close(); }
});
