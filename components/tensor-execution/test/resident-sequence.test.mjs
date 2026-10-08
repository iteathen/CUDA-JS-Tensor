import assert from 'node:assert/strict';
import test from 'node:test';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';
import { TensorProgram, TensorPlan } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { resolveTensorPlan } from '../index.mjs';
import { lowerSimtPlan, createCudaJsTensorBackend, createBackendProfileRequest } from '../testing.mjs';

function chain(length = 80) {
  return TensorProgram.define((graph) => {
    let value = graph.input('items', { dtype: 'f32', capacityShape: [2, 2], access: 'read-write' });
    for (let i = 0; i < length; i++) { value = graph.unary('neg', value); value = graph.reshape(value, [2, 2]); }
    return value;
  });
}
function independentPairs() { return TensorProgram.define((graph) => Object.fromEntries(Array.from({ length: 35 }, (_, i) => [`out${i}`, graph.binary('add', graph.input(`left${i}`, { dtype: 'f32', capacityShape: [4], access: 'read-write' }), graph.input(`right${i}`, { dtype: 'f32', capacityShape: [4], access: 'read-write' }))]))); }
test('explicit resident sequence preserves whole material layout above 64 bindings and 32 kernels', () => {
  const plan = TensorPlan.create(chain());
  assert.throws(() => lowerSimtPlan(plan), { code: 'TENSOR_SIMT_BINDING_LIMIT' });
  const lowered = lowerSimtPlan(plan, { execution: 'resident-sequence' });
  assert.equal(lowered.bindings.length, 81); assert.equal(lowered.kernels.length, 80); assert.equal(lowered.materialBytes, 1280);
  assert.deepEqual(lowered.preparedSequence.map((chunk) => chunk.kernelIds.length), [32, 32, 16]);
  for (const chunk of lowered.preparedSequence) { assert(chunk.bindingNames.length <= 64); assert(chunk.edgeCount <= 64); }
  assert.equal(lowered.outputs[0].baseValueId, 'node:158');
});
test('resident sequence cuts at binding pressure before reaching the node limit', () => { const lowered = lowerSimtPlan(TensorPlan.create(independentPairs()), { execution: 'resident-sequence' }); assert.deepEqual(lowered.preparedSequence.map((chunk) => chunk.kernelIds.length), [21, 14]); assert.deepEqual(lowered.preparedSequence.map((chunk) => chunk.bindingNames.length), [63, 42]); });
test('resident sequence public resolver rejects unknown execution profiles and accelerated policy', async () => {
  const runtime = await openCudaRuntimeForTesting(); const session = await TensorSession.open(runtime);
  try {
    await assert.rejects(resolveTensorPlan(session, chain(), { execution: 'unknown' }), { code: 'TENSOR_EXECUTION_PROFILE_UNSUPPORTED' });
    await assert.rejects(resolveTensorPlan(session, chain(), { execution: 'resident-sequence', backend: 'prefer-cublaslt' }), { code: 'TENSOR_RESIDENT_SEQUENCE_BACKEND_UNSUPPORTED' });
  } finally { await session.close(); await runtime.close(); }
});
test('public resident result keeps one whole-plan lifecycle and an explicit physical contract', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true }); const session = await TensorSession.open(runtime);
  let resolved, input, result;
  try {
    const program = chain(), plan = TensorPlan.create(program);
    resolved = await resolveTensorPlan(session, plan, { execution: 'resident-sequence' });
    assert.equal(resolved.contract, 'SPEC-0006-resolved-dense-plan-v1+SPEC-0007-exact-elementwise-fusion-v1+SPEC-0005-resident-sequence-v1');
    assert.equal(resolved.plan.compatibilityIdentity, plan.compatibilityIdentity);
    assert.equal(resolved.canonical.backendDescriptor.chunkCount, 3);
    input = await session.allocate(program.inputs[0].spec); result = await resolved.run(input);
    assert.equal(result.execution.chunkCount, 3); assert.equal(result.execution.engineActiveInferenceProfileClaimed, false);
    assert.equal((await result.close()).graceful, true); assert.equal(input.state, 'open');
    assert.equal((await resolved.close()).graceful, true); await input.close();
    assert.equal((await session.status()).accounting.reservedBytes, 0);
  } finally { await session.close(); await runtime.close(); }
});
test('explicit single-dag selection keeps the original resolved physical identity', async () => {
  const runtime = await openCudaRuntimeForTesting({ compiler: true }); const session = await TensorSession.open(runtime);
  try { const program = chain(2), first = await resolveTensorPlan(session, program), second = await resolveTensorPlan(session, program, { execution: 'single-dag' }); assert.equal(first.compatibilityIdentity, second.compatibilityIdentity); await first.close(); await second.close(); }
  finally { await session.close(); await runtime.close(); }
});

function provider({ failPrepare = -1, failRun = -1, failOperationClose = -1 } = {}) {
  const events = [], resources = [], submitted = [];
  const capability = (kind, extra = {}) => { const value = { kind, state: 'open', ...extra, async close() { events.push(`close:${kind}`); this.state = 'closed'; return { graceful: true }; } }; resources.push(value); return value; };
  const runtime = {
    async loadModule() { return capability('module', { sha256: 'module-identity', format: 'cubin', async getFunction() { return capability('function'); } }); },
    async prepareOperationDag(nodes) {
      const index = submitted.length; submitted.push(nodes); if (index === failPrepare) throw Error('prepare failure');
      const bindingNames = [...new Set(nodes.flatMap((node) => node.arguments.map((arg) => arg.binding)))];
      assert(nodes.length <= 32); assert(bindingNames.length <= 64); assert(nodes[0].after === undefined);
      return capability('dag', { realization: 'prepared', contract: 'public-prepared-dag', sha256: `dag${index}`, nodeCount: nodes.length, edgeCount: nodes.length - 1, async submit({ bindings }) {
        events.push(`submit:${index}`); assert.deepEqual(Object.keys(bindings).sort(), bindingNames.sort());
        return { async wait() { events.push(`wait:${index}`); if (index === failRun) throw Error('execution failure'); return { status: 'completed', operationSequence: index }; }, async close() { events.push(`operation-close:${index}`); if (index === failOperationClose) throw Error('operation close failure'); } };
      } });
    },
  };
  const compileDeviceProgram = async (_runtime, { functions }) => ({ compiler: { artifact: { format: 'cubin', bytes: Uint8Array.of(1), sha256: 'artifact', architecture: 'compute_75' }, headerProfile: 'cuda-numeric' }, deviceProgram: { contract: 'public-device-js', sha256: 'program', kernels: functions.map((fn) => ({ name: fn.name, functionName: fn.name, parameters: fn.parameters })) } });
  return { runtime, compileDeviceProgram, events, resources };
}
async function backend(p) { const plan = TensorPlan.create(chain()); const options = { execution: 'resident-sequence', backend: 'simt', maxWorkspaceBytes: 67108864 }; const lowering = lowerSimtPlan(plan, options); return { lowering, adapter: await createCudaJsTensorBackend(p.runtime, lowering, createBackendProfileRequest(plan, lowering, options), options, { compileDeviceProgram: p.compileDeviceProgram }) }; }
test('sequence awaits and closes each operation before the next chunk while borrowing whole bindings', async () => { const p = provider(); const { lowering, adapter } = await backend(p); const bindings = Object.fromEntries(lowering.bindings.map((b) => [b.name, { id: b.name }])); await adapter.execute(bindings); assert.deepEqual(p.events, ['submit:0', 'wait:0', 'operation-close:0', 'submit:1', 'wait:1', 'operation-close:1', 'submit:2', 'wait:2', 'operation-close:2']); assert.equal((await adapter.close()).graceful, true); assert(p.resources.every((r) => r.state === 'closed')); });
test('failed sequence execution closes its operation and never submits later chunks', async () => { const p = provider({ failRun: 1 }); const { lowering, adapter } = await backend(p); try { await assert.rejects(adapter.execute(Object.fromEntries(lowering.bindings.map((b) => [b.name, {}]))), /execution failure/); assert(!p.events.includes('submit:2')); assert(p.events.includes('operation-close:1')); } finally { await adapter.close(); } assert(p.resources.every((r) => r.state === 'closed')); });
test('partial sequence preparation rolls back every acquired DAG, function, and module', async () => { const p = provider({ failPrepare: 1 }); await assert.rejects(backend(p), /prepare failure/); assert(p.resources.every((r) => r.state === 'closed')); });
test('failed operation cleanup stays cleanup-unproved and stops the sequence', async () => { const p = provider({ failOperationClose: 0 }); const { lowering, adapter } = await backend(p); try { await assert.rejects(adapter.execute(Object.fromEntries(lowering.bindings.map((b) => [b.name, {}]))), { code: 'TENSOR_EXECUTION_OPERATION_CLEANUP_UNPROVED' }); assert(!p.events.includes('submit:1')); } finally { await adapter.close(); } });
