import assert from 'node:assert/strict';
import { CUDA_JS_COMPATIBILITY, openCudaRuntime } from 'cuda-js';
import { CUDA_JS_TENSOR_COMPATIBILITY, resolveTensorPlan, TensorProgram, TensorSession } from 'cuda-js-tensor';

assert.equal(process.version, 'v26.11.1');
assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.package.version, '0.1.0-alpha.8');
assert.equal(CUDA_JS_COMPATIBILITY.package.version, '0.1.0-alpha.21');
const data = Buffer.alloc(16); [1, 2, -3, 4].forEach((value, index) => data.writeFloatLE(value, index * 4));
const runtime = await openCudaRuntime({ compiler: true, driver: { memory: { maxDeviceBytes: 32 * 1024 * 1024, maxAllocationBytes: 8 * 1024 * 1024, maxTransferBytes: 8 * 1024 * 1024 }, execution: { maxModuleBytes: 4 * 1024 * 1024, maxArguments: 32, maxCompletionMilliseconds: 30000 } } });
const session = await TensorSession.open(runtime); const cases = []; let sessionTerminal, runtimeTerminal;
try {
  const chain = TensorProgram.define((graph) => { let value = graph.input('items', { dtype: 'f32', capacityShape: [2, 2], access: 'read-write' }); for (let i = 0; i < 80; i++) { value = graph.unary('neg', value); value = graph.reshape(value, [2, 2]); } return value; });
  const pairs = TensorProgram.define((graph) => Object.fromEntries(Array.from({ length: 35 }, (_, i) => [`out${i}`, graph.binary('add', graph.input(`left${i}`, { dtype: 'f32', capacityShape: [4], access: 'read-write' }), graph.input(`right${i}`, { dtype: 'f32', capacityShape: [4], access: 'read-write' }))])));
  for (const [name, program, expected] of [['view-chain', chain, [1, 2, -3, 4]], ['binding-cut', pairs, [2, 4, -6, 8]]]) {
    const resolved = await resolveTensorPlan(session, program, { execution: 'resident-sequence' }); const inputs = {};
    try {
      for (const input of program.inputs) { inputs[input.name] = await session.allocate(input.spec); await inputs[input.name].write(data); }
      for (let replay = 0; replay < 2; replay++) {
        const result = await resolved.run(inputs);
        try {
          for (const tensor of Object.values(result.outputs)) { const bytes = Buffer.from((await tensor.read()).bytes); assert.deepEqual(Array.from({ length: 4 }, (_, i) => bytes.readFloatLE(i * 4)), expected); }
          assert.equal(result.execution.engineActiveInferenceProfileClaimed, false);
        } finally { assert.equal((await result.close()).graceful, true); }
      }
      const descriptor = resolved.canonical.backendDescriptor;
      for (const chunk of descriptor.preparedSequence) { assert(chunk.prepared.nodeCount <= 32); assert(chunk.prepared.edgeCount <= 64); assert(chunk.bindingNames.length <= 64); }
      cases.push({ name, contract: resolved.contract, planIdentity: resolved.plan.compatibilityIdentity, resolvedIdentity: resolved.compatibilityIdentity, materialBytes: resolved.canonical.materialBytes, bindingCount: resolved.bindingCount, kernelCount: resolved.kernelCount, chunks: descriptor.preparedSequence, expected, replays: 2 });
    } finally { assert.equal((await resolved.close()).graceful, true); for (const tensor of Object.values(inputs).reverse()) assert.equal((await tensor.close()).state, 'closed'); }
  }
  const tiny = await TensorSession.open({ runtime, limits: { maxSessionBytes: 16, maxTensorBytes: 16 } });
  try { await assert.rejects(resolveTensorPlan(tiny, chain, { execution: 'resident-sequence' }), { code: 'TENSOR_RESOLVE_SESSION_BYTE_LIMIT' }); }
  finally { assert.equal((await tiny.close()).graceful, true); }
} finally { sessionTerminal = await session.close(); runtimeTerminal = await runtime.close(); }
assert.equal(sessionTerminal.graceful, true); assert.equal(runtimeTerminal.graceful, true);
console.log(JSON.stringify({ consumer: 'installed-resident-sequence-native', node: process.version, tensorCompatibility: CUDA_JS_TENSOR_COMPATIBILITY, cases, sessionTerminal, runtimeTerminal, hostPlanQualificationOnly: true, engineActiveInferenceProfileClaimed: false }));
