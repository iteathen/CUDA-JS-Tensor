import assert from 'node:assert/strict';
import { compileDeviceProgram, openCudaRuntime } from 'cuda-js';
import { compileTensorDeviceProgram, CUDA_JS_TENSOR_COMPATIBILITY, TensorProgram, TensorSession } from 'cuda-js-tensor';

assert.equal(process.version, 'v26.11.1');
assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.package.version, '0.1.0-alpha.9');
const f32 = (values) => { const b = Buffer.alloc(values.length * 4); values.forEach((v, i) => b.writeFloatLE(v, i * 4)); return b; };
const runtime = await openCudaRuntime({ compiler: true, driver: { memory: { maxDeviceBytes: 128 * 1024 * 1024, maxAllocationBytes: 64 * 1024 * 1024, maxTransferBytes: 64 * 1024 * 1024 }, execution: { maxModuleBytes: 4 * 1024 * 1024, maxArguments: 32, maxCompletionMilliseconds: 30000 } } });
const session = await TensorSession.open(runtime); const observations = [];

async function exercise(program, inputData, expected, { occupancy = 2, macs = 0 } = {}) {
  const callable = await compileTensorDeviceProgram(session, program, { itemCapacity: 2, itemInputs: ['items'], participation: 'block32' });
  callable.requireParticipation({ block: { x: 32, y: 1, z: 1 }, uniformItemIndex: true, uniformCall: true });
  const pointers = callable.parameters.slice(1), parameters = [...pointers.map((p) => ({ name: p.parameterName, type: p.type })), { name: 'statuses', type: 'ptr<u32>' }, { name: 'occupancy', type: 'u32' }];
  const source = `function evaluate(${parameters.map((p) => p.name).join(', ')}) { let group = gpu.block.x(); let item = group; if (group >= occupancy) { item = gpu.u32(2); } let status = evaluateItem(item, ${pointers.map((p) => p.parameterName).join(', ')}); if (gpu.thread.x() === gpu.u32(0)) { statuses[group] = status; } }`;
  const compiled = await compileDeviceProgram(runtime, { source, functions: [{ name: 'evaluate', kind: 'kernel', parameters, returns: 'void' }], imports: [callable.importAs('evaluateItem')] });
  const module = await runtime.loadModule({ format: compiled.linker.artifact.format, bytes: compiled.linker.artifact.bytes });
  const kernel = compiled.deviceProgram.kernels.find((entry) => entry.name === 'evaluate');
  const fn = await module.getFunction({ name: kernel.functionName, parameters: kernel.parameters });
  const memories = [], views = [], payloads = []; let operation;
  try {
    for (const p of pointers) {
      const bytes = Buffer.alloc(p.byteLength + 32, 0xa5); if (p.role === 'input') inputData[p.name].copy(bytes, 16);
      const memory = await runtime.allocateDevice({ byteLength: bytes.length }); memories.push(memory); payloads.push(bytes);
      await memory.write(Uint8Array.from(bytes)); views.push(await memory.view({ dtype: p.dtype, byteOffset: 16, elementCount: p.elementCount, access: p.access }));
    }
    const statusMemory = await runtime.allocateDevice({ byteLength: 48 }); memories.push(statusMemory); await statusMemory.write(Uint8Array.from(Buffer.alloc(48, 0xa5)));
    views.push(await statusMemory.view({ dtype: 'u32', byteOffset: 16, elementCount: 4, access: 'write' }));
    const start = performance.now();
    operation = await fn.submit({ grid: { x: 4, y: 1, z: 1 }, block: { x: 32, y: 1, z: 1 }, arguments: [...views, occupancy], accesses: [...pointers.map((p, argumentIndex) => ({ argumentIndex, byteOffset: 0, byteLength: p.byteLength, mode: p.access })), { argumentIndex: pointers.length, byteOffset: 0, byteLength: 16, mode: 'write' }] });
    assert.equal((await operation.wait()).status, 'completed');
    const elapsedMilliseconds = performance.now() - start;
    await operation.close(); operation = null;
    for (let i = 0; i < pointers.length; i++) {
      const p = pointers[i], bytes = Buffer.from((await memories[i].read({ byteLength: p.byteLength + 32 })).bytes);
      assert(bytes.subarray(0, 16).equals(Buffer.alloc(16, 0xa5))); assert(bytes.subarray(16 + p.byteLength).equals(Buffer.alloc(16, 0xa5)));
      if (p.role === 'input') assert(bytes.equals(payloads[i]));
      if (p.role === 'output') {
        const values = expected[p.name].slice(0, occupancy * p.perItemElements);
        for (let k = 0; k < values.length; k++) assert(Math.abs(bytes.readFloatLE(16 + k * 4) - values[k]) <= 0.000002);
      }
      if (occupancy === 1 && p.itemVarying && p.role !== 'input') assert(bytes.subarray(16 + p.byteLength / 2, 16 + p.byteLength).equals(Buffer.alloc(p.byteLength / 2, 0xa5)));
    }
    const statusBytes = Buffer.from((await statusMemory.read({ byteLength: 48 })).bytes);
    assert(statusBytes.subarray(0, 16).equals(Buffer.alloc(16, 0xa5))); assert(statusBytes.subarray(32).equals(Buffer.alloc(16, 0xa5)));
    const statuses = Array.from({ length: 4 }, (_, i) => statusBytes.readUInt32LE(16 + i * 4)); assert.deepEqual(statuses, occupancy === 2 ? [0, 0, 1, 1] : [0, 1, 1, 1]);
    return { programIdentity: program.compatibilityIdentity, callableIdentity: callable.compatibilityIdentity, contract: callable.contract, participation: callable.participation, compiler: callable.canonical.compiler, linkedProgram: compiled.deviceProgram.sha256, linkedArtifact: compiled.linker.artifact.sha256, workspaceBytes: callable.totalWorkspaceBytes, occupancy, statuses, guardedNoWrite: true, scalarMacCount: macs, elapsedMilliseconds, latencyMeaning: 'single native submission through completed public wait, excluding compile and transfers' };
  } finally {
    if (operation) await operation.close(); await fn.close(); await module.close();
    for (const view of views.reverse()) await view.close(); for (const memory of memories.reverse()) await memory.close();
  }
}

let sessionTerminal, runtimeTerminal;
try {
  const mixed = TensorProgram.define((g) => { const items = g.input('items', { dtype: 'f32', capacityShape: [2, 2, 3], access: 'read' }), weights = g.input('weights', { dtype: 'f32', capacityShape: [1, 3, 2], access: 'read' }); const product = g.matmul(items, weights); return { selected: g.unary('tanh', g.unary('erf', g.gather(product, 2, [1, 0, 1]))), sum: g.reduce('sum', product, { axes: [2] }) }; });
  const seeds = { items: f32([0, 1, 99, 1, 0, 88, -1, 0, 77, 0, -1, 66]), weights: f32([1, 0, 0, 1, 0, 0]) };
  const expectations = { selected: [0.68723746, 0, 0.68723746, 0, 0.68723746, 0, 0, -0.68723746, 0, -0.68723746, 0, -0.68723746], sum: [1, 1, -1, -1] };
  for (const occupancy of [2, 1]) observations.push({ case: 'mixed', ...await exercise(mixed, seeds, expectations, { occupancy }) });
  // Each stage has 2*64*256*256 exact serial MACs, distributed only across independent outputs.
  for (const stages of [1, 8, 16, 32]) {
    const program = TensorProgram.define((g) => { let value = g.input('items', { dtype: 'f32', capacityShape: [2, 64, 256], access: 'read' }); const weights = g.input('weights', { dtype: 'f32', capacityShape: [1, 256, 256], access: 'read' }); for (let i = 0; i < stages; i++) value = g.matmul(value, weights); return value; });
    const observation = await exercise(program, { items: f32(Array(2 * 64 * 256).fill(1)), weights: f32(Array(256 * 256).fill(1 / 256)) }, { output: Array(2 * 64 * 256).fill(1) }, { macs: stages * 2 * 64 * 256 * 256 });
    observations.push({ case: 'bounded-matmul-ladder', stages, ...observation });
    console.log(JSON.stringify({ stages, scalarMacCount: observation.scalarMacCount, elapsedMilliseconds: observation.elapsedMilliseconds }));
    assert(observation.elapsedMilliseconds < 1500, 'Workload ladder exceeded conservative 1.5-second host completion bound; do not launch a larger profile');
  }
} finally { sessionTerminal = await session.close(); runtimeTerminal = await runtime.close(); }
assert.equal(sessionTerminal.graceful, true); assert.equal(runtimeTerminal.graceful, true);
console.log(JSON.stringify({ consumer: 'installed-block32-native', node: process.version, tensorCompatibility: CUDA_JS_TENSOR_COMPATIBILITY, observations, sessionTerminal, runtimeTerminal, performanceClaim: 'measured exact workloads only; no inferred model latency or throughput' }));
