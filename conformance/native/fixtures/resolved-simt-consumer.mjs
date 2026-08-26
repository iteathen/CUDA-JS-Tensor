import assert from 'node:assert/strict';

import { openCudaRuntime } from 'cuda-js';
import { CUDA_JS_TENSOR_COMPATIBILITY, resolveTensorPlan, TensorProgram, TensorSession } from 'cuda-js-tensor';

function numericBytes(dtype, values) {
  const widths = { i32: 4, f32: 4, f64: 8, f16: 2, bf16: 2 };
  const width = widths[dtype];
  const bytes = new Uint8Array(values.length * width);
  const view = new DataView(bytes.buffer);
  for (let index = 0; index < values.length; index += 1) {
    const offset = index * width;
    if (dtype === 'i32') view.setInt32(offset, values[index], true);
    else if (dtype === 'f32') view.setFloat32(offset, values[index], true);
    else if (dtype === 'f64') view.setFloat64(offset, values[index], true);
    else view.setUint16(offset, values[index], true);
  }
  return bytes;
}

function numericValues(dtype, bytes) {
  const widths = { i32: 4, f32: 4, f64: 8, f16: 2, bf16: 2 };
  const width = widths[dtype];
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const values = [];
  for (let offset = 0; offset < bytes.byteLength; offset += width) {
    if (dtype === 'i32') values.push(view.getInt32(offset, true));
    else if (dtype === 'f32') values.push(view.getFloat32(offset, true));
    else if (dtype === 'f64') values.push(view.getFloat64(offset, true));
    else values.push(view.getUint16(offset, true));
  }
  return values;
}

async function execute(session, program, inputBytes, { replays = 1 } = {}) {
  const resolved = await resolveTensorPlan(session, program);
  const inputs = {};
  const observations = [];
  try {
    for (const input of program.inputs) {
      const tensor = await session.allocate(input.spec);
      inputs[input.name] = tensor;
      await tensor.write(inputBytes[input.name]);
    }
    for (let replay = 0; replay < replays; replay += 1) {
      const result = await resolved.run(inputs);
      try {
        const outputs = {};
        for (const output of program.outputs) outputs[output.name] = (await result.get(output.name).read()).bytes;
        observations.push({ outputs, execution: result.execution });
      } finally {
        assert.equal((await result.close()).graceful, true);
      }
    }
    return {
      kernelCount: resolved.kernelCount,
      bindingCount: resolved.bindingCount,
      workspaceBytes: resolved.workspaceBytes,
      prepared: resolved.canonical.backendDescriptor.prepared ?? null,
      compiler: resolved.canonical.backendDescriptor.compiler ?? null,
      observations,
    };
  } finally {
    assert.equal((await resolved.close()).graceful, true);
    for (const tensor of Object.values(inputs).reverse()) if (tensor.state === 'open') await tensor.close();
  }
}

assert.equal(process.version, 'v26.7.0');
assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.package.version, '0.1.0-alpha.3');
assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version, '0.1.0-alpha.14');

const runtime = await openCudaRuntime({
  compiler: true,
  driver: {
    memory: { maxDeviceBytes: 32 * 1024 * 1024, maxAllocationBytes: 8 * 1024 * 1024, maxTransferBytes: 8 * 1024 * 1024 },
    execution: { maxModuleBytes: 4 * 1024 * 1024, maxArguments: 32, maxCompletionMilliseconds: 30_000 },
  },
});
const session = await TensorSession.open(runtime);
const runtimeDescription = await runtime.describe();
let runtimeTerminal;
let sessionTerminal;
const summary = {};
try {
  const composite = TensorProgram.define((graph) => {
    const source = graph.input('source', { dtype: 'f32', capacityShape: [2, 3], access: 'read-write' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read-write' });
    const copied = graph.copy(source);
    const widened = graph.cast(copied, 'f64');
    const reshaped = graph.reshape(widened, [3, 2]);
    const permuted = graph.permute(reshaped, [1, 0]);
    const sliced = graph.slice(permuted, [null, { start: 0, length: 1 }]);
    const broadcast = graph.broadcast(sliced, [2, 3]);
    const contiguous = graph.contiguous(broadcast);
    const rooted = graph.unary('sqrt', contiguous);
    const selected = graph.binary('maximum', rooted, rooted);
    const reduced = graph.reduce('sum', selected, { axes: [1], order: 'fixed-tree-v1' });
    const matmul = graph.matmul(source, right);
    const filled = graph.fill({ dtype: 'f16', capacityShape: [2, 3] }, 1.5);
    return { reduced, matmul, filled };
  });
  const compositeRun = await execute(session, composite, {
    source: numericBytes('f32', [1, 4, 9, 16, 25, 36]),
    right: numericBytes('f32', [1, 2, 3, 4, 5, 6]),
  }, { replays: 2 });
  for (const replay of compositeRun.observations) {
    assert.deepEqual(numericValues('f64', replay.outputs.reduced), [3, 6]);
    assert.deepEqual(numericValues('f32', replay.outputs.matmul), [58, 72, 271, 348]);
    assert.deepEqual(numericValues('f16', replay.outputs.filled), [0x3e00, 0x3e00, 0x3e00, 0x3e00, 0x3e00, 0x3e00]);
  }
  const sequences = compositeRun.observations.map((entry) => entry.execution.operationSequence);
  assert(sequences.every(Number.isSafeInteger));
  assert(sequences[1] > sequences[0]);
  assert.equal(compositeRun.prepared.nodeCount, compositeRun.kernelCount);
  assert.equal(compositeRun.compiler.architecture, 'compute_75');
  assert.equal(compositeRun.compiler.headerProfile, 'cuda-numeric');
  summary.composite = { kernels: compositeRun.kernelCount, bindings: compositeRun.bindingCount, workspaceBytes: compositeRun.workspaceBytes, replays: compositeRun.observations.length, compiler: { architecture: compositeRun.compiler.architecture, headerProfile: compositeRun.compiler.headerProfile } };

  const casts = TensorProgram.define((graph) => graph.cast(graph.input('values', { dtype: 'f64', capacityShape: [5], access: 'read-write' }), 'i32'));
  const castRun = await execute(session, casts, { values: numericBytes('f64', [Number.NaN, Infinity, -Infinity, 3.9, -3.9]) });
  assert.deepEqual(numericValues('i32', castRun.observations[0].outputs.output), [0, 2_147_483_647, -2_147_483_648, 3, -3]);
  summary.cast = numericValues('i32', castRun.observations[0].outputs.output);

  const half = TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f16', capacityShape: [4], access: 'read-write' });
    const right = graph.input('right', { dtype: 'f16', capacityShape: [4], access: 'read-write' });
    return graph.binary('add', left, right);
  });
  const halfRun = await execute(session, half, {
    left: numericBytes('f16', [0x3c00, 0x4000, 0x8000, 0x7bff]),
    right: numericBytes('f16', [0x3800, 0x4200, 0x0000, 0x3c00]),
  });
  assert.deepEqual(numericValues('f16', halfRun.observations[0].outputs.output), [0x3e00, 0x4500, 0x0000, 0x7bff]);
  summary.f16 = numericValues('f16', halfRun.observations[0].outputs.output);

  const bfloat = TensorProgram.define((graph) => graph.unary('neg', graph.input('values', { dtype: 'bf16', capacityShape: [4], access: 'read-write' })));
  const bfloatRun = await execute(session, bfloat, { values: numericBytes('bf16', [0x3f80, 0xc000, 0x3f00, 0x8000]) });
  assert.deepEqual(numericValues('bf16', bfloatRun.observations[0].outputs.output), [0xbf80, 0x4000, 0xbf00, 0x0000]);
  summary.bf16 = numericValues('bf16', bfloatRun.observations[0].outputs.output);

  const reduction = TensorProgram.define((graph) => graph.reduce('sum', graph.input('values', { dtype: 'f32', capacityShape: [4], access: 'read-write' }), { order: 'fixed-tree-v1' }));
  const reductionRun = await execute(session, reduction, { values: numericBytes('f32', [1e20, 1, -1e20, 3]) });
  const reduced = numericValues('f32', reductionRun.observations[0].outputs.output);
  assert.deepEqual(reduced, [0]);
  assert(reductionRun.workspaceBytes > 0);
  summary.fixedTree = { output: reduced, workspaceBytes: reductionRun.workspaceBytes };
} finally {
  sessionTerminal = await session.close();
  runtimeTerminal = await runtime.close();
}

assert.equal(sessionTerminal.graceful, true);
assert.equal(sessionTerminal.runtimeClosed, false);
assert.deepEqual(sessionTerminal.accounting, { liveTensors: 0, reservedBytes: 0, resolvedPlans: 0, pendingResolutions: 0 });
assert.equal(runtimeTerminal.graceful, true);
assert.equal(runtimeTerminal.driver.resourceCounts.live, 0);
assert.equal(runtimeTerminal.driver.resourceCounts.orphaned, 0);
console.log(JSON.stringify({ consumer: 'installed-native-resolved-simt', node: process.version, platform: process.platform, device: runtimeDescription.device?.architecture ?? null, summary, sessionGraceful: sessionTerminal.graceful, runtimeGraceful: runtimeTerminal.graceful }));
