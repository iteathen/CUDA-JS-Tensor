import assert from 'node:assert/strict';

import { compileDeviceProgram, openCudaRuntime } from 'cuda-js';
import { compileTensorDeviceProgram, CUDA_JS_TENSOR_COMPATIBILITY, resolveTensorPlan, TensorProgram, TensorSession } from 'cuda-js-tensor';

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

async function execute(session, program, inputBytes, { replays = 1, resolveOptions = undefined } = {}) {
  const resolved = await resolveTensorPlan(session, program, resolveOptions);
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
      backend: resolved.backend,
      backendPolicy: resolved.backendPolicy,
      fusionPolicy: resolved.fusionPolicy,
      fusionRegionCount: resolved.fusionRegionCount,
      fusedNodeCount: resolved.fusedNodeCount,
      kernelCount: resolved.kernelCount,
      cublasLtNodeCount: resolved.cublasLtNodeCount,
      bindingCount: resolved.bindingCount,
      workspaceBytes: resolved.workspaceBytes,
      backendProfile: resolved.canonical.backendProfile,
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
assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.package.version, '0.1.0-alpha.6');
assert.equal(CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version, '0.1.0-alpha.16');

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

  const accelerated = TensorProgram.define((graph) => {
    const left = graph.unary('abs', graph.input('left', { dtype: 'f32', capacityShape: [2, 3], access: 'read-write' }));
    const right = graph.input('right', { dtype: 'f32', capacityShape: [3, 2], access: 'read-write' });
    return graph.unary('neg', graph.matmul(left, right));
  });
  const acceleratedRun = await execute(session, accelerated, {
    left: numericBytes('f32', [1, 4, 9, 16, 25, 36]),
    right: numericBytes('f32', [1, 2, 3, 4, 5, 6]),
  }, { replays: 2, resolveOptions: { backend: 'cublaslt', maxWorkspaceBytes: 1024 * 1024 } });
  for (const replay of acceleratedRun.observations) assert.deepEqual(numericValues('f32', replay.outputs.output), [-58, -72, -271, -348]);
  assert.equal(acceleratedRun.backendPolicy, 'cublaslt');
  assert.equal(acceleratedRun.backend, 'mixed');
  assert.equal(acceleratedRun.kernelCount, 2);
  assert.equal(acceleratedRun.cublasLtNodeCount, 1);
  assert.equal(acceleratedRun.prepared.contract, 'SPEC-0020-prepared-kernel-dag-v1+SPEC-0031-prepared-cublaslt-f32-matmul-node-v1');
  assert.equal(acceleratedRun.prepared.nodeCount, acceleratedRun.kernelCount + acceleratedRun.cublasLtNodeCount);
  assert.equal(acceleratedRun.backendProfile.provider.profile, 'cublaslt-f32-row-major-matmul-v1');
  summary.accelerated = {
    backend: acceleratedRun.backend,
    kernels: acceleratedRun.kernelCount,
    cublasLtNodes: acceleratedRun.cublasLtNodeCount,
    workspaceBytes: acceleratedRun.workspaceBytes,
    replays: acceleratedRun.observations.length,
    provider: acceleratedRun.backendProfile.provider,
  };

  const transposed = TensorProgram.define((graph) => graph.matmul(
    graph.input('left', { dtype: 'f32', capacityShape: [3, 2], access: 'read-write' }),
    graph.input('right', { dtype: 'f32', capacityShape: [2, 3], access: 'read-write' }),
    { transposeA: true, transposeB: true },
  ));
  const transposedRun = await execute(session, transposed, {
    left: numericBytes('f32', [1, 4, 2, 5, 3, 6]),
    right: numericBytes('f32', [7, 9, 11, 8, 10, 12]),
  }, { resolveOptions: { backend: 'cublaslt', maxWorkspaceBytes: 1 } });
  assert.deepEqual(numericValues('f32', transposedRun.observations[0].outputs.output), [58, 64, 139, 154]);
  assert.equal(transposedRun.backend, 'cublaslt');
  assert.equal(transposedRun.kernelCount, 0);
  assert.equal(transposedRun.cublasLtNodeCount, 1);
  assert.deepEqual(transposedRun.backendProfile.matmuls[0].plan.requirements, { a: 6, b: 6, c: 4, d: 4 });
  summary.transposedAccelerated = {
    backend: transposedRun.backend,
    cublasLtNodes: transposedRun.cublasLtNodeCount,
    output: numericValues('f32', transposedRun.observations[0].outputs.output),
  };

  const casts = TensorProgram.define((graph) => graph.cast(graph.input('values', { dtype: 'f64', capacityShape: [5], access: 'read-write' }), 'i32'));
  const castRun = await execute(session, casts, { values: numericBytes('f64', [Number.NaN, Infinity, -Infinity, 3.9, -3.9]) });
  assert.deepEqual(numericValues('i32', castRun.observations[0].outputs.output), [0, 2_147_483_647, -2_147_483_648, 3, -3]);
  summary.cast = numericValues('i32', castRun.observations[0].outputs.output);

  const fusion = TensorProgram.define((graph) => {
    const values = graph.input('values', { dtype: 'f32', capacityShape: [4], access: 'read-write' });
    return graph.unary('neg', graph.unary('sqrt', graph.unary('abs', values)));
  });
  const fusionInputs = { values: numericBytes('f32', [-4, 9, -16, 25]) };
  const unfusedRun = await execute(session, fusion, fusionInputs, { resolveOptions: { fusion: 'none' } });
  const fusedRun = await execute(session, fusion, fusionInputs, { replays: 2, resolveOptions: { fusion: 'exact-elementwise' } });
  const expectedFusion = [-2, -3, -4, -5];
  assert.deepEqual(numericValues('f32', unfusedRun.observations[0].outputs.output), expectedFusion);
  for (const replay of fusedRun.observations) assert.deepEqual(numericValues('f32', replay.outputs.output), expectedFusion);
  assert.equal(unfusedRun.fusionPolicy, 'none');
  assert.equal(unfusedRun.kernelCount, 3);
  assert.equal(fusedRun.fusionPolicy, 'exact-elementwise');
  assert.equal(fusedRun.fusionRegionCount, 1);
  assert.equal(fusedRun.fusedNodeCount, 3);
  assert.equal(fusedRun.kernelCount, 1);
  const typedFusion = TensorProgram.define((graph) => graph.unary('sqrt', graph.cast(
    graph.input('values', { dtype: 'f32', capacityShape: [2], access: 'read-write' }),
    'f64',
  )));
  const typedFusionRun = await execute(session, typedFusion, { values: numericBytes('f32', [4, 9]) }, { resolveOptions: { fusion: 'exact-elementwise' } });
  assert.deepEqual(numericValues('f64', typedFusionRun.observations[0].outputs.output), [2, 3]);
  assert.equal(typedFusionRun.fusedNodeCount, 2);
  const specialFusion = TensorProgram.define((graph) => graph.unary('neg', graph.unary('abs',
    graph.input('values', { dtype: 'f32', capacityShape: [4], access: 'read-write' }),
  )));
  const specialFusionRun = await execute(session, specialFusion, { values: numericBytes('f32', [-0, Number.NaN, -Infinity, 4]) }, { resolveOptions: { fusion: 'exact-elementwise' } });
  const specialOutput = numericValues('f32', specialFusionRun.observations[0].outputs.output);
  assert(Object.is(specialOutput[0], -0));
  assert(Number.isNaN(specialOutput[1]));
  assert.deepEqual(specialOutput.slice(2), [-Infinity, -4]);
  summary.fusion = {
    fusedKernels: fusedRun.kernelCount,
    unfusedKernels: unfusedRun.kernelCount,
    fusedNodes: fusedRun.fusedNodeCount,
    output: numericValues('f32', fusedRun.observations[0].outputs.output),
    typedOutput: numericValues('f64', typedFusionRun.observations[0].outputs.output),
    specialValues: { negativeZero: Object.is(specialOutput[0], -0), nan: Number.isNaN(specialOutput[1]), tail: specialOutput.slice(2) },
  };

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

  const deviceDense = TensorProgram.define((graph) => {
    const features = graph.input('features', { dtype: 'f32', capacityShape: [4, 3], access: 'read' });
    const weights = graph.input('weights', { dtype: 'f32', capacityShape: [3, 2], access: 'read' });
    const bias = graph.input('bias', { dtype: 'f32', capacityShape: [2], access: 'read' });
    const values = graph.binary('add', graph.matmul(features, weights), bias);
    const score = graph.reduce('sum', values, { axes: [1], order: 'fixed-tree-v1' });
    return { values, score };
  });
  const deviceProgram = await compileTensorDeviceProgram(session, deviceDense, { itemCapacity: 4, itemInputs: ['features'] });
  const pointerParameters = deviceProgram.parameters.filter((entry) => entry.role !== 'item-index');
  const kernelParameters = [...pointerParameters.map((entry) => ({ name: entry.parameterName, type: entry.type })), { name: 'status', type: 'ptr<u32>' }];
  const callArguments = ['itemIndex', ...pointerParameters.map((entry) => entry.parameterName)].join(', ');
  const callerSource = `function evaluateBatch(${kernelParameters.map((entry) => entry.name).join(', ')}) {
  let itemIndex = gpu.thread.globalX();
  status[itemIndex] = evaluateItem(${callArguments});
}`;
  const composed = await compileDeviceProgram(runtime, {
    source: callerSource,
    functions: [{ name: 'evaluateBatch', kind: 'kernel', parameters: kernelParameters, returns: 'void' }],
    imports: [deviceProgram.importAs('evaluateItem')],
  });
  const module = await runtime.loadModule({ format: composed.linker.artifact.format, bytes: composed.linker.artifact.bytes });
  const descriptor = composed.deviceProgram.kernels.find((entry) => entry.name === 'evaluateBatch');
  const fn = await module.getFunction({ name: descriptor.functionName, parameters: descriptor.parameters });
  const allocations = [];
  const views = [];
  let operation = null;
  try {
    const inputValues = {
      features: numericBytes('f32', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
      weights: numericBytes('f32', [1, 2, 3, 4, 5, 6]),
      bias: numericBytes('f32', [10, 20]),
    };
    const arguments_ = [];
    for (const parameter of pointerParameters) {
      const memory = await runtime.allocateDevice({ byteLength: parameter.byteLength });
      allocations.push(memory);
      const view = await memory.view({ dtype: parameter.dtype, elementCount: parameter.elementCount, access: parameter.access });
      views.push(view);
      arguments_.push(view);
      if (parameter.role === 'input') await memory.write(inputValues[parameter.name]);
    }
    const statusMemory = await runtime.allocateDevice({ byteLength: 8 * 4 });
    allocations.push(statusMemory);
    const statusView = await statusMemory.view({ dtype: 'u32', elementCount: 8, access: 'write' });
    views.push(statusView);
    arguments_.push(statusView);
    operation = await fn.submit({
      grid: { x: 1, y: 1, z: 1 },
      block: { x: 8, y: 1, z: 1 },
      arguments: arguments_,
      accesses: [
        ...pointerParameters.map((parameter, argumentIndex) => ({ argumentIndex, byteOffset: 0, byteLength: parameter.byteLength, mode: parameter.access })),
        { argumentIndex: pointerParameters.length, byteOffset: 0, byteLength: 32, mode: 'write' },
      ],
    });
    assert.equal((await operation.wait()).status, 'completed');
    const valuesParameter = pointerParameters.find((entry) => entry.role === 'output' && entry.name === 'values');
    const scoreParameter = pointerParameters.find((entry) => entry.role === 'output' && entry.name === 'score');
    const valuesMemory = allocations[pointerParameters.indexOf(valuesParameter)];
    const scoreMemory = allocations[pointerParameters.indexOf(scoreParameter)];
    const values = numericValues('f32', (await valuesMemory.read({ byteLength: valuesParameter.byteLength })).bytes);
    const scores = numericValues('f32', (await scoreMemory.read({ byteLength: scoreParameter.byteLength })).bytes);
    const statuses = Array.from(new Uint32Array((await statusMemory.read({ byteLength: 32 })).bytes.buffer));
    assert.deepEqual(values, [32, 48, 59, 84, 86, 120, 113, 156]);
    assert.deepEqual(scores, [80, 143, 206, 269]);
    assert.deepEqual(statuses, [0, 0, 0, 0, 1, 1, 1, 1]);
    summary.deviceCallable = {
      format: deviceProgram.library.format,
      architecture: deviceProgram.library.architecture,
      parameters: deviceProgram.parameters.length,
      workspaceBytes: deviceProgram.totalWorkspaceBytes,
      values,
      scores,
      statuses,
    };
  } finally {
    if (operation) await operation.close();
    await fn.close();
    await module.close();
    for (const view of views.reverse()) if (view.state === 'open') await view.close();
    for (const memory of allocations.reverse()) if (memory.state === 'open') await memory.close();
  }
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
console.log(JSON.stringify({ consumer: 'installed-native-resolved-dense', node: process.version, platform: process.platform, device: runtimeDescription.device?.architecture ?? null, summary, sessionGraceful: sessionTerminal.graceful, runtimeGraceful: runtimeTerminal.graceful }));
