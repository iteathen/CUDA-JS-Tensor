import { compileDeviceProgram } from 'cuda-js';

import { failureSummary, fail, identity } from './contract.mjs';

async function closeResources(resources) {
  const failures = [];
  if (resources.dag?.state === 'open') {
    try { await resources.dag.close(); } catch (error) { failures.push(failureSummary(error)); }
  }
  for (const fn of [...resources.functions].reverse()) {
    if (fn?.state !== 'open') continue;
    try { await fn.close(); } catch (error) { failures.push(failureSummary(error)); }
  }
  if (resources.module?.state === 'open') {
    try { await resources.module.close(); } catch (error) { failures.push(failureSummary(error)); }
  }
  return Object.freeze({ graceful: failures.length === 0, failures: Object.freeze(failures) });
}

export async function createCudaJsSimtBackend(runtime, lowering) {
  if (lowering.kernels.length === 0) {
    return Object.freeze({
      identity: identity('tensor-simt-empty-backend-v1', { lowering: lowering.compatibilityIdentity }),
      descriptor: Object.freeze({ realization: 'empty', compiler: null, module: null, prepared: null }),
      async execute() { return Object.freeze({ status: 'completed', realization: 'empty' }); },
      async close() { return Object.freeze({ graceful: true, failures: Object.freeze([]) }); },
    });
  }

  const resources = { module: null, functions: [], dag: null };
  try {
    const compiled = await compileDeviceProgram(runtime, { source: lowering.source, functions: lowering.functions });
    resources.module = await runtime.loadModule({ format: compiled.compiler.artifact.format, bytes: compiled.compiler.artifact.bytes });
    const descriptors = new Map(compiled.deviceProgram.kernels.map((entry) => [entry.name, entry]));
    for (const kernel of lowering.kernels) {
      const descriptor = descriptors.get(kernel.functionName);
      if (!descriptor) fail('TENSOR_SIMT_COMPILED_KERNEL_MISSING', 'internal', 'Compiled Device-JS output omitted a generated tensor kernel.', { functionName: kernel.functionName });
      resources.functions.push(await resources.module.getFunction({ name: descriptor.functionName, parameters: descriptor.parameters }));
    }
    const functionByName = new Map(lowering.kernels.map((kernel, index) => [kernel.functionName, resources.functions[index]]));
    resources.dag = await runtime.prepareOperationDag(lowering.kernels.map((kernel) => ({
      id: kernel.id,
      ...(kernel.after.length ? { after: kernel.after } : {}),
      function: functionByName.get(kernel.functionName),
      grid: kernel.grid,
      block: kernel.block,
      arguments: kernel.parameterRecords.map((entry) => ({ binding: entry.binding })),
      accesses: kernel.accesses,
    })));
    const descriptor = Object.freeze({
      realization: resources.dag.realization,
      compiler: Object.freeze({ sha256: compiled.compiler.artifact.sha256, format: compiled.compiler.artifact.format, architecture: compiled.compiler.artifact.architecture, headerProfile: compiled.compiler.headerProfile }),
      deviceProgram: Object.freeze({ contract: compiled.deviceProgram.contract, sha256: compiled.deviceProgram.sha256 }),
      module: Object.freeze({ sha256: resources.module.sha256, format: resources.module.format }),
      prepared: Object.freeze({ sha256: resources.dag.sha256, nodeCount: resources.dag.nodeCount, edgeCount: resources.dag.edgeCount }),
    });
    let closed = false;
    return Object.freeze({
      identity: identity('tensor-cuda-js-simt-backend-v1', descriptor),
      descriptor,
      async execute(bindings) {
        if (closed) fail('TENSOR_RESOLVED_PLAN_CLOSED', 'closed-plan', 'The resolved CUDA-JS SIMT backend is closed.');
        let operation = null;
        let primary = null;
        try {
          operation = await resources.dag.submit({ bindings });
          const terminal = await operation.wait();
          if (terminal.status !== 'completed') fail('TENSOR_EXECUTION_FAILED', 'execution', 'CUDA-JS did not report completed prepared execution.', { status: terminal.status });
          return Object.freeze({ status: terminal.status, realization: resources.dag.realization, operationSequence: terminal.operationSequence, preparedSha256: terminal.preparedSha256 });
        } catch (error) {
          primary = error;
          throw error;
        } finally {
          if (operation) {
            try { await operation.close(); } catch (cleanupError) {
              fail('TENSOR_EXECUTION_OPERATION_CLEANUP_UNPROVED', 'cleanup-unproved', 'CUDA-JS operation cleanup was not proved.', {
                ...(primary ? { primary: failureSummary(primary) } : {}),
                cleanup: failureSummary(cleanupError),
              }, { cause: primary ?? cleanupError });
            }
          }
        }
      },
      async close() {
        if (closed) return Object.freeze({ graceful: true, failures: Object.freeze([]) });
        const report = await closeResources(resources);
        if (report.graceful) closed = true;
        return report;
      },
    });
  } catch (error) {
    const cleanup = await closeResources(resources);
    if (!cleanup.graceful) {
      fail('TENSOR_RESOLVE_ROLLBACK_UNPROVED', 'cleanup-unproved', 'SIMT resolution failed and CUDA-JS rollback was not proved.', { primary: failureSummary(error), cleanup: cleanup.failures }, { cause: error });
    }
    throw error;
  }
}
