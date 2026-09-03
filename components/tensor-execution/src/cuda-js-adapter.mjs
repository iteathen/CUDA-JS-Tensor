import { compileDeviceProgram } from 'cuda-js';

import { failureSummary, fail, identity } from './contract.mjs';
import { realizeBackendProfile } from './backend-profile.mjs';
import { CUDA_JS_TENSOR_COMPATIBILITY } from './cuda-js-compatibility.mjs';

const TENSOR_PREFERENCE_FALLBACK_REASONS = new Map([
  ['TENSOR_CUBLASLT_BINDING_LIMIT', 'binding-limit'],
  ['TENSOR_CUBLASLT_WORKSPACE_COUNT_LIMIT', 'workspace-count-limit'],
]);
const DEFAULT_PORTS = Object.freeze({ compileDeviceProgram });
const PREPARED_DAG_LIMITS = CUDA_JS_TENSOR_COMPATIBILITY.preparedOperationDagLimits;

function tensorCleanupFailure(code, message, primary, cleanup) {
  fail(code, 'cleanup-unproved', message, { primary: failureSummary(primary), cleanup }, { cause: primary });
}

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
  for (const plan of [...resources.plans].reverse()) {
    if (plan?.state !== 'open') continue;
    try { await plan.close(); } catch (error) { failures.push(failureSummary(error)); }
  }
  if (resources.adapter) {
    const adapter = resources.adapter;
    resources.adapter = null;
    try { await adapter.close(); } catch (error) { failures.push(failureSummary(error)); }
  }
  return Object.freeze({ graceful: failures.length === 0, failures: Object.freeze(failures) });
}

function fallbackAllowed(request, error) {
  return request.policy === 'prefer-cublaslt'
    && (error?.category === 'unsupported' || TENSOR_PREFERENCE_FALLBACK_REASONS.has(error?.code));
}

function planFallbackReason(error) {
  return TENSOR_PREFERENCE_FALLBACK_REASONS.get(error?.code) ?? 'plan-unsupported';
}

function simtOutcome(entry, reasons = []) {
  return Object.freeze({ semanticNode: entry.semanticNode, backend: 'simt', reasons: Object.freeze(reasons) });
}

function planRecord(plan) {
  return Object.freeze({
    contract: plan.contract,
    m: plan.m,
    n: plan.n,
    k: plan.k,
    transposeA: plan.transposeA,
    transposeB: plan.transposeB,
    maxWorkspaceBytes: plan.maxWorkspaceBytes,
    workspaceBytes: plan.workspaceBytes,
    requirements: Object.freeze({ ...plan.requirements }),
  });
}

async function compileSimtKernels(runtime, lowering, selectedNodeIds, resources, compileProgram) {
  const kernels = lowering.kernels.filter((kernel) => !selectedNodeIds.has(kernel.id));
  if (kernels.length === 0) return Object.freeze({ kernels: Object.freeze([]), functionByName: new Map(), compiler: null, deviceProgram: null, module: null });
  const functionNames = new Set(kernels.map((kernel) => kernel.functionName));
  const functions = lowering.functions.filter((entry) => functionNames.has(entry.name));
  const source = kernels.map((kernel) => kernel.source).join('\n\n');
  const compiled = await compileProgram(runtime, { source, functions });
  resources.module = await runtime.loadModule({ format: compiled.compiler.artifact.format, bytes: compiled.compiler.artifact.bytes });
  const descriptors = new Map(compiled.deviceProgram.kernels.map((entry) => [entry.name, entry]));
  for (const kernel of kernels) {
    const descriptor = descriptors.get(kernel.functionName);
    if (!descriptor) fail('TENSOR_SIMT_COMPILED_KERNEL_MISSING', 'internal', 'Compiled Device-JS output omitted a generated tensor kernel.', { functionName: kernel.functionName });
    resources.functions.push(await resources.module.getFunction({ name: descriptor.functionName, parameters: descriptor.parameters }));
  }
  const functionByName = new Map(kernels.map((kernel, index) => [kernel.functionName, resources.functions[index]]));
  return Object.freeze({
    kernels: Object.freeze(kernels),
    functionByName,
    compiler: Object.freeze({
      sha256: compiled.compiler.artifact.sha256,
      format: compiled.compiler.artifact.format,
      architecture: compiled.compiler.artifact.architecture,
      headerProfile: compiled.compiler.headerProfile,
    }),
    deviceProgram: Object.freeze({ contract: compiled.deviceProgram.contract, sha256: compiled.deviceProgram.sha256 }),
    module: Object.freeze({ sha256: resources.module.sha256, format: resources.module.format }),
  });
}

export async function createCudaJsTensorBackend(runtime, lowering, request, options, ports = DEFAULT_PORTS) {
  if (ports === null || typeof ports !== 'object' || Object.keys(ports).some((key) => key !== 'compileDeviceProgram') || typeof ports.compileDeviceProgram !== 'function') {
    fail('TENSOR_BACKEND_PORTS_INVALID', 'validation', 'Tensor backend ports require exactly one compileDeviceProgram function.');
  }
  const resources = { module: null, functions: [], dag: null, plans: [], adapter: null };
  const outcomes = request.matmuls.map((entry) => simtOutcome(entry));
  const outcomeIndex = new Map(outcomes.map((entry, index) => [entry.semanticNode, index]));
  const selected = new Map();
  const workspaces = [];
  let provider = null;
  let selectedWorkspaceBytes = 0;

  try {
    if (request.candidates.length > 0) {
      try {
        resources.adapter = await runtime.openCublasLt();
        provider = Object.freeze({
          profile: resources.adapter.profile,
          identity: Object.freeze({ ...resources.adapter.provider }),
        });
        if (!Number.isSafeInteger(provider.identity.workspaceAlignmentBytes) || provider.identity.workspaceAlignmentBytes < 1) {
          fail('TENSOR_CUBLASLT_PROVIDER_ALIGNMENT_INVALID', 'internal', 'CUDA-JS returned an invalid cuBLASLt workspace alignment capability.', {
            workspaceAlignmentBytes: provider.identity.workspaceAlignmentBytes,
          });
        }
      } catch (error) {
        if (request.policy !== 'prefer-cublaslt' || error?.category !== 'unsupported') throw error;
        for (const candidate of request.candidates) {
          outcomes[outcomeIndex.get(candidate.semanticNode)] = simtOutcome(candidate, ['provider-unsupported']);
        }
      }
    }

    if (resources.adapter) {
      for (const candidate of request.candidates) {
        const remainingWorkspaceBytes = Math.min(
          options.acceleratorWorkspaceBytes - selectedWorkspaceBytes,
          options.maxAcceleratorWorkspaceBytesPerPlan,
        );
        let plan = null;
        try {
          plan = await resources.adapter.createF32MatmulPlan({
            m: candidate.m,
            n: candidate.n,
            k: candidate.k,
            transposeA: candidate.transposeA,
            transposeB: candidate.transposeB,
            maxWorkspaceBytes: remainingWorkspaceBytes,
          });
          resources.plans.push(plan);
          if (!Number.isSafeInteger(plan.workspaceBytes) || plan.workspaceBytes < 0 || plan.workspaceBytes > remainingWorkspaceBytes || plan.workspaceBytes % 4 !== 0) {
            fail('TENSOR_CUBLASLT_WORKSPACE_INVALID', 'internal', 'CUDA-JS returned a cuBLASLt workspace outside the selected bounded profile.', { workspaceBytes: plan.workspaceBytes, maximum: remainingWorkspaceBytes });
          }
          const needsBinding = plan.workspaceBytes > 0;
          if (needsBinding && workspaces.length >= options.maxAcceleratorWorkspaceCount) {
            fail('TENSOR_CUBLASLT_WORKSPACE_COUNT_LIMIT', 'pressure', 'Selected cuBLASLt workspace would exceed the resolved session tensor-count gate.', { maximum: options.maxAcceleratorWorkspaceCount });
          }
          if (needsBinding && lowering.bindings.length + workspaces.length >= PREPARED_DAG_LIMITS.bindings) {
            fail('TENSOR_CUBLASLT_BINDING_LIMIT', 'pressure', 'Selected cuBLASLt workspace would exceed the prepared binding limit.', { maximum: PREPARED_DAG_LIMITS.bindings });
          }
          const workspace = needsBinding ? Object.freeze({
            id: `cublaslt-workspace:${candidate.semanticNode}`,
            binding: `b${lowering.bindings.length + workspaces.length}`,
            dtype: 'u32',
            elementCount: plan.workspaceBytes / 4,
            byteLength: plan.workspaceBytes,
            requiredByteOffsetAlignment: provider.identity.workspaceAlignmentBytes,
          }) : null;
          if (workspace) workspaces.push(workspace);
          selectedWorkspaceBytes += plan.workspaceBytes;
          const outcome = Object.freeze({ semanticNode: candidate.semanticNode, backend: 'cublaslt', plan: planRecord(plan), workspace, reasons: Object.freeze([]) });
          outcomes[outcomeIndex.get(candidate.semanticNode)] = outcome;
          selected.set(candidate.executionNodeId, Object.freeze({ candidate, plan, workspace }));
        } catch (error) {
          if (!fallbackAllowed(request, error)) throw error;
          if (plan?.state === 'open') {
            try { await plan.close(); } catch (cleanupError) { tensorCleanupFailure('TENSOR_CUBLASLT_PLAN_ROLLBACK_UNPROVED', 'Rejected cuBLASLt plan cleanup was not proved.', error, [failureSummary(cleanupError)]); }
          }
          outcomes[outcomeIndex.get(candidate.semanticNode)] = simtOutcome(candidate, [planFallbackReason(error)]);
        }
      }
    }

    if (request.policy === 'cublaslt' && selected.size !== request.candidates.length) {
      fail('TENSOR_CUBLASLT_STRICT_SELECTION_FAILED', 'unsupported', 'A strict cuBLASLt resolution did not select every eligible matmul.');
    }

    if (selected.size === 0 && resources.adapter) {
      const adapter = resources.adapter;
      resources.adapter = null;
      try { await adapter.close(); }
      catch (error) {
        fail('TENSOR_CUBLASLT_ADAPTER_ROLLBACK_UNPROVED', 'cleanup-unproved', 'Unused cuBLASLt borrower cleanup was not proved.', { cleanup: failureSummary(error) }, { cause: error });
      }
    }

    const compiled = await compileSimtKernels(runtime, lowering, new Set(selected.keys()), resources, ports.compileDeviceProgram);
    const nodes = lowering.kernels.map((kernel) => {
      const accelerated = selected.get(kernel.id);
      if (accelerated) {
        return Object.freeze({
          id: kernel.id,
          kind: 'cublaslt-f32-matmul',
          ...(kernel.after.length ? { after: kernel.after } : {}),
          plan: accelerated.plan,
          a: Object.freeze({ binding: accelerated.candidate.aBinding }),
          b: Object.freeze({ binding: accelerated.candidate.bBinding }),
          c: Object.freeze({ binding: accelerated.candidate.cBinding }),
          d: Object.freeze({ binding: accelerated.candidate.dBinding }),
          alpha: 1,
          beta: 0,
          ...(accelerated.workspace ? { workspace: Object.freeze({ binding: accelerated.workspace.binding }) } : {}),
        });
      }
      return Object.freeze({
        id: kernel.id,
        ...(kernel.after.length ? { after: kernel.after } : {}),
        function: compiled.functionByName.get(kernel.functionName),
        grid: kernel.grid,
        block: kernel.block,
        arguments: kernel.parameterRecords.map((entry) => ({ binding: entry.binding })),
        accesses: kernel.accesses,
      });
    });
    if (nodes.length > 0) resources.dag = await runtime.prepareOperationDag(nodes);

    const profile = realizeBackendProfile(request, lowering, outcomes, provider);
    const bindingRecords = Object.freeze([
      ...lowering.bindings,
      ...workspaces.map((entry) => Object.freeze({ name: entry.binding, role: 'accelerator-workspace', dtype: entry.dtype, byteLength: entry.byteLength, valueId: entry.id })),
    ]);
    const descriptor = Object.freeze({
      realization: resources.dag?.realization ?? 'empty',
      profile: profile.canonical,
      compiler: compiled.compiler,
      deviceProgram: compiled.deviceProgram,
      module: compiled.module,
      prepared: resources.dag ? Object.freeze({ contract: resources.dag.contract, sha256: resources.dag.sha256, nodeCount: resources.dag.nodeCount, edgeCount: resources.dag.edgeCount }) : null,
      workspaceBytes: lowering.totalWorkspaceBytes + selectedWorkspaceBytes,
    });
    let closed = false;
    return Object.freeze({
      identity: identity('tensor-cuda-js-backend-v1', descriptor),
      descriptor,
      profile,
      bindingRecords,
      workspaces: Object.freeze(workspaces),
      workspaceBytes: lowering.totalWorkspaceBytes + selectedWorkspaceBytes,
      async execute(bindings) {
        if (closed) fail('TENSOR_RESOLVED_PLAN_CLOSED', 'closed-plan', 'The resolved CUDA-JS tensor backend is closed.');
        if (!resources.dag) return Object.freeze({ status: 'completed', realization: 'empty' });
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
    if (!cleanup.graceful) tensorCleanupFailure('TENSOR_RESOLVE_ROLLBACK_UNPROVED', 'Tensor backend resolution failed and CUDA-JS rollback was not proved.', error, cleanup.failures);
    throw error;
  }
}
