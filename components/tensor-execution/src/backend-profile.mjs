import { TensorPlan } from '../../tensor-program/index.mjs';

import { deepFreeze, fail, identity } from './contract.mjs';

export const TENSOR_BACKEND_POLICIES = Object.freeze(['simt', 'prefer-cublaslt', 'cublaslt']);

function matmulDimensions(node, left, right) {
  const m = node.options.transposeA ? left.spec.logicalShape[1] : left.spec.logicalShape[0];
  const k = node.options.transposeA ? left.spec.logicalShape[0] : left.spec.logicalShape[1];
  const n = node.options.transposeB ? right.spec.logicalShape[0] : right.spec.logicalShape[1];
  return Object.freeze({ m, n, k });
}

function eligibility(node, lowering) {
  const left = lowering.references.get(node.inputIds[0]);
  const right = lowering.references.get(node.inputIds[1]);
  const output = lowering.references.get(node.id);
  const executionNode = lowering.kernels.find((kernel) => kernel.semanticNode === node.id) ?? null;
  const reasons = [];
  if (node.outputSpec.rank !== 2 || left.spec.rank !== 2 || right.spec.rank !== 2) reasons.push('rank-not-2');
  if (left.spec.dtype !== 'f32' || right.spec.dtype !== 'f32' || node.outputSpec.dtype !== 'f32') reasons.push('dtype-not-f32');
  if (node.options.accumulatorDtype !== 'f32') reasons.push('accumulator-not-f32');
  if (left.spec.layout !== 'row-major-contiguous' || right.spec.layout !== 'row-major-contiguous' || node.outputSpec.layout !== 'row-major-contiguous') reasons.push('layout-not-row-major-contiguous');
  if (left.spec.activeAxis0 !== null || right.spec.activeAxis0 !== null || node.outputSpec.activeAxis0 !== null) reasons.push('active-extent');
  if (left.spec.byteOffset !== left.originByteOffset || right.spec.byteOffset !== right.originByteOffset) reasons.push('derived-byte-offset');
  if (left.spec.byteOffset % 4 !== 0 || right.spec.byteOffset % 4 !== 0 || node.outputSpec.byteOffset % 4 !== 0) reasons.push('operand-alignment');
  if (!['read', 'read-write'].includes(left.spec.access) || !['read', 'read-write'].includes(right.spec.access) || node.outputSpec.access !== 'read-write') reasons.push('access-incompatible');
  if (node.outputSpec.logicalElementCount === 0 || executionNode === null) reasons.push('empty-matrix');

  const dimensions = reasons.includes('rank-not-2') ? null : matmulDimensions(node, left, right);
  if (dimensions && (dimensions.m < 1 || dimensions.n < 1 || dimensions.k < 1)) reasons.push('empty-matrix');
  const normalizedReasons = Object.freeze([...new Set(reasons)]);
  return Object.freeze({
    eligible: normalizedReasons.length === 0,
    reasons: normalizedReasons,
    executionNodeId: executionNode?.id ?? null,
    descriptor: normalizedReasons.length === 0 ? Object.freeze({
      semanticNode: node.id,
      executionNodeId: executionNode.id,
      after: executionNode.after,
      aBinding: left.binding,
      bBinding: right.binding,
      cBinding: output.binding,
      dBinding: output.binding,
      ...dimensions,
      transposeA: node.options.transposeA,
      transposeB: node.options.transposeB,
    }) : null,
  });
}

export function createBackendProfileRequest(plan, lowering, options) {
  if (!(plan instanceof TensorPlan)) fail('TENSOR_RESOLVE_PLAN_INVALID', 'validation', 'Backend selection requires a TensorPlan.');
  const matmuls = plan.program.nodes.filter((node) => node.op === 'matmul').map((node) => {
    const result = eligibility(node, lowering);
    return Object.freeze({ semanticNode: node.id, ...result });
  });
  if (options.backend === 'cublaslt') {
    if (matmuls.length === 0) fail('TENSOR_CUBLASLT_NODE_REQUIRED', 'unsupported', 'A strict cuBLASLt resolution requires at least one matmul node.');
    const rejected = matmuls.filter((entry) => !entry.eligible);
    if (rejected.length > 0) {
      fail('TENSOR_CUBLASLT_NODE_INELIGIBLE', 'unsupported', 'A strict cuBLASLt resolution contains an ineligible matmul node.', {
        nodes: rejected.map((entry) => ({ semanticNode: entry.semanticNode, reasons: entry.reasons })),
      });
    }
  }
  const canonical = deepFreeze({
    contract: 'SPEC-0006-cublaslt-profile-request-v1',
    planIdentity: plan.compatibilityIdentity,
    completeSimtLoweringIdentity: lowering.compatibilityIdentity,
    policy: options.backend,
    alignment: { operandByteOffset: 4 },
    matmuls: matmuls.map((entry) => ({
      semanticNode: entry.semanticNode,
      executionNodeId: entry.executionNodeId,
      eligible: entry.eligible,
      reasons: [...entry.reasons],
      descriptor: entry.descriptor ? { ...entry.descriptor, after: [...entry.descriptor.after] } : null,
    })),
  });
  return Object.freeze({
    policy: options.backend,
    matmuls: Object.freeze(matmuls),
    candidates: Object.freeze(matmuls.filter((entry) => entry.eligible && options.backend !== 'simt').map((entry) => entry.descriptor)),
    canonical,
    compatibilityIdentity: identity('tensor-backend-profile-request-v1', canonical),
  });
}

export function realizeBackendProfile(request, lowering, outcomes, provider = null) {
  const outcomeByNode = new Map(outcomes.map((entry) => [entry.semanticNode, entry]));
  const operationByNode = new Map(lowering.kernels.map((kernel) => [kernel.id, kernel]));
  const nodes = lowering.kernels.map((kernel) => {
    const outcome = outcomeByNode.get(kernel.semanticNode);
    return Object.freeze({
      id: kernel.id,
      semanticNode: kernel.semanticNode,
      backend: outcome?.backend === 'cublaslt' ? 'cublaslt' : 'simt',
    });
  });
  const matmuls = request.matmuls.map((entry) => {
    const outcome = outcomeByNode.get(entry.semanticNode);
    const backend = outcome?.backend === 'cublaslt' ? 'cublaslt' : 'simt';
    const reasons = backend === 'cublaslt'
      ? []
      : [...entry.reasons, ...(outcome?.reasons ?? []), ...(request.policy === 'simt' && entry.reasons.length === 0 ? ['policy-simt'] : [])];
    return Object.freeze({
      semanticNode: entry.semanticNode,
      executionNodeId: entry.executionNodeId,
      eligible: entry.eligible,
      backend,
      reasons: Object.freeze([...new Set(reasons)]),
      plan: outcome?.plan ?? null,
      workspace: outcome?.workspace ?? null,
    });
  });
  const cublasLtNodeCount = nodes.filter((entry) => entry.backend === 'cublaslt').length;
  const simtNodeCount = nodes.length - cublasLtNodeCount;
  const backend = cublasLtNodeCount === 0 ? 'simt' : simtNodeCount === 0 ? 'cublaslt' : 'mixed';
  const canonical = deepFreeze({
    contract: 'SPEC-0006-resolved-backend-profile-v1',
    requestIdentity: request.compatibilityIdentity,
    policy: request.policy,
    backend,
    simtNodeCount,
    cublasLtNodeCount,
    nodes: nodes.map((entry) => ({ ...entry })),
    matmuls: matmuls.map((entry) => ({ ...entry, reasons: [...entry.reasons], plan: entry.plan ? { ...entry.plan, requirements: { ...entry.plan.requirements } } : null, workspace: entry.workspace ? { ...entry.workspace } : null })),
    provider: provider ? { profile: provider.profile, identity: { ...provider.identity } } : null,
  });
  for (const node of nodes) if (!operationByNode.has(node.id)) fail('TENSOR_BACKEND_NODE_MISSING', 'internal', 'Resolved backend profile references an unknown execution node.', { node: node.id });
  return Object.freeze({
    policy: request.policy,
    backend,
    simtNodeCount,
    cublasLtNodeCount,
    nodes: Object.freeze(nodes),
    matmuls: Object.freeze(matmuls),
    provider,
    canonical,
    compatibilityIdentity: identity('tensor-resolved-backend-profile-v1', canonical),
  });
}
