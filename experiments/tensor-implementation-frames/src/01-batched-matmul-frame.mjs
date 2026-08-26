import { frameRecord } from './frame-contract.mjs';

function dimensions(node, left, right) {
  const leftShape = left.logicalShape;
  const rightShape = right.logicalShape;
  return {
    batchA: leftShape[0],
    batchB: rightShape[0],
    batch: node.outputSpec.logicalShape[0],
    m: node.options.transposeA ? leftShape[2] : leftShape[1],
    k: node.options.transposeA ? leftShape[1] : leftShape[2],
    n: node.options.transposeB ? rightShape[1] : rightShape[2],
  };
}

// Chunk 1 extends host-planned cuBLASLt selection to SPEC-0004 rank-3
// batched matmul.  The implemented f32 rank-2 path cannot simply loop here:
// doing that would create host submissions and scheduling semantics that are not
// present in one prepared batched operation.  This function therefore performs
// only the backend-neutral candidate analysis that a future resolver will need.
export function frameBatchedMatmul(plan) {
  const candidates = [];
  for (const node of plan.program.nodes) {
    if (node.op !== 'matmul' || node.outputSpec.rank !== 3) continue;
    const left = plan.program.valueSpec(node.inputIds[0]);
    const right = plan.program.valueSpec(node.inputIds[1]);
    const reasons = [];
    if (left.rank !== 3 || right.rank !== 3) reasons.push('rank-mismatch');
    if (left.dtype !== 'f32' || right.dtype !== 'f32' || node.outputSpec.dtype !== 'f32') reasons.push('dtype-not-f32');
    if (node.options.accumulatorDtype !== 'f32') reasons.push('accumulator-not-f32');
    if ([left, right, node.outputSpec].some((spec) => spec.layout !== 'row-major-contiguous')) reasons.push('layout-not-contiguous');
    if ([left, right, node.outputSpec].some((spec) => spec.activeAxis0 !== null)) reasons.push('active-batch-extent');
    if (node.outputSpec.logicalElementCount === 0) reasons.push('empty-output');
    candidates.push({
      semanticNode: node.id,
      structurallyEligible: reasons.length === 0,
      reasons,
      dimensions: reasons.includes('rank-mismatch') ? null : dimensions(node, left, right),
      transposeA: node.options.transposeA,
      transposeB: node.options.transposeB,
    });
  }

  return frameRecord({
    kind: 'cublaslt-batched-matmul',
    plan,
    scope: 'Rank-3 contiguous f32 matmul with finite explicit batch broadcasting.',
    candidates,
    blockers: [
      'CUDA-JS has no accepted public batched/strided cuBLASLt prepared-node contract for this profile.',
      'Batch broadcasting must map to one bounded public plan without a host submission loop.',
      'Workspace, access-range, alias, failure, and cleanup identity are not yet specified.',
    ],
    completion: [
      'Accept the consumer-neutral CUDA-JS batched plan/node contract.',
      'Replace candidate analysis with fixed public plan creation and same-DAG substitution.',
      'Retain complete SIMT fallback and prove strict/preference behavior.',
    ],
  });
}

