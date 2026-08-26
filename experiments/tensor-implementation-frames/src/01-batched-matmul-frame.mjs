import { assertTensorPlan, frameRecord } from './frame-contract.mjs';

const TARGET_DTYPE = 'f32';

function matmulDimensions(node, left, right) {
  const leftShape = left.capacityShape;
  const rightShape = right.capacityShape;
  return Object.freeze({
    batchA: leftShape[0],
    batchB: rightShape[0],
    batch: node.outputSpec.capacityShape[0],
    m: node.options.transposeA ? leftShape[2] : leftShape[1],
    k: node.options.transposeA ? leftShape[1] : leftShape[2],
    n: node.options.transposeB ? rightShape[1] : rightShape[2],
  });
}

function analyzeMatmulReasons(left, right, node) {
  const reasons = [];
  if (left.rank !== 3 || right.rank !== 3) reasons.push('rank-mismatch');
  if (left.dtype !== TARGET_DTYPE || right.dtype !== TARGET_DTYPE || node.outputSpec.dtype !== TARGET_DTYPE) reasons.push('dtype-not-f32');
  if (node.options.accumulatorDtype !== TARGET_DTYPE) reasons.push('accumulator-not-f32');
  if ([left, right, node.outputSpec].some((spec) => spec.layout !== 'row-major-contiguous')) reasons.push('layout-not-contiguous');
  if ([left, right, node.outputSpec].some((spec) => spec.activeAxis0 !== null)) reasons.push('active-batch-extent');
  if (left.byteOffset !== 0 || right.byteOffset !== 0 || node.outputSpec.byteOffset !== 0) reasons.push('derived-offset');
  if (node.outputSpec.logicalElementCount === 0) reasons.push('empty-output');
  if (left.rank === 3 && matmulDimensions(node, left, right).k === 0) reasons.push('empty-contraction');
  return Object.freeze([...new Set(reasons)]);
}

function profile(node, left, right, reasons) {
  return Object.freeze({
    semanticNode: node.id,
    rank: node.outputSpec.rank,
    dtype: left.dtype,
    transposeA: node.options.transposeA,
    transposeB: node.options.transposeB,
    accumulatorDtype: node.options.accumulatorDtype,
    dimensions: reasons.includes('rank-mismatch') ? null : matmulDimensions(node, left, right),
    byteOffsets: Object.freeze({
      left: left.byteOffset,
      right: right.byteOffset,
      output: node.outputSpec.byteOffset,
    }),
    activeAxis0: Object.freeze({
      left: left.activeAxis0,
      right: right.activeAxis0,
      output: node.outputSpec.activeAxis0,
    }),
    reasons,
    coveredByCurrentSIMTSemantics: true,
    supportedByCurrentCublasLtProfile: false,
    structurallyEligibleForProposedCublasLtProfile: reasons.length === 0,
  });
}

export function frameBatchedMatmul(plan) {
  assertTensorPlan(plan, 'frameBatchedMatmul');
  const candidates = [];
  for (const node of plan.program.nodes) {
    if (node.op !== 'matmul' || node.outputSpec.rank !== 3) continue;
    const left = plan.program.valueSpec(node.inputIds[0]);
    const right = plan.program.valueSpec(node.inputIds[1]);
    const reasons = analyzeMatmulReasons(left, right, node);
    const entry = profile(node, left, right, reasons);
    candidates.push(entry);
  }

  return frameRecord({
    kind: 'cublaslt-batched-matmul',
    plan,
    scope: 'Nonempty rank-3 contiguous f32 matmul with f32 accumulation and explicit finite batch broadcasting.',
    candidates,
    blockers: [
      'CUDA-JS has no accepted public batched/strided cuBLASLt prepared-node contract for this profile.',
      'Batch broadcasting must map to one bounded public plan without host-side launch loops in steady-state.',
      'Workspace, access-range, alias, failure, and cleanup identity are not yet specified.',
    ],
    completion: [
      'Accept the consumer-neutral CUDA-JS batched plan/node contract.',
      'Replace candidate analysis with fixed public plan creation and same-DAG substitution.',
      'Retain complete SIMT fallback and prove strict/preference behavior.',
    ],
  });
}
