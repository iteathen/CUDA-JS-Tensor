import { assertTensorPlan, frameRecord } from './frame-contract.mjs';

const FUTURE_DTYPES = new Set(['f16', 'bf16', 'f64']);

function supportedRank(node) {
  return [2, 3].includes(node.outputSpec.rank) ? node.outputSpec.rank : null;
}

function contractionExtent(node, left) {
  if (![2, 3].includes(left.rank)) return null;
  const shape = left.capacityShape;
  return node.options.transposeA ? shape.at(-2) : shape.at(-1);
}

function matmulReasons(node, left, right, rank) {
  const reasons = [];
  if (left.dtype !== right.dtype || left.dtype !== node.outputSpec.dtype) reasons.push('operand-output-dtype-mismatch');
  if (!FUTURE_DTYPES.has(left.dtype)) reasons.push('dtype-not-f16-bf16-f64');
  if (rank === null) reasons.push('rank-mismatch');
  if (![left, right, node.outputSpec].every((spec) => spec.layout === 'row-major-contiguous')) reasons.push('layout-not-contiguous');
  if (left.activeAxis0 || right.activeAxis0 || node.outputSpec.activeAxis0) reasons.push('active-extent');
  if (left.byteOffset || right.byteOffset || node.outputSpec.byteOffset) reasons.push('derived-offset');
  if (node.outputSpec.logicalElementCount === 0) reasons.push('empty-output');
  if (contractionExtent(node, left) === 0) reasons.push('empty-contraction');
  return Object.freeze([...new Set(reasons)]);
}

function classify(node, left, right, rank) {
  const reasons = matmulReasons(node, left, right, rank);
  return Object.freeze({
    semanticNode: node.id,
    rank,
    dtype: left.dtype,
    rankCompatible: rank !== null,
    accumulatorDtype: node.options.accumulatorDtype,
    outputDtype: node.outputSpec.dtype,
    contiguous: [left, right, node.outputSpec].every((spec) => spec.layout === 'row-major-contiguous'),
    coveredByCurrentSIMTSemantics: true,
    supportedByCurrentCublasLtProfile: false,
    reasons,
    implementationGap: FUTURE_DTYPES.has(left.dtype)
      ? `Need public CUDA-JS typed matmul support for ${left.dtype} input/output with ${node.options.accumulatorDtype} accumulation.`
      : null,
    structurallyEligibleForProposedProfile: reasons.length === 0,
  });
}

export function framePrecisionMatmul(plan) {
  assertTensorPlan(plan, 'framePrecisionMatmul');
  const candidates = [];
  for (const node of plan.program.nodes) {
    if (node.op !== 'matmul') continue;
    const left = plan.program.valueSpec(node.inputIds[0]);
    const right = plan.program.valueSpec(node.inputIds[1]);
    const rank = supportedRank(node);
    const profile = classify(node, left, right, rank);
    if (FUTURE_DTYPES.has(left.dtype) || FUTURE_DTYPES.has(right.dtype) || FUTURE_DTYPES.has(node.outputSpec.dtype)) {
      candidates.push(profile);
    }
  }

  return frameRecord({
    kind: 'cublaslt-precision-matmul',
    plan,
    scope: 'Contiguous f16, bf16, and f64 matmul profiles with exact declared accumulation semantics.',
    candidates,
    blockers: [
      'The current public CUDA-JS prepared cuBLASLt path remains fixed to rank-2 f32.',
      'Typed compute/input/accumulation/output contracts for these dtypes are not yet represented by a Tensor child profile.',
      'No exact numeric-compatibility or error-bound evidence exists for fast mixed-precision selection.',
    ],
    completion: [
      'Accept exact public CUDA-JS typed matmul plan/node profiles.',
      'Bind Tensor dtype and accumulator semantics to explicit provider facts.',
      'Implement fixed-plan substitution, workspace ownership, fallback, and cleanup per dtype profile.',
    ],
  });
}
