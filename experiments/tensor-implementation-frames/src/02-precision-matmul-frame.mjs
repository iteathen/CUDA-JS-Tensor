import { frameRecord } from './frame-contract.mjs';

const FUTURE_DTYPES = new Set(['f16', 'bf16', 'f64']);

// Chunk 2 separates dtype/accumulation expansion from batch expansion.  It does
// not assume that a provider's supported CUDA datatype has the same numeric
// contract as SPEC-0004.  In particular, f16/bf16 inputs with f32 accumulation
// need explicit compute, scale, rounding, tolerance, and output facts before a
// library algorithm can be selected truthfully.
export function framePrecisionMatmul(plan) {
  const candidates = [];
  for (const node of plan.program.nodes) {
    if (node.op !== 'matmul') continue;
    const left = plan.program.valueSpec(node.inputIds[0]);
    const right = plan.program.valueSpec(node.inputIds[1]);
    if (!FUTURE_DTYPES.has(left.dtype) && !FUTURE_DTYPES.has(right.dtype)) continue;
    const reasons = [];
    if (left.dtype !== right.dtype || left.dtype !== node.outputSpec.dtype) reasons.push('operand-output-dtype-mismatch');
    if (!FUTURE_DTYPES.has(left.dtype)) reasons.push('dtype-outside-framed-profile');
    if (![2, 3].includes(node.outputSpec.rank) || left.rank !== node.outputSpec.rank || right.rank !== node.outputSpec.rank) reasons.push('rank-mismatch');
    if ([left, right, node.outputSpec].some((spec) => spec.layout !== 'row-major-contiguous')) reasons.push('layout-not-contiguous');
    if ([left, right, node.outputSpec].some((spec) => spec.activeAxis0 !== null)) reasons.push('active-extent');
    candidates.push({
      semanticNode: node.id,
      dtype: left.dtype,
      accumulatorDtype: node.options.accumulatorDtype,
      rank: node.outputSpec.rank,
      structurallyEligible: reasons.length === 0,
      reasons,
    });
  }

  return frameRecord({
    kind: 'cublaslt-precision-matmul',
    plan,
    scope: 'Contiguous f16, bf16, and f64 matmul profiles with exact declared accumulation semantics.',
    candidates,
    blockers: [
      'The current public CUDA-JS prepared cuBLASLt node is fixed to f32.',
      'Provider compute/scale/output types and precision equivalence are not represented by an accepted Tensor child profile.',
      'No end-to-end evidence selects reduced precision or Tensor Core use as a recommendation.',
    ],
    completion: [
      'Accept exact public CUDA-JS typed matmul plan/node profiles.',
      'Bind Tensor dtype and accumulator semantics to explicit provider facts.',
      'Implement fixed-plan substitution, workspace ownership, fallback, and cleanup per dtype profile.',
    ],
  });
}
