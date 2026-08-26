# Remaining tensor implementation frames

**Status:** Implemented analysis scaffolding; not implementation authority

**Owner direction:** Create real, well-commented incomplete code quickly and preserve deterministic analysis without changing production claims

## Outcome

The remaining known tensor work was split into five independently finishable implementation chunks under [`experiments/tensor-implementation-frames/`](../../experiments/tensor-implementation-frames/). Four analysis frames remain outside production components and package exports. The fusion frame was deleted after its unique continuation value moved into accepted SPEC-0007 production source and focused tests.

The remaining frames do not change the implemented alpha.5 support surface. Complete SIMT execution, optional exact elementwise fusion and optional contiguous rank-2 f32 cuBLASLt matmul are the production execution profiles.

## Chunk map

| ID | Owned outcome | Real code preserved now | Required before production | Explicit non-goals |
|---|---|---|---|---|
| `TENSOR-DEVICE-DENSE-014` | One finite typed device-callable dense region through public CUDA-JS SPEC-0028 | Maximal connected matmul-anchored region and exact external-boundary discovery | Second-instance value; accepted Tensor callable ABI/resource/participation profile; exact provider/toolkit/license/architecture gates | Elementwise-only fusion, host scheduler, private CUDA-JS imports, mandatory cuBLASDx/CUTLASS |
| `TENSOR-BATCHED-GEMM-015` | Rank-3 contiguous f32 batched matmul in one prepared DAG | Separate SIMT semantic coverage, current backend support, proposed structural eligibility, and capacity batch/m/n/k facts | Public CUDA-JS batched/strided plan and prepared-node contract; workspace/access/failure/cleanup profile | Host loop over rank-2 calls, hidden packing, active batch extent in first profile |
| `TENSOR-PRECISION-GEMM-016` | f16, bf16, and f64 matmul profiles | Dtype/rank/accumulator structural classification without contract-readiness claims | Typed public CUDA-JS plans/nodes; compute/scale/output semantics; numeric equivalence and lifecycle evidence | Implicit reduced precision, Tensor Core claim, default promotion |
| `TENSOR-ARENA-018` | Bounded result-owned material reuse | Alias-closed deterministic first-fit candidate arena with explicit alignment and signed byte effect | Accepted suballocation/view ownership; range/rollback/child-view cleanup proof; high-water evidence | Hidden allocation, backend workspace, cross-run arena, cross-session memory, automatic spill |

## Integration order

The bounded [production LEGO readiness assessment](2026-08-26-tensor-production-lego-readiness-assessment.md) supersedes the provisional device-callable-first order. CUDA-JS SPEC-0028 supplies leaf-library composition but not the efficient collective participation primitives required by a useful dense region, so `TENSOR-DEVICE-DENSE-014` is deferred rather than implemented as a serial device function.

`TENSOR-FUSION-017` completed that replacement under accepted SPEC-0007: production resolution owns deterministic selection and exact generated substitution, the complete unfused path remains, focused/native evidence moved to production boundaries, and `04-fusion-frame.mjs` plus its experiment tests were deleted. `TENSOR-ARENA-018` is next over the resulting post-fusion material schedule. `TENSOR-BATCHED-GEMM-015` and `TENSOR-PRECISION-GEMM-016` remain blocked on a consumer-neutral CUDA-JS typed/strided-batched matmul family. Each remaining chunk must replace—not wrap—the corresponding experimental frame and delete its experiment file when its unique continuation value has moved into accepted source and tests.

Performance recommendation is a later evidence gate across completed profiles, not a sixth runtime policy manager. Multi-device tensor composition remains a separate future owner above one-device sessions and is not framed here. Convolution, sparse/ragged tensors, collectives, neural/training layers, package publication, and `the_restaurant` integration remain outside this frame.

## Acceptance and cleanup

No chunk is accepted, exported, natively qualified, performance-qualified, or eligible for a support claim in this task. Each frame now returns executable analysis records and remains useful only while it prevents reconstruction and keeps boundaries visible. A chunk is complete only after accepted authority, production implementation, risk-appropriate focused/full/native evidence, exact-head review, protected integration, and removal or archival of its superseded experiment scaffold.

The remaining analysis frames have focused portable tests for deterministic identity and immutability, valid and blocked matmul profiles, connected matmul-region boundaries, alias-closed allocation lifetimes, reuse, and alignment overhead. This evidence qualifies only their disposable analysis behavior; it does not satisfy native or performance gates and does not promote any remaining frame into production support.
