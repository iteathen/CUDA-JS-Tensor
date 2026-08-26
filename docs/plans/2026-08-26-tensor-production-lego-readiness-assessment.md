# Tensor production LEGO readiness assessment

**Status:** Bounded historical assessment; device-callable disposition superseded by the later CUDA-MCGS caller assessment, other pending children remain current

**Exact CUDA-JS-Tensor input:** `main@cf975f906d9c3ede8b94275bb2a6e1447634dc9f`

**Exact CUDA-JS input at assessment:** `cuda-js@0.1.0-alpha.15` from protected `main@af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d`

**Later dependency update:** SPEC-0009 consumes selected-target-correct `cuda-js@0.1.0-alpha.16` from protected `main@4971302cfb48431c0843126a59d5884d84a81641`

## Question and disposition

The assessment asks whether the five executable analysis frames can now become production Legos before CUDA-MCGS consumes them.

They cannot be promoted as one batch. The frames cross three different owners:

1. Tensor-owned graph realization: exact elementwise fusion and result-owned material arenas.
2. CUDA-JS-owned native library mechanisms: typed and strided-batched cuBLASLt plans/prepared nodes.
3. CUDA-JS-owned device-language mechanisms plus Tensor-owned participation semantics: item-parallel correctness now implemented; cooperative intra-item acceleration remains later.

`TENSOR-FUSION-017` completed under accepted SPEC-0007. `TENSOR-ARENA-018` is dependency-ready over the accepted realized material schedule. Batched and precision provider profiles remain blocked on bounded CUDA-JS capabilities. The later [CUDA-MCGS readiness assessment](2026-08-26-cuda-mcgs-readiness-assessment-and-plan.md) supplies a real item-batch caller and supersedes this document's collective-only device-callable assumption: SPEC-0009 now implements one caller participant per independent item without a host loop or guessed scheduler.

## Strongest adversarial case

The dangerous path is to interpret “real executable frame” as “nearly production code.” Each frame currently owns deterministic classification only. Promoting it directly would hide at least one of mathematical equivalence, physical storage ownership, provider admission, device participation, workspace, failure attribution, or cleanup behind a helper that looks reusable but exports the hard problem to its callers.

The selected path keeps one owner per invariant and retains complete deletion:

- deleting fusion restores the exact unfused SIMT schedule;
- deleting arena reuse restores one distinct allocation per material;
- deleting any accelerator restores complete SIMT matmul;
- deleting device-callable execution leaves host-planned tensor execution complete;
- deleting CUDA-MCGS or `the_restaurant` leaves every Tensor contract product-neutral.

## LEGO ownership tests

| Lego | Owned invariant | Same-class uses | Explicit exclusions | First-consumer deletion |
|---|---|---|---|---|
| Exact elementwise fusion | A selected unobservable single-consumer elementwise region may replace its kernels and intermediate materials without changing declared rounding, outputs, access, failure, bounds or cleanup. | Generated SIMT programs for unrelated tensor consumers and shapes. | General graph optimization, reduction/matmul fusion, approximate math, hidden runtime selection. | Unfused generated kernels remain complete. |
| Result-owned material arena | Non-overlapping alias-closed material lifetimes may occupy bounded aligned subranges of one per-run allocation while logical tensor alias semantics remain unchanged. | Any finite TensorPlan result graph, independent of evaluator/search meaning. | Backend workspace, cross-run pools, cross-session memory, automatic spill, allocator policy in CUDA-JS. | Distinct per-material allocation remains complete. |
| Typed/strided-batched matmul adapter | CUDA-JS admits one closed provider-neutral typed matrix-plan family with explicit batch/stride/compute/scale/output/workspace/access/lifecycle facts. | Tensor, signal-processing and direct CUDA-JS numerical consumers. | Tensor shapes/broadcast policy, NN semantics, hidden packing, arbitrary provider calls. | Current rank-2 f32 plan and complete Tensor SIMT path remain intact. |
| Device-callable dense region | One fixed Tensor region exposes a finite typed callable ABI, exact participant set, workspace, output publication and failure contract. | A persistent search evaluator, a fused simulation kernel, or another compatible Device-JS program. | Host scheduler, dynamic allocation, universal cuBLASDx/CUTLASS requirement, serial fallback presented as acceleration. | Ordinary resolved Tensor plans remain complete. |

## Per-frame disposition

### `TENSOR-FUSION-017` — implemented under SPEC-0007

Current CUDA-JS already provides deterministic Device-JS generation with `fmad: false`, dense scalar semantics, finite prepared kernels and exact access records. No new upstream mechanism is required for a first exact profile.

Implementation is tracked by [CUDA-JS-Tensor issue #14](https://github.com/iteathen/CUDA-JS-Tensor/issues/14).

The child specification must restrict v1 to materialized `cast`/`unary`/`binary` chains with one downstream consumer per internal value and no public/internal observable output. It must preserve operation-by-operation dtype rounding, NaN/signed-zero behavior, external access ranges, active extents, launch bounds and aggregate failure truth. Selection happens once during resolution and is identity-affecting. `fusion: "none" | "exact-elementwise"` should normalize through the existing resolved-plan options; `none` remains the default until representative total-cost evidence supports a change.

Implementation replaces the selected chain with one generated kernel and removes only its unobservable intermediate material allocations. It does not create another scheduler or backend manager.

### `TENSOR-ARENA-018` — select after fusion

CUDA-JS SPEC-0021 already provides allocation-owned, aligned typed child views with exact ranges, access roles, leases and child-before-parent cleanup. That is sufficient mechanism for a Tensor-owned per-run arena; CUDA-JS does not need tensor liveness or arena policy.

The child specification must consume the final post-fusion material schedule. It must close liveness over complete alias classes, retain every observable output until result close, preserve prepared-DAG ordering for reused ranges, expose exact arena bytes/alignment/high-water facts and make rollback close child views before the parent allocation. `materialStorage: "distinct" | "arena"` is identity-affecting; `distinct` remains the default until measured evidence supports a change.

Backend reduction/cuBLASLt workspace stays separately owned and is not placed in this first arena. Cross-run persistence, pooling and multi-session/device reuse remain excluded.

### `TENSOR-BATCHED-GEMM-015` — upstream blocked

Rank-3 f32 batched matmul mathematics and SIMT execution are already complete. CUDA-JS SPEC-0029/0031 exposes only one rank-2 contiguous f32 plan/node. Repeating rank-2 plans or launches is not the intended batched Lego and would create bounded-by-node-count overhead rather than one reusable provider primitive.

CUDA-JS must first accept a consumer-neutral strided-batched matmul plan/prepared-node child with explicit batch count, zero-stride broadcast admission, matrix/batch strides, transpose, workspace, access ranges, provider identity, failure and cleanup. Tensor then owns rank-3 shape/broadcast mapping, preference/strict fallback and resolved identity. The upstream mechanism is tracked by [CUDA-JS issue #75](https://github.com/iteathen/CUDA-JS/issues/75).

### `TENSOR-PRECISION-GEMM-016` — upstream blocked

The complete SIMT path already supports f16, bf16 and f64 under exact Tensor semantics, and CUDA-JS SPEC-0030 supplies their Device-JS scalar behavior. The CUDA-JS cuBLASLt port remains fixed to f32 input, compute and output.

CUDA-JS must first accept closed typed plan/prepared-node profiles that distinguish input, compute/accumulator, scale and output dtypes without adding Tensor promotion policy. CUDA-JS-Tensor then binds only combinations that exactly preserve an accepted Tensor precision/tolerance profile. Reduced precision, tensor-core use and default promotion are never inferred from dtype names.

The batched and precision gaps should share one CUDA-JS matmul-family owner but enter as independently bounded child profiles so either can be deleted without removing the other.

### `TENSOR-DEVICE-ITEM-014` — superseded disposition; implemented under SPEC-0009

This assessment originally assumed the first useful callable had to accelerate one item cooperatively. CUDA-MCGS later supplied a materially different natural participation class: many independent evaluator items are ready together, and one caller-owned participant may execute each item. That is batch-parallel device execution, not one device function serializing the whole batch.

SPEC-0009 therefore uses existing public CUDA-JS SPEC-0028/0030, proves item independence, exposes finite item-major outputs/workspace and leaves scheduling/publication to the caller. Block/warp-cooperative intra-item execution, shared-memory helpers, cuBLASDx and CUTLASS remain optional later acceleration profiles rather than correctness prerequisites.

## Dependency and implementation order

```text
SPEC-0007 exact elementwise fusion
  -> TENSOR-FUSION-017 production implementation
  -> SPEC-0008 result-owned material arena
  -> TENSOR-ARENA-018 production implementation

CUDA-JS typed/strided-batched matmul family
  -> TENSOR-BATCHED-GEMM-015 and TENSOR-PRECISION-GEMM-016 child specs
  -> independently removable production profiles

selected real independent-item caller + CUDA-JS SPEC-0028/0030
  -> SPEC-0009 item-parallel callable profile
  -> TENSOR-DEVICE-ITEM-014 production implementation

selected cooperative caller + minimum CUDA-JS trusted parallel helpers
  -> optional block/warp-cooperative device-callable acceleration profile
```

Fusion and the item-parallel callable are completed integration-spine children; arena remains the next independent host-planned child. The two provider dependency lanes may proceed independently but do not block arena. Cooperative intra-item acceleration stays later because choosing its participation shape without evidence would hard-code a speculative physical scheduler into a universal library.

## Acceptance, falsifiers and cleanup

Each selected child requires an accepted specification before production mutation, focused semantic/lifecycle tests, repository verification, exact-head author review and protected integration. Native and performance evidence narrow support/recommendation claims but do not replace the code boundary.

Stop a child if it requires private CUDA-JS imports, maintained CUDA/PTX, a second scheduler, host advancement between prepared nodes, hidden allocation/packing, provider objects in public identity, changed Tensor mathematics, cross-session ownership or consumer-specific policy.

When a production child integrates, remove its superseded frame source and unique tests in the same coherent change after preserving any still-useful adversarial cases under the owning component. Retain the three blocked frames only while they remain the shortest accurate dependency record; replace them with accepted upstream issue/spec links when those owners exist.
