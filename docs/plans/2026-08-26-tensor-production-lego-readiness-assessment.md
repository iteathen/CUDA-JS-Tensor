# Tensor production LEGO readiness assessment

**Status:** Bounded production-readiness assessment; child specifications and implementations remain pending

**Exact CUDA-JS-Tensor input:** `main@cf975f906d9c3ede8b94275bb2a6e1447634dc9f`

**Exact CUDA-JS input:** `cuda-js@0.1.0-alpha.15` from protected `main@af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d`

## Question and disposition

The assessment asks whether the five executable analysis frames can now become production Legos before CUDA-MCGS consumes them.

They cannot be promoted as one batch. The frames cross three different owners:

1. Tensor-owned graph realization: exact elementwise fusion and result-owned material arenas.
2. CUDA-JS-owned native library mechanisms: typed and strided-batched cuBLASLt plans/prepared nodes.
3. CUDA-JS-owned device-language mechanisms plus Tensor-owned participation semantics: efficient device-callable dense regions.

`TENSOR-FUSION-017` and `TENSOR-ARENA-018` are dependency-ready for child specification work using current public CUDA-JS. Fusion is first because it changes the material schedule consumed by arena planning. The other three profiles remain blocked on explicitly bounded CUDA-JS capabilities; CUDA-JS-Tensor must not substitute host loops, serial device functions, private imports, generated CUDA escape paths, or provider-specific state.

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

### `TENSOR-FUSION-017` — select next

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

### `TENSOR-DEVICE-DENSE-014` — upstream and participation blocked

CUDA-JS SPEC-0028 solves typed leaf-library compilation/import/linking. It does not provide the efficient collective execution model required by a dense region. CUDA-JS's shared-memory, local-array, multidimensional-index, block/warp synchronization and related trusted parallel primitives remain proposal-only. A serial device function would be functionally possible for tiny regions but would not be an honest reusable acceleration Lego for CUDA-MCGS.

Before Tensor implementation, a real caller profile must select the smallest useful participation class—likely one fixed block or warp group—with exact participant count, barrier legality, shared/global workspace, output ownership and caller obligations. CUDA-JS should then accept only the generic parallel helpers required by that profile. Tensor may generate a fixed-shape typed leaf library through SPEC-0028 after those gates; cuBLASDx or CUTLASS remains an optional separately qualified provider, not the contract.

## Dependency and implementation order

```text
SPEC-0007 exact elementwise fusion
  -> TENSOR-FUSION-017 production implementation
  -> SPEC-0008 result-owned material arena
  -> TENSOR-ARENA-018 production implementation

CUDA-JS typed/strided-batched matmul family
  -> TENSOR-BATCHED-GEMM-015 and TENSOR-PRECISION-GEMM-016 child specs
  -> independently removable production profiles

selected real device caller + minimum CUDA-JS trusted parallel helpers
  -> Tensor device-callable participation/ABI child spec
  -> TENSOR-DEVICE-DENSE-014 production implementation
```

Fusion and arena are the immediate integration spine. The two CUDA-JS dependency lanes may proceed independently but do not block them. Device-callable dense stays later because choosing participation without the caller shape would hard-code a speculative scheduler into a universal library.

## Acceptance, falsifiers and cleanup

Each selected child requires an accepted specification before production mutation, focused semantic/lifecycle tests, repository verification, exact-head author review and protected integration. Native and performance evidence narrow support/recommendation claims but do not replace the code boundary.

Stop a child if it requires private CUDA-JS imports, maintained CUDA/PTX, a second scheduler, host advancement between prepared nodes, hidden allocation/packing, provider objects in public identity, changed Tensor mathematics, cross-session ownership or consumer-specific policy.

When a production child integrates, remove its superseded frame source and unique tests in the same coherent change after preserving any still-useful adversarial cases under the owning component. Retain the three blocked frames only while they remain the shortest accurate dependency record; replace them with accepted upstream issue/spec links when those owners exist.
