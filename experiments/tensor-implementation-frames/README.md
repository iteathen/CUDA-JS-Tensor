# Remaining tensor implementation frames

**Status:** Disposable analysis scaffolding; production support still separate.

This experiment turns the known remaining tensor work into small executable analysis modules. Each module consumes the accepted public `TensorPlan`, performs deterministic analysis, and returns an immutable analysis record. Nothing here is exported by `cuda-js-tensor` or consulted by the production resolver.

The files are numbered only for quick scanning; accepted specs and dependency readiness decide execution order:

1. `01-batched-matmul-frame.mjs` separates complete SIMT semantic coverage, current rank-2-only cuBLASLt support, and structural eligibility for a future nonempty rank-3 f32 profile.
2. `02-precision-matmul-frame.mjs` separates f16, bf16, and f64 semantic coverage and structural facts from the missing typed provider contract.
3. `03-device-callable-dense-frame.mjs` finds maximal connected material regions anchored by at least one matmul without pretending the callable ABI/resource profile is accepted.
4. `05-arena-reuse-frame.mjs` closes material lifetimes over complete alias classes before first-fit assignment and reports alignment, signed byte effect, and reuse separately.

Each remaining frame owns only its analysis record and consumes only the public immutable `TensorPlan` contract. Batched/precision classification does not own SIMT or CUDA-JS provider behavior; device-callable discovery does not absorb elementwise-only fusion; and arena analysis does not allocate or own runtime memory. Deleting a frame leaves the accepted program, plan, resolver, complete SIMT implementation and optional production fusion unchanged. The former `04-fusion-frame.mjs` was deleted after SPEC-0007 moved its bounded continuation value into the production execution owner and focused tests.

Performance-policy promotion is still not represented in this folder as executable runtime behavior; evidence, not a second manager, must decide that next. Multi-device composition is also omitted: one tensor session remains one device, and a future multi-session owner requires separate accepted semantics. `the_restaurant`, neural layers, training, sparse tensors, convolution, collectives, and package publication remain outside this experiment.

Focused portable tests cover contract rejection, deterministic immutable records, batched and precision matmul classification, matmul-anchored connected regions, alias-closed arena lifetimes, and alignment overhead. These tests qualify the remaining analysis behavior only; a production chunk must still begin with its stated blockers, accept the owning specification and public CUDA-JS dependency, add production-boundary evidence, then pass the full repository/native/performance gates appropriate to its claim.
