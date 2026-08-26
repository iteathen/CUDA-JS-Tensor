# Remaining tensor implementation frames

**Status:** Disposable analysis scaffolding; production support still separate.

This experiment turns the known remaining tensor work into small executable analysis modules. Each module consumes the accepted public `TensorPlan`, performs deterministic analysis, and returns an immutable analysis record. Nothing here is exported by `cuda-js-tensor` or consulted by the production resolver.

The files are numbered only for quick scanning; accepted specs and dependency readiness decide execution order:

1. `01-batched-matmul-frame.mjs` isolates rank-3 f32 cuBLASLt work from the implemented rank-2 profile.
2. `02-precision-matmul-frame.mjs` isolates f16, bf16, and f64 provider/numeric work from batch semantics.
3. `03-device-callable-dense-frame.mjs` finds compact dense regions without pretending the callable ABI/resource profile is accepted.
4. `04-fusion-frame.mjs` returns conservative single-consumer cast/unary/binary chains as fusion candidates.
5. `05-arena-reuse-frame.mjs` returns a first-fit deterministic arena assignment for material lifetime classes.

Performance-policy promotion is still not represented in this folder as executable runtime behavior; evidence, not a second manager, must decide that next. Multi-device composition is also omitted: one tensor session remains one device, and a future multi-session owner requires separate accepted semantics. `the_restaurant`, neural layers, training, sparse tensors, convolution, collectives, and package publication remain outside this experiment.

Focused portable tests cover contract rejection, deterministic immutable records, batched and precision matmul classification, dense-region and fusion boundaries, and conservative arena reuse. These tests qualify the analysis behavior only; a production chunk must still begin with its stated blockers, accept the owning specification and public CUDA-JS dependency, add production-boundary evidence, then pass the full repository/native/performance gates appropriate to its claim.
