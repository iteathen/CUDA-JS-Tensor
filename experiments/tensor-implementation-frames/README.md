# Remaining tensor implementation frames

**Status:** Disposable incomplete implementation scaffolding; not a public contract or support claim

This experiment turns the known remaining tensor work into small code-shaped chunks. Each module consumes the accepted public `TensorPlan`, performs useful deterministic analysis, and returns an immutable record with `executable: false` and `supportClaim: false`. Nothing here is exported by `cuda-js-tensor` or consulted by the production resolver.

The files are numbered only for quick scanning; the accepted plan and upstream dependency readiness decide execution order:

1. `01-batched-matmul-frame.mjs` isolates rank-3 f32 cuBLASLt work from the implemented rank-2 profile.
2. `02-precision-matmul-frame.mjs` isolates f16, bf16, and f64 provider/numeric work from batch semantics.
3. `03-device-callable-dense-frame.mjs` finds compact dense regions without pretending the callable ABI/resource profile is accepted.
4. `04-fusion-frame.mjs` finds conservative elementwise chains without changing SPEC-0004 rounding or observability.
5. `05-arena-reuse-frame.mjs` attaches lifecycle questions to a deterministic candidate reuse layout without enabling it.

Performance-policy promotion is not represented as executable code because evidence, rather than another runtime manager, must decide it. Multi-device composition is also omitted: one tensor session remains one device, and a future multi-session owner requires separate accepted semantics. `the_restaurant`, neural layers, training, sparse tensors, convolution, collectives, and package publication remain outside this experiment.

No tests or support qualification accompany this owner-requested quick frame. A future chunk must begin with its stated blockers, accept the owning specification and public CUDA-JS dependency, add focused tests, then pass the full repository/native/performance gates appropriate to its claim.

