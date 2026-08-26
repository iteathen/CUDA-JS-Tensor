# Specifications

- [`SPEC-0000-tensor-contract-map.md`](SPEC-0000-tensor-contract-map.md) — normative owner and dependency map.
- [`SPEC-0001-tensor-session-spec-and-value-model.md`](SPEC-0001-tensor-session-spec-and-value-model.md) — session, dtype/shape/layout and tensor capability.
- [`SPEC-0002-tensor-program-plan-and-dense-operations.md`](SPEC-0002-tensor-program-plan-and-dense-operations.md) — immutable program, finite plan and first dense operations.
- [`SPEC-0003-accelerated-dense-backend-profiles.md`](SPEC-0003-accelerated-dense-backend-profiles.md) — complete SIMT baseline and optional qualified accelerators.
- [`SPEC-0004-first-dense-program-semantics.md`](SPEC-0004-first-dense-program-semantics.md) — exact first operation catalog, immutable DAG semantics and static plan boundary.
- [`SPEC-0005-resolved-simt-execution.md`](SPEC-0005-resolved-simt-execution.md) — session-bound resolved plan, complete generated SIMT lowering, prepared execution and result lifecycle.
- [`SPEC-0006-host-planned-cublaslt-matmul.md`](SPEC-0006-host-planned-cublaslt-matmul.md) — optional resolved rank-2 contiguous f32 cuBLASLt nodes composed with the complete SIMT fallback in one prepared DAG.
- [`SPEC-0007-exact-elementwise-fusion.md`](SPEC-0007-exact-elementwise-fusion.md) — optional exact cast/unary/binary kernel fusion with unobservable-intermediate material deletion and complete unfused fallback.

Accepted status authorizes only the bounded contracts written here. The required public CUDA-JS mechanisms are available in the exact recorded compatibility revision; CUDA-JS-Tensor semantic, lifecycle, native, and performance evidence remains independently gated per component.
