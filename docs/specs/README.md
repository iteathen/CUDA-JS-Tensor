# Specifications

- [`SPEC-0000-tensor-contract-map.md`](SPEC-0000-tensor-contract-map.md) — normative owner and dependency map.
- [`SPEC-0001-tensor-session-spec-and-value-model.md`](SPEC-0001-tensor-session-spec-and-value-model.md) — session, dtype/shape/layout and tensor capability.
- [`SPEC-0002-tensor-program-plan-and-dense-operations.md`](SPEC-0002-tensor-program-plan-and-dense-operations.md) — immutable program, finite plan and first dense operations.
- [`SPEC-0003-accelerated-dense-backend-profiles.md`](SPEC-0003-accelerated-dense-backend-profiles.md) — complete SIMT baseline and optional qualified accelerators.
- [`SPEC-0004-first-dense-program-semantics.md`](SPEC-0004-first-dense-program-semantics.md) — exact first operation catalog, immutable DAG semantics and static plan boundary.
- [`SPEC-0005-resolved-simt-execution.md`](SPEC-0005-resolved-simt-execution.md) — session-bound resolved plan, complete generated SIMT lowering, prepared execution and result lifecycle.
- [`SPEC-0005-physical-boundary-addendum.md`](SPEC-0005-physical-boundary-addendum.md) — accepted physical-boundary correction consuming public CUDA-JS prepared/Device-JS compatibility limits while preserving Tensor-owned block geometry, logical-work policy, artifact-family selection and public lower LEGO composition.
- [`SPEC-0006-host-planned-cublaslt-matmul.md`](SPEC-0006-host-planned-cublaslt-matmul.md) — optional resolved rank-2 contiguous f32 cuBLASLt nodes composed with the complete SIMT fallback in one prepared DAG.
- [`SPEC-0006-provider-boundary-addendum.md`](SPEC-0006-provider-boundary-addendum.md) — accepted prerelease correction consuming CUDA-JS-owned cuBLASLt borrowers, public `unsupported` fallback classification and public workspace alignment instead of Tensor-local provider lifecycle/facts.
- [`SPEC-0007-exact-elementwise-fusion.md`](SPEC-0007-exact-elementwise-fusion.md) — optional exact cast/unary/binary kernel fusion with unobservable-intermediate material deletion and complete unfused fallback.
- `SPEC-0008` remains reserved for the planned result-owned material arena tracked by TENSOR-ARENA-018 / issue #17; it has no accepted implementation authority yet.
- [`SPEC-0009-item-parallel-device-callable-tensor-program.md`](SPEC-0009-item-parallel-device-callable-tensor-program.md) — one caller-owned device participant per independent item, with a finite typed callable ABI, item-isolated workspace and public CUDA-JS leaf-library composition.
- [`SPEC-0010-erf-static-gather-and-concat.md`](SPEC-0010-erf-static-gather-and-concat.md) — accepted consumer-backed mathematical and ordinary-SIMT authority for `f32`/`f64` `erf`, bounded static indexed gather and finite ordered concat; device-callable `gather`/`concat` remain separately SPEC-0009-gated and `erf` realization remains dependent on the public CUDA-JS helper implementation.

Accepted status authorizes only the bounded contracts written in accepted specifications. CUDA-JS-Tensor semantic, lifecycle, native, mechanism-dependency and performance evidence remains independently gated per component.
