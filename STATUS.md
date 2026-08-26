# Status

```text
repository:                 public protected foundation established
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.5 (publication-guarded)
phase:                      exact fusion production LEGO implemented; result-owned arena child next
production implementation: value + immutable program/static plan + complete SIMT + optional exact fusion + bounded optional cuBLASLt
native qualification:       exact installed-package Windows CUDA 13.3/compute_75/GTX 1660 Ti alpha.5 candidate passes
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.15 at af29b95e0707
current target:             SPEC-0008 result-owned material arena, then TENSOR-ARENA-018
```

The first product is a universal tensor library with convenience layered over complete contracts. Dense operations are the first bounded slice, not evidence that every future tensor or consumer is dense.

The CUDA-JS prerequisites for the implemented surface and the next Tensor-owned arena slice are integrated upstream: public typed device views, typed device-callable library composition, prepared operation DAGs, context-bound CUDA library adapters and SPEC-0031 mixed kernel/cuBLASLt prepared nodes. CUDA-JS-Tensor consumes the exact public package revision only; no private import or native workaround is authorized here. Future typed/strided-batched cuBLASLt profiles and efficient collective device-callable dense execution still require bounded CUDA-JS children.

The [production LEGO readiness assessment](docs/plans/2026-08-26-tensor-production-lego-readiness-assessment.md) classifies the retained analysis frames by owner. Exact elementwise fusion is now implemented under accepted SPEC-0007 and its superseded experiment frame is deleted. A result-owned per-run material arena is next because it consumes the realized post-fusion allocation schedule. Batched/precision acceleration remains upstream-blocked, and device-callable dense execution is deferred until a real caller selects a useful participation class plus the minimum generic CUDA-JS parallel helpers.

Static-program assessment exposed one additional generic upstream prerequisite before the complete SIMT baseline. CUDA-JS SPEC-0030 now supplies f64/f16/bf16 pointer/local arithmetic, exact casts and special-value math; alpha.14 also corrects native prepared-DAG identity projection. SPEC-0005 therefore owns the Tensor-only resolution, lowering, binding, result and cleanup semantics without a private/native workaround.

`the_restaurant` integration is deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this repository does not currently depend on or modify it.

The alpha.5 exact installed-package native fixture passes through public exports on Node 26.7.0 and the recorded Windows CUDA 13.3/compute_75/GTX 1660 Ti profile. It retains the complete SIMT and mixed cuBLASLt replays, independently compares fused and unfused three-operation output against `[-2, -3, -4, -5]`, verifies three semantic kernels become one fused kernel, checks a fused f32-to-f64 dtype transition plus NaN/signed-zero/infinity behavior, records exact provider/compiler identity and closes with zero live/orphaned CUDA-JS resources. It remains correctness/lifecycle evidence only—not performance, tensor-core, broader device/platform, multi-GPU or production-stability evidence.

TENSOR-CUBLASLT-013 integrated through protected PR #10 at `main@8910309a0aff9b8da4fc281949068d8d1fcaa6ea`; issue #8 is closed. Author-side review covered the complete exact head. Independent review was waived under the project owner's sole-maintainer direction and is not represented as independent evidence.

TENSOR-FUSION-017 integrated through protected PR #16 at `main@821b1cbe1c32835559cb669248771310718b2f05`; issue #14 is closed. Author-side review covered exact source head `d75b48435e1570d1dfb4a4b8202d49f15d2f9a7d`, whose tree is identical to the squash merge. Independent review was waived under the project owner's sole-maintainer direction and is not represented as independent evidence. No performance qualification or default promotion is claimed.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. GitHub Actions are limited to GitHub-owned actions with read-only workflow permissions, and repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting are enabled.
