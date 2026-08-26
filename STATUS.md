# Status

```text
repository:                 public protected foundation established
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.3 (publication-guarded)
phase:                      resolved generated-SIMT implementation pre-release
production implementation: value + immutable program/static plan + resolved SIMT
native qualification:       exact installed-package Windows CUDA 13.3/compute_75/GTX 1660 Ti profile passes
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.14 at fb27296cffd7
current target:             TENSOR-CUBLASLT-013 assessment/specification under issue #8
```

The first product is a universal tensor library with convenience layered over complete contracts. Dense operations are the first bounded slice, not evidence that every future tensor or consumer is dense.

The four initial CUDA-JS prerequisites are integrated upstream: public typed device views, typed device-callable library composition, prepared operation DAGs, and context-bound CUDA library adapters including the first cuBLASLt profile. CUDA-JS-Tensor consumes the exact public package revision only; no private import or native workaround is authorized here.

Static-program assessment exposed one additional generic upstream prerequisite before the complete SIMT baseline. CUDA-JS SPEC-0030 now supplies f64/f16/bf16 pointer/local arithmetic, exact casts and special-value math; alpha.14 also corrects native prepared-DAG identity projection. SPEC-0005 therefore owns the Tensor-only resolution, lowering, binding, result and cleanup semantics without a private/native workaround.

`the_restaurant` integration is deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this repository does not currently depend on or modify it.

The exact installed-package native SPEC-0005 fixture passes through public exports on Node 26.7.0 and the recorded Windows CUDA 13.3/compute_75/GTX 1660 Ti profile. It covers deterministic generated view/material operations, a 10-kernel semantic prepared DAG replayed twice, f64-to-i32 NaN/infinity saturation, f16/bf16 arithmetic, adversarial fixed-tree reduction, matmul, copied transfers and zero live/orphaned CUDA-JS resources at terminal close. It is correctness/lifecycle evidence only—not performance, tensor-core, broader device/platform, accelerator or production-stability evidence.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. GitHub Actions are limited to GitHub-owned actions with read-only workflow permissions, and repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting are enabled.
