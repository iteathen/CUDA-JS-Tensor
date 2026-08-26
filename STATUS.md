# Status

```text
repository:                 public protected foundation established
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.4 (publication-guarded)
phase:                      optional host-planned cuBLASLt implementation/evidence active
production implementation: value + immutable program/static plan + complete SIMT + bounded optional cuBLASLt
native qualification:       exact installed-package Windows CUDA 13.3/compute_75/GTX 1660 Ti alpha.4 candidate passes
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.15 at af29b95e0707
current target:             TENSOR-CUBLASLT-013 exact-head review and protected integration under issue #8
```

The first product is a universal tensor library with convenience layered over complete contracts. Dense operations are the first bounded slice, not evidence that every future tensor or consumer is dense.

The required CUDA-JS prerequisites are integrated upstream: public typed device views, typed device-callable library composition, prepared operation DAGs, context-bound CUDA library adapters and SPEC-0031 mixed kernel/cuBLASLt prepared nodes. CUDA-JS-Tensor consumes the exact public package revision only; no private import or native workaround is authorized here.

Static-program assessment exposed one additional generic upstream prerequisite before the complete SIMT baseline. CUDA-JS SPEC-0030 now supplies f64/f16/bf16 pointer/local arithmetic, exact casts and special-value math; alpha.14 also corrects native prepared-DAG identity projection. SPEC-0005 therefore owns the Tensor-only resolution, lowering, binding, result and cleanup semantics without a private/native workaround.

`the_restaurant` integration is deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this repository does not currently depend on or modify it.

The alpha.4 candidate exact installed-package native fixture passes through public exports on Node 26.7.0 and the recorded Windows CUDA 13.3/compute_75/GTX 1660 Ti profile. It retains the complete 10-kernel SIMT replay, adds a three-node kernel-to-cuBLASLt-to-kernel plan replayed twice, independently checks strict double-transpose cuBLASLt output `[58, 64, 139, 154]`, records exact cuBLASLt 13.5.1 provider identity and closes with zero live/orphaned CUDA-JS resources. It remains correctness/lifecycle evidence only—not performance, tensor-core, broader device/platform, multi-GPU or production-stability evidence.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. GitHub Actions are limited to GitHub-owned actions with read-only workflow permissions, and repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting are enabled.
