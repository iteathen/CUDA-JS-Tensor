# Status

```text
repository:                 public protected foundation established
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.1 (publication-guarded)
phase:                      portable value implementation pre-release
production implementation: TensorSession/TensorSpec/Tensor portable/package slice
native qualification:       none
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.12 at 2da65ff2e428
current target:             immutable TensorProgram/TensorPlan dense semantics
```

The first product is a universal tensor library with convenience layered over complete contracts. Dense operations are the first bounded slice, not evidence that every future tensor or consumer is dense.

The four initial CUDA-JS prerequisites are integrated upstream: public typed device views, typed device-callable library composition, prepared operation DAGs, and context-bound CUDA library adapters including the first cuBLASLt profile. CUDA-JS-Tensor consumes the exact public package revision only; no private import or native workaround is authorized here.

`the_restaurant` integration is deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this repository does not currently depend on or modify it.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. GitHub Actions are limited to GitHub-owned actions with read-only workflow permissions, and repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting are enabled.
