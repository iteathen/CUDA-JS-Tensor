# Status

```text
repository:                 public protected foundation established
package name:               cuda-js-tensor (reserved by intent, unpublished)
phase:                      specification-first pre-release
production implementation: not started
native qualification:       none
performance claims:         none
current target:             specify generic CUDA-JS prerequisites, then portable value/program foundation
```

The first product is a universal tensor library with convenience layered over complete contracts. Dense operations are the first bounded slice, not evidence that every future tensor or consumer is dense.

CUDA-JS prerequisites remain separate upstream work: public typed device views, typed device-callable library composition, prepared operation DAGs, and context-bound CUDA library adapters including a cuBLASLt profile. No private import or native workaround is authorized here.

`the_restaurant` integration is deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this repository does not currently depend on or modify it.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. GitHub Actions are limited to GitHub-owned actions with read-only workflow permissions, and repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting are enabled.
