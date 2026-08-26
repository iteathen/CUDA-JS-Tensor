# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      first device-callable correctness profile implemented
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt
device-callable execution:  item-parallel dense Tensor program through public CUDA-JS leaf libraries
native qualification:       exact installed-package Windows CUDA 13.3/compute_75/GTX 1660 Ti candidate passes
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.16 at 4971302cfb48
current target:             protected integration and CUDA-MCGS readiness reconciliation
```

SPEC-0009 closes the host-progress gap that previously prevented CUDA-JS-Tensor from participating in CUDA-MCGS active-search evaluation. `compileTensorDeviceProgram(...)` proves one static item axis, exact shared versus item-varying inputs, item-preserving dense operations, finite typed parameters, explicit item-major outputs and dtype-partitioned per-item workspace. It returns one copied CUDA-JS SPEC-0028 leaf library. A caller-owned Device-JS participant invokes one item; many items can run in parallel without Tensor owning a scheduler, queue, launch, request incarnation, scatter or publication protocol.

The exact installed-package native fixture passes through public exports on Node 26.7.0 and the recorded Windows CUDA 13.3/compute_75/GTX 1660 Ti profile. In addition to the retained SIMT, cuBLASLt and fusion coverage, it compiles and imports a Tensor leaf library into an unrelated Device-JS kernel. Eight GPU threads invoke a four-item program: four valid items independently match expected matmul/bias/fixed-tree-reduction outputs, four excess indices return the no-write status, and terminal CUDA-JS accounting reaches zero live/orphaned resources. This is correctness/lifecycle evidence only.

CUDA-MCGS remains responsible for evaluator/model meaning, resident input/artifact allocation, ready-item selection, device-owned batching, request/result incarnations, scatter, publication, cancellation, cache/root validity, search policy and multi-GPU coordination. One Tensor session/device remains the v1 owner; independent sessions naturally support replicated per-GPU consumers. Cross-device tensor identity, P2P, collectives and sharding remain later contracts.

The complete host-planned SIMT path remains available. `simt`, `fusion: 'none'` and PTX library output remain documented safe defaults. Exact fusion, cuBLASLt and item-callable execution are independently removable Legos. The first callable profile is item-parallel, not block/warp-cooperative, and makes no speedup or Tensor Core claim. Cooperative intra-item participation, cuBLASDx/CUTLASS, workspace reuse and performance recommendation require separate exact evidence.

The existing result-owned arena issue #17 remains a useful host-planned optimization but is no longer the CUDA-MCGS readiness critical path. Batched/precision cuBLASLt profiles remain upstream-blocked host-planned accelerators. The superseded device-callable analysis frame has been deleted after its useful boundary cases moved into accepted SPEC-0009 production tests.

`the_restaurant` integration remains deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this work does not modify that repository.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. Independent review is waived only under the project owner's sole-maintainer direction and is never represented as independent evidence.

Earlier protected integrations remain part of the repository record: TENSOR-CUBLASLT-013 entered through PR #10 at `main@8910309a0aff9b8da4fc281949068d8d1fcaa6ea`, and TENSOR-FUSION-017 entered through PR #16 at `main@821b1cbe1c32835559cb669248771310718b2f05`. GitHub Actions remain limited to GitHub-owned actions with read-only workflow permissions; repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting remain enabled.
