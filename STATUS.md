# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      CUDA physical-boundary refactor before further semantic expansion
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt
device-callable execution:  item-parallel dense Tensor program through public CUDA-JS leaf libraries
native qualification:       prior exact installed-package Windows CUDA 13.3/compute_75/GTX 1660 Ti correctness/lifecycle cell passes; not requalified by the provider-ownership refactor
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.16 at 85d92d4a04385b0edbc7a19c2bce3d256642bf2f
current target:             CUDA-JS #180/#181 dispositions, then Tensor #45 physical-lowering migration
```

The accepted SPEC-0006 provider-boundary correction removes Tensor ownership of CUDA-JS provider lifetime and provider facts. Each resolved accelerated backend owns one ordinary public cuBLASLt borrower and its own plans; CUDA-JS owns the shared underlying provider resource. Tensor preference fallback consumes the public lower `unsupported` category, selected workspace alignment comes from the public provider descriptor, and Tensor retains its mathematical eligibility, backend-selection policy, workspace budgets, binding pressure and semantic fallback meaning. The exact dependency is the integrated CUDA-JS provider/capability revision above. Existing native numerical/provider evidence predates this exact lower revision and is not silently promoted to a new native compatibility claim.

The immediate architecture gate is now the lower preparation/launch disposition: CUDA-JS #180 determines whether the existing compiler/module/function/prepared-DAG bricks are already the correct composition seam, and #181 determines whether any additive logical-work resolver is justified while preserving explicit expert geometry. Tensor #45 follows only after those decisions; it must remove copied lower validity/limit facts without moving Tensor mathematics, liveness, access, Device-JS generation or intentionally selected backend policy downward.

SPEC-0009 continues to provide the device-callable correctness profile required by downstream evaluators. `compileTensorDeviceProgram(...)` proves one static item axis, exact shared versus item-varying inputs, item-preserving dense operations, finite typed parameters, explicit item-major outputs and dtype-partitioned per-item workspace. It returns one copied CUDA-JS SPEC-0028 leaf library. A caller-owned Device-JS participant invokes one item; many items can run in parallel without Tensor owning a scheduler, queue, launch, request incarnation, scatter or publication protocol.

The recorded installed-package native fixture passes through public exports on Node 26.7.0 and the earlier recorded Windows CUDA 13.3/compute_75/GTX 1660 Ti pair. In addition to retained SIMT, cuBLASLt and fusion coverage, it compiles and imports a Tensor leaf library into an unrelated Device-JS kernel. Eight GPU threads invoke a four-item program: four valid items independently match expected matmul/bias/fixed-tree-reduction outputs, four excess indices return the no-write status, and terminal CUDA-JS accounting reaches zero live/orphaned resources. This is correctness/lifecycle evidence only and does not requalify the new provider-ownership pair.

CUDA-MCGS remains responsible for evaluator/model meaning, resident input/artifact allocation, ready-item selection, device-owned batching, request/result incarnations, scatter, publication, cancellation, cache/root validity, search policy and multi-GPU coordination. One Tensor session/device remains the v1 owner; independent sessions naturally support replicated per-GPU consumers. Cross-device tensor identity, P2P, collectives and sharding remain later contracts. The previously active CUDA-MCGS/Vector consumer-qualification lane remains valid parallel work, but it does not displace the current lower-boundary correction.

The complete host-planned SIMT path remains available. `simt`, `fusion: 'none'` and PTX library output remain documented safe defaults. Exact fusion, cuBLASLt and item-callable execution are independently removable Legos. The first callable profile is item-parallel, not block/warp-cooperative, and makes no speedup or Tensor Core claim. Cooperative intra-item participation, cuBLASDx/CUTLASS, workspace reuse and performance recommendation require separate exact evidence.

The existing result-owned arena issue #17 remains a useful host-planned optimization but is not the CUDA boundary-refactor critical path. Batched/precision cuBLASLt profiles remain upstream-blocked host-planned accelerators. The superseded device-callable analysis frame has been deleted after its useful boundary cases moved into accepted SPEC-0009 production tests.

`the_restaurant` integration remains deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this work does not modify that repository.

TENSOR-DEVICE-ITEM-014 integrated through protected PR #20 at `main@9ecc1d78bca989ec456c897dec215e82ce4cd311`; issue #19 is closed. Author-side review covered exact source head `6c2b3570f0f326cdceb30e6540435ff3e5df92a9`, whose tree is identical to the squash merge. The required PR check passed. CUDA-MCGS read-back is preserved on its research branch at `ade6e0bcbef4289e538f0ac02b5bfb9ee8568abc`; that is dependency documentation, not CUDA-MCGS production integration.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. Independent review is waived only under the project owner's sole-maintainer direction and is never represented as independent evidence.

Earlier protected integrations remain part of the repository record: TENSOR-CUBLASLT-013 entered through PR #10 at `main@8910309a0aff9b8da4fc281949068d8d1fcaa6ea`, and TENSOR-FUSION-017 entered through PR #16 at `main@821b1cbe1c32835559cb669248771310718b2f05`. GitHub Actions remain limited to GitHub-owned actions with read-only workflow permissions; repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting remain enabled.
