# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      CUDA boundary refactor complete; downstream adapter/package audit handoff
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt
device-callable execution:  item-parallel dense Tensor program through public CUDA-JS leaf libraries
native qualification:       prior exact installed-package Windows CUDA 13.3/compute_75/GTX 1660 Ti correctness/lifecycle evidence remains historical; alpha.17 pair not natively requalified
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.17 at bc2700f2e5c654567c2e17bf8d67b882351b8681
current target:             CUDA-MCGS #193 execution-package/CUDA-JS adapter ownership audit
```

The CUDA-JS/Tensor physical boundary refactor is complete. CUDA-JS #178/#179 own provider capability facts and the cuBLASLt borrower lifecycle; Tensor #44 consumes those lower facts without a local provider singleton, provider-family error-code table or copied workspace alignment. CUDA-JS #180 concluded that the existing compiler/module/function/provider-plan/prepared-DAG bricks are the correct preparation LEGO seam and no higher-level `PreparedExecutable` transaction is justified. CUDA-JS #181 retained explicit physical geometry as selected upper-profile policy and added no logical-work resolver.

Tensor #45 integrated through protected PR #49 at `main@be49a63fe0727664a6d27454bfa611ba9a33187f`. The accepted SPEC-0005 physical-boundary correction keeps Tensor-selected `blockSize`, deterministic `workItems -> grid/block`, Tensor workspace policy, u32 logical-work ceiling, Device-JS generation, liveness, accesses and semantic dependencies here. Lower prepared-DAG ceilings, Device-JS function-parameter limits and compiler output families are consumed through the supported public `cuda-js/compatibility` entry. Tensor retains exact `ptx | lto-ir` expert output selection but validates the selected family against the lower public capability. Host planning still composes public CUDA-JS compile/load/function/prepare operations explicitly and closes resources in reverse dependency order; it does not create a second generic runtime lifecycle.

Tensor #40 is closed after a field-by-field ownership matrix on the protected integrated tree. No unresolved CUDA primitive owner remains in this repository: Tensor mathematics/planning/profile policy stays here; lower provider/resource/validity/compatibility facts come from public CUDA-JS; native/private/deep-import escape paths remain absent and forbidden. The next portfolio refactor owner is CUDA-MCGS #193.

The exact lower dependency is CUDA-JS `0.1.0-alpha.17@bc2700f2e5c654567c2e17bf8d67b882351b8681`. Tensor package, public compatibility identity, host/device physical compatibility records, repository verification and independent packed-package consumer all use that exact pair. The prior native numerical/provider/device-callable evidence predates this pair and is not silently promoted to a new native support claim.

SPEC-0009 continues to provide the device-callable correctness profile required by downstream evaluators. `compileTensorDeviceProgram(...)` proves one static item axis, exact shared versus item-varying inputs, item-preserving dense operations, finite typed parameters, explicit item-major outputs and dtype-partitioned per-item workspace. It returns one copied CUDA-JS SPEC-0028 leaf library. A caller-owned Device-JS participant invokes one item; many items can run in parallel without Tensor owning a scheduler, queue, request incarnation, scatter or publication protocol.

CUDA-MCGS remains responsible for evaluator/model meaning, resident input/artifact allocation, ready-item selection, device-owned batching, request/result incarnations, scatter, publication, cancellation, cache/root validity, search policy and multi-GPU coordination. One Tensor session/device remains the v1 owner; independent sessions naturally support replicated per-GPU consumers. Cross-device tensor identity, P2P, collectives and sharding remain later contracts. CUDA-MCGS #125 remains independently blocked by held #122 and by the #193 audit until its lower-boundary findings are dispositioned.

The complete host-planned SIMT path remains available. `simt`, `fusion: 'none'` and PTX library output remain documented safe defaults. Exact fusion, cuBLASLt and item-callable execution are independently removable Legos. The first callable profile is item-parallel, not block/warp-cooperative, and makes no speedup or Tensor Core claim. Cooperative intra-item participation, cuBLASDx/CUTLASS, workspace reuse and performance recommendation require separate exact evidence.

The existing result-owned arena issue #17 remains a useful host-planned optimization but is not the CUDA boundary-refactor critical path. Batched/precision cuBLASLt profiles remain separately dependency-gated. The CUDA-MCGS/Vector consumer-qualification lane remains valid parallel work but does not displace the ownership refactor handoff.

`the_restaurant` integration remains deferred. Its future plan is retained in [`docs/integrations/the_restaurant.md`](docs/integrations/the_restaurant.md); this work does not modify that repository.

TENSOR-DEVICE-ITEM-014 integrated through protected PR #20 at `main@9ecc1d78bca989ec456c897dec215e82ce4cd311`; issue #19 is closed. CUDA-MCGS read-back is preserved on its research branch at `ade6e0bcbef4289e538f0ac02b5bfb9ee8568abc`; that is dependency documentation, not CUDA-MCGS production integration.

Public issues and discussions are enabled. Protected `main` requires a current `verify` check, pull-request integration, linear history, resolved conversations and admin enforcement; force pushes and deletion are disabled. Independent review is waived only under the project owner's sole-maintainer direction and is never represented as independent evidence.

Earlier protected integrations remain part of the repository record: TENSOR-CUBLASLT-013 entered through PR #10 at `main@8910309a0aff9b8da4fc281949068d8d1fcaa6ea`, and TENSOR-FUSION-017 entered through PR #16 at `main@821b1cbe1c32835559cb669248771310718b2f05`. GitHub Actions remain limited to GitHub-owned actions with read-only workflow permissions; repository security scanning, push protection, vulnerability alerts, automated security fixes and private reporting remain enabled.
