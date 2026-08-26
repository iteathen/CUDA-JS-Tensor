# CUDA-MCGS tensor readiness assessment and parent plan

**Status:** Implementation and exact available evidence complete; protected integration and consumer read-back pending

**Parent objective:** Make CUDA-JS-Tensor usable by CUDA-MCGS's device-resident evaluator path without importing search, evaluator, neural-network, scheduler, or product meaning into Tensor.

**Integration owner:** CUDA-JS-Tensor protected `main`

**Exact Tensor input:** `main@cd7245a3ea647e821a2edd4cdbb915e6ee060e15`

**Exact CUDA-JS input:** assessment began on `cuda-js@0.1.0-alpha.15` / `main@af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d`; final candidate consumes selected-target-correct `cuda-js@0.1.0-alpha.16` / protected `main@4971302cfb48431c0843126a59d5884d84a81641`

**Exact consumer input:** CUDA-MCGS research branch `codex/tensor-mcgs-research@667fd655edb462d4f5c23cc1b5d3e4761e9086d1`, especially ADR-0024, ADR-0025, the tensor-math assessment, and proposal SPEC-0009. These documents state the owner's intended evaluator direction but are not represented here as integrated CUDA-MCGS protected-`main` production authority.

## Decision

At the frozen input revision, CUDA-JS-Tensor was ready for host-planned finite Tensor programs but not CUDA-MCGS's active-search evaluator path. `ResolvedTensorPlan.run()` required host submission, while CUDA-MCGS requires evaluator batching, execution, result production, and publication to advance after ignition without a host gather/launch/poll/relaunch loop.

**Implementation update:** SPEC-0009 and `TENSOR-DEVICE-ITEM-014` now implement the selected missing LEGO as the alpha.6 candidate. Portable/package validation and exact installed-package Windows native import/call/output/cleanup evidence pass. The original readiness finding remains the reason for the work; the host-progress gap is closed in code pending protected integration and exact merged-revision reconciliation.

The smallest sufficient missing Tensor LEGO is an **item-parallel device-callable Tensor program**:

- one caller-owned Device-JS participant invokes one finite Tensor batch item;
- many independent items may execute concurrently under the caller's scheduler;
- the Tensor program owns dense mathematics, item-axis independence, typed pointer ABI, finite per-item workspace, exact outputs, identity, and generated Device-JS library source;
- CUDA-JS owns Device-JS validation/lowering, typed library compilation, RDC/LTO composition, artifacts, provider compatibility, and cleanup;
- the consumer owns ready-item selection, participant scheduling, request/result incarnations, scatter, publication, cancellation, and search meaning.

This first profile is deliberately item-parallel rather than block-cooperative. It is useful for dynamic device-owned batches, requires no host progress, and does not guess CUDA-MCGS's eventual physical scheduler. It also applies to streaming, simulation, signal, and other bounded independent-item consumers. Deleting CUDA-MCGS leaves the Tensor contract coherent.

## Strong adversarial case

A serial device function could be dismissed as moving a host loop onto one GPU thread. That would be true if one invocation owned the whole batch. It is not the selected contract. The caller assigns one independent item to each participating thread or other caller-defined participant, so batch concurrency is real and CUDA-MCGS can fill capacity from its device-owned ready queue.

The profile still does not claim efficient execution for every model shape. A large single item, a scalar reduction, or a small ready batch may underutilize the GPU. Block/warp-cooperative execution, shared-memory tiling, cuBLASDx/CUTLASS, typed batched provider calls, and tensor-core profiles remain additive acceleration children that require exact participation and measured end-to-end evidence. They are not prerequisites for a complete device-closed correctness path.

The opposite error is to make Tensor own CUDA-MCGS batching or result publication. That would duplicate evaluator/progress authority and make the library search-specific. Tensor therefore exposes a pure finite callable with disjoint item storage and no queue, readiness, incarnation, cache, policy, or publication state.

## Readiness boundary

### Required in CUDA-JS-Tensor

1. A batch-item independence classifier over the accepted TensorProgram operations.
2. A closed item-axis profile with explicit capacity and explicitly selected item-varying inputs.
3. Deterministic restricted Device-JS library generation through public CUDA-JS SPEC-0028 only.
4. A finite callable ABI containing item index, exact input/output pointers, and dtype-partitioned workspace pointers.
5. Per-item workspace layout, alignment, byte formulas, output packing, compatibility identity, and no-write out-of-range behavior.
6. Complete dense semantics for the admitted slice: views, elementwise operations, reductions that do not reduce the item axis, and rank-2/rank-3 matmul shapes that preserve item independence.
7. A copied immutable public package record that a consumer can import under its own local alias.
8. Portable/package evidence plus exact native composition evidence where the available qualified environment can run it.

### Owned by CUDA-MCGS, not Tensor

- evaluator capabilities, model/package meaning, encoders, request and result identities;
- device-owned queueing, dynamic batch formation, fairness and partial-batch progress;
- request-incarnation validation, result scatter, readiness publication and cancellation;
- cache/root-advance validity, search policy, graph ownership and multi-GPU search coordination;
- model-head meaning or neural-network layer abstractions.

### Owned by CUDA-JS, not Tensor

- Device-JS syntax and generic helpers;
- library compilation/import/linking and opaque artifacts;
- device memory/views, operations, contexts, provider adapters and lifecycle;
- later shared/local/warp primitives or device-callable provider libraries.

## First profile and exclusions

The first profile has one explicit item axis, initially axis 0. Inputs are explicitly classified as item-varying or shared. Every materialized operation must be item-dependent and preserve the item axis; a reduction may not reduce it. Shared immutable inputs may broadcast into item-dependent operations. Public outputs are packed into distinct item-major output bindings. Internal materials use finite dtype-partitioned per-item workspace.

The callable returns without writes when `itemIndex >= itemCapacity`. The caller must still validate its own request/item incarnation before invoking and before publishing results; Tensor does not turn an index bound into semantic freshness.

Excluded from the first profile:

- host submission or a Tensor-owned scheduler;
- queues, scatter tables, readiness words, atomics, publication or cancellation;
- block/warp barriers, dynamic shared memory, local arrays or architecture-specific participation;
- hidden allocation, host spill, dynamic shapes, item-axis reduction, cross-item normalization or batch-sensitive mathematics;
- cuBLASLt, cuBLASDx, CUTLASS, Tensor Core, performance-default or speedup claims;
- multi-device tensors, P2P, collectives, sharding or migration;
- neural layers, model formats, autodiff, training, losses, optimizers or MCGS semantics.

## Focus-branch map

| ID | Owner/output | Exact inputs | Acceptance and falsifier | Cleanup/integration |
|---|---|---|---|---|
| `MCGS-TENSOR-REQ-01` | This assessment and exact ownership/readiness map | Frozen Tensor, CUDA-JS and CUDA-MCGS revisions above | Every stated MCGS tensor need is assigned to Tensor, CUDA-JS or CUDA-MCGS; falsified by a host-progress dependency or duplicated owner | Integrate with the first specification change; update if consumer authority changes |
| `TENSOR-DEVICE-ITEM-SPEC-01` | Accepted Tensor item-parallel callable specification | `MCGS-TENSOR-REQ-01`, Tensor SPEC-0000–0007, CUDA-JS SPEC-0013/0028/0030 | Closed ABI, independence, workspace, numeric, lifecycle, deletion and compatibility rules are decision-complete; falsified by ambiguous item ownership or consumer policy | Registry/status/roadmap reconciliation; no runtime residue if rejected |
| `TENSOR-DEVICE-ITEM-IMPL-01` | Production compiler/profile and public immutable API | Accepted child spec and exact CUDA-JS alpha.16 package | Generated library is deterministic, item-isolated, complete for admitted operations, importable per selected runtime target, and uses no private/native escape; falsified by host advancement, cross-item write, hidden allocation, target drift or semantic drift | Replace/delete the superseded device-callable experiment frame after unique evidence moves to production tests |
| `TENSOR-DEVICE-ITEM-EVIDENCE-01` | Focused, package and available native CUDA-MCGS-style composition evidence | Exact implementation head | Independent oracle covers multiple items, partial capacity, shared weights, multi-head outputs, reduction and matmul; terminal CUDA-JS resource inventory is clean | Preserve exact evidence and claim limits; no performance promotion |
| `MCGS-TENSOR-READY-01` | Final readiness matrix and forward dependencies | Exact integrated Tensor and CUDA-JS revisions plus evidence | CUDA-MCGS can compile/import and invoke the Tensor callable wholly from Device-JS; remaining gaps are optional acceleration or CUDA-MCGS-owned integration | Protected merge, remote read-back, issue/branch/worktree cleanup |

Only one focus branch is executed at a time because specification, public API, generated ABI and evidence share the same owner and invalidate downstream work when changed.

## Dependency order and forward work

```text
item-parallel callable specification
  -> production Tensor device-library compiler/profile
  -> CUDA-MCGS-style Device-JS import/call evidence
  -> readiness reconciliation

result-owned arena
  -> optional host-planned allocation reduction

typed/strided-batched provider family
  -> optional host-planned batched/precision acceleration

real cooperative caller profile + minimum CUDA-JS helpers
  -> optional block/warp-cooperative intra-item Tensor acceleration
```

The existing arena issue remains useful but is no longer the consumer-readiness critical path. Arena reuse optimizes host-planned result-owned material allocations; it does not eliminate the active-search host submission boundary. Batched/precision cuBLASLt likewise remains an optional host-planned accelerator. A later device-callable cooperative profile may reuse their mathematical classification but has a different provider and participation owner.

## Verification and closure

The minimum practice floor is accepted specification, focused ownership/failure/deletion tests, package-consumer import evidence, full `npm run verify`, exact-head author review, protected pull-request integration, remote tree read-back, and complete task-created branch/issue/artifact disposition.

Native evidence must name the exact CUDA-JS/Node/Driver/toolkit/GPU profile and independently check numerical output plus terminal cleanup. It qualifies only that cell. Performance recommendation requires a separate representative method including batch occupancy, tails, workspace, compilation, scheduling, synchronization and best credible alternative; this plan makes no performance claim.

Stop and reclassify upstream if implementation requires private CUDA-JS imports, raw CUDA/PTX, pointer arithmetic escape, unsupported Device-JS syntax, a missing generic scalar/helper, or provider state. Stop and narrow Tensor if item independence cannot be proven statically or if a requested program requires cross-item semantics.
