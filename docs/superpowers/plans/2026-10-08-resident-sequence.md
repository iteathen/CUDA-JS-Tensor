# Resident sequence implementation plan

> **For agentic workers:** Use superpowers:executing-plans with TDD; root independently reviews the candidate before main integration.

**Goal:** Realize a whole generic TensorPlan through bounded public prepared-DAG chunks without changing mathematical lowering or material ownership.

**Architecture:** Add explicit `execution: 'resident-sequence'` to the existing resolver. Existing SIMT lowering emits its same ordered kernels and whole material graph, then partitions only prepared realization. The existing CUDA-JS backend compiles/resides chunks and serializes public completion/close; the existing run/result lifecycle remains authoritative.

**Tech stack:** JavaScript/Node, exact public CUDA-JS alpha.21, Tensor alpha.8 candidate.

**Spec:** `docs/specs/SPEC-0005-resident-sequence-addendum.md` (candidate approved for implementation by root), preserving accepted SPEC-0004/0005/0006 baseline.

## Global constraints

- Single-DAG default and old physical identities remain unchanged.
- Per-chunk public limits stay nodes 32, bindings 64, edges 64; no native/private CUDA imports.
- First sequence profile is SIMT only and qualification-only host execution.
- One whole TensorPlan identity, material layout and result lifecycle; no CPU Tensor math.
- Package alpha.7 commit `9fb3845c35e4fba29972722426159767ef1171a9` and historical evidence remain immutable.

## Review focus

- Cross-chunk dependency and view storage must remain correct.
- More than 64 global bindings and 32 kernels must produce bounded chunks.
- Chunk N failures must close the operation, stop later submissions and return no partial result.
- Partial resolution rollback must close every acquired DAG/function/module.
- Unknown execution options and accelerated sequence policies reject before compiler work.

## Task

- [x] Reproduce consumer full-plan `TENSOR_SIMT_BINDING_LIMIT` before GPU work; write six failing bounds/lifecycle tests.
- [x] Implement explicit option, bounded static chunk projection and existing-backend sequence execution; run tests green.
- [x] Add installed-public native fixture with 80 kernels/81 bindings and view crossings; run exact Node 26.11.1 native qualification.
- [x] Update current package/compatibility projection to alpha.8, run all generic tests and pack the candidate externally for independent review.

## Execution record

Root approved this candidate addendum after confirming accepted baseline profiles require one DAG and do not implicitly authorize host waits across DAGs. No baseline authority is relabeled. A successful candidate native receipt is confined to generic qualification and does not promote active product inference.

118 tests pass. Native installed package on Node 26.11.1 passed an 80-kernel/81-binding view chain (32/32/16 chunks) and 35 independent add nodes/105 bindings (21/14 chunks cut at 63/42 bindings), each replayed twice against literal expectations. Driver terminal reports 254 closed, zero live/orphaned resources; compiler created/destroyed five programs; session bytes/tensors/resolved plans return to zero. Default native conformance on Node 26.7.0 also passes. The native fixture's initial Tensor.close terminal assertion was corrected from unsupported `graceful` to its documented `state: 'closed'`; no execution source repair was needed.

Package retained externally at `E:/uci-arena-task-builds/tensor-resident-sequence-evidence-20261008/cuda-js-tensor-0.1.0-alpha.8.tgz`; installed unrelated native consumer retained for integration, with all runtime resources closed. Alpha.7 commit and its historical receipt stay untouched. Independent review is pending; publication/main integration is not authorized by this packet.
