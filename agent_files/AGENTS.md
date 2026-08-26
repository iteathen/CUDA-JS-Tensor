# CUDA-JS-Tensor Operating Manual

## Task lifecycle

For every task:

1. identify the exact outcome, authority, current state, smallest coherent scope, and evidence needed;
2. read the owning ADR/specification and registry boundary;
3. assess architecture, dependency, memory, concurrency, generated-code, lifecycle, compatibility, and cleanup consequences before planning;
4. split large work by semantic owner with exact inputs, output, non-goals, acceptance, falsifier, rollback, cleanup, and integration obligations;
5. execute one dependency-ready ownership-sized operation;
6. inspect actual effects immediately and revise rather than silently expanding scope;
7. run focused tests, repository validation, and any required native qualification;
8. review the exact head, push it, use protected PR integration, and verify remote tree plus cleanup.

Token or time pressure narrows scope and claim; it never removes the minimum practice floor.

## Branch and repository model

`main` is protected integration authority. Use short-lived `codex/*`, `feature/*`, or `agent/*` branches and squash one coherent PR. A semantic focus branch is not automatically a Git branch or issue.

Preserve user work. Never reset, overwrite, or delete unrelated local or remote state. Every task-created branch, worktree, artifact, cache, process, issue, PR, and external setting receives a verified disposition.

## Design and dependency rules

- Repository -> product area -> component -> subsystem/module -> file.
- New components require a manifest, README, registry entry, dependency declaration, and validation boundary in one change.
- No `common`, `shared`, `misc`, `utils`, or `helpers` dumping grounds.
- Dependencies point from public tensor semantics to injected CUDA-JS ports and private adapters, never from CUDA-JS to this package.
- Specialized generated hot paths remain behind universal tensor contracts.
- Reject prospective abstraction without a second materially different use or a stable owned invariant.

## Native and performance evidence

Portable tests prove semantics and orchestration only. Native promotion requires an exact Node/OS/ABI/CUDA-JS/Driver/toolkit/GPU/profile record, an independent mathematical oracle, terminal cleanup, and claim limits. Performance requires representative shape distributions, warmup, tail behavior, allocation/workspace cost, synchronization, raw results, and the best credible non-accelerated baseline.

