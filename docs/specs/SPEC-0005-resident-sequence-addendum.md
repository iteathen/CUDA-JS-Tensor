# SPEC-0005: Resident sequence execution addendum

**Status:** Candidate implementation profile, requiring independent review before integration
**Date:** 2026-10-08
**Authority:** Explicit owner instruction to close the demonstrated generic public host-plan admission gap without bypassing CUDA-JS bounds
**Parent:** SPEC-0004, SPEC-0005 and the SPEC-0005 physical-boundary addendum

The existing single-DAG profile, contracts, defaults and identities remain unchanged. This additive, explicitly selected qualification profile realizes one immutable TensorPlan whose complete SIMT lowering exceeds a single public prepared DAG. It is not an active device-closed product inference profile.

`resolveTensorPlan(session, plan, { execution: 'resident-sequence', backend: 'simt' })` selects the candidate. Omitted execution or explicit `'single-dag'` retains existing behavior; unknown execution values reject. Accelerated backend policies with resident sequence reject before compiler work in this first profile.

Tensor preserves the original plan identity, topological operations, material storage bindings, view offsets/strides, fixed-tree reduction stages, dtype rounding and distinct allocation policy. No graph rewrite, CPU Tensor mathematics, value-based scheduling, allocation reuse, semantic substitution or lower limit increase occurs.

The existing lowering partitions its ordered generated kernels greedily into finite contiguous chunks. Every chunk independently obeys public CUDA-JS prepared limits for nodes, edges, named bindings and predecessors. An atomic kernel exceeding a chunk limit rejects. Global generated kernels are bounded by the accepted 4096 semantic nodes and at most 34 generated stages per node; memory and tensor-count admission remains whole-plan/session-owned. Chunk identities include exact kernel IDs and binding names; each compiler module and public prepared DAG is recorded separately.

Each run creates one whole-plan material/workspace graph. Inputs remain borrowed, and one TensorExecutionResult owns all run-created resources. Host execution submits one public chunk operation, awaits completion and proves operation close before submitting the next. Completion of the earlier public operation supplies cross-chunk dependency order. An execution/operation-close failure stops the sequence, rolls back the whole run, and returns no partially successful result.

The resolved backend owns every chunk module, function and prepared DAG. Resolution rollback closes all acquired chunks in reverse order, including partially acquired chunks. Active-run close remains backpressured by the existing resolver. Unproved cleanup retains its existing failure category and cannot become a success receipt.

The resolved physical contract selects an explicit resident-sequence child and a public descriptor containing the chunk identities/counts. Generic mathematical plan identity remains unchanged. Portable tests prove bounds/lifecycle only; native qualification must use an installed public package, more than 64 global bindings and 32 global kernels, independently expected mathematical outputs, cross-boundary views, failure/cleanup evidence, and exact dependency/runtime/hardware records. No performance, production, model numerical, or device-closed engine claim follows from this qualification.
