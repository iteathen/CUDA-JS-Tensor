# SPEC-0007: Exact elementwise fusion

**Status:** Accepted implementation profile

**Date:** 2026-08-26

**Parent:** SPEC-0002 through SPEC-0006

**Issue:** #14

## Outcome

Permit one resolved tensor plan to replace eligible generated `cast`/`unary`/`binary` kernel chains with one exact generated Device-JS kernel while deleting only unobservable intermediate materials. Fusion is an optional realization of unchanged TensorProgram mathematics. It creates no graph optimizer, scheduler, native boundary, provider contract or consumer policy.

The complete unfused SPEC-0005 SIMT lowering remains available for every accepted program. Removing this child restores that path without semantic loss.

## Public policy and normalization

Resolved-plan options add exactly:

```text
fusion = "none" | "exact-elementwise"
```

`none` is the documented default. `exact-elementwise` selects every eligible deterministic region during resolution. The policy, selected regions and generated lowering enter immutable compatibility identity. Convenience and expert resolution forms normalize through the same option record.

Changing the default requires representative total-cost evidence. This profile makes no performance recommendation.

## Exact v1 region

An eligible region contains two or more topologically consecutive-by-dependency materialized nodes selected by this finite rule:

- every node is `cast`, `unary` or `binary` under accepted SPEC-0004 semantics;
- every output is nonempty and has the same capacity shape, logical shape and active-axis-0 identity;
- each internal output is not public and has exactly one distinct downstream consumer;
- the next node directly consumes the prior output;
- no next node also reaches an earlier non-immediate region value;
- external side inputs remain explicit read bindings with their accepted broadcast/view ranges;
- the final material remains the sole region output allocation.

Discovery scans program topological order and greedily selects maximal non-overlapping regions. Fan-out, an observable intermediate, a view boundary, shape/active-extent change, empty output, unsupported operation or non-linear internal dependency ends the region. Ineligible nodes remain on the complete unfused path; explicit fusion is not a strict all-nodes request.

## Mathematical equivalence

The fused kernel evaluates nodes in original order into one typed local value per node. Every node retains its declared dtype and exact helper/operator lowering. CUDA-JS compilation keeps `fmad: false`; no fast math, reassociation, contraction or approximate helper is selected. Cast, modulo-width integer arithmetic, floating rounding, NaN propagation, minimum/maximum and signed-zero behavior remain the corresponding SPEC-0004/SPEC-0030 meanings after every operation.

The final typed local is written to the original final material. No internal material is publicly observable or externally consumed, so removing its global store/load cannot remove an accepted output, alias or synchronization boundary.

## Lowering, resources and identity

Each selected region becomes one generated kernel at the final node's execution position. It retains the canonical single-stream predecessor order, original final work count, block-size policy, explicit external parameters, exact read ranges and final write range. Its descriptor records the complete ordered semantic-node list and fusion-region identity.

Internal material bindings and physical byte requirements are absent from the resolved lowering. Session-fit and run admission use the realized material list and byte total, not the unfused TensorPlan allocation total. Fusion adds no workspace. Kernel/binding ceilings apply after substitution.

Backend selection remains separate. Adjacent eligible cuBLASLt matmul nodes retain SPEC-0006 selection and may feed or consume a fused kernel through existing bindings in the same prepared DAG.

## Lifecycle and failure

Each run allocates only realized final/non-fused materials plus existing reduction/accelerator workspaces. Results own those materials and any output views exactly as in SPEC-0005/0006. Rollback and close order are unchanged.

A fused kernel failure is reported under the ordinary whole-DAG operation failure contract and its descriptor identifies every aggregated semantic node. Fusion never converts compilation, preparation, execution or cleanup failure into unfused retry during `run()`.

## Evidence and limits

Portable evidence must cover deterministic selection/identity, option rejection, public-output and fan-out boundaries, duplicated immediate inputs, dtype changes, side-input broadcasting, material/binding/kernel deletion, complete unfused equivalence, run allocation/cleanup and composition with existing backend profiles.

Native evidence must execute fused and unfused public-package plans against independently computed expected outputs and prove terminal cleanup on the exact recorded environment. Passing evidence does not establish a speedup, automatic default, Tensor Core use, Linux support or broader device/provider support.

## Falsifiers and non-goals

Rollback this child if exact rounding requires private/native escape, an observable or independently consumed value is deleted, access ranges become implicit, session admission retains unfused phantom resources, failure attribution becomes false, or deletion damages the complete unfused path.

Reduction, matmul, copy/contiguous, view, arbitrary DAG, approximate and provider fusion are excluded. Arena reuse, cross-run pools, multi-device tensors, neural/training/search policy and `the_restaurant` remain separate owners.
