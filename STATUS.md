# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      CUDA ownership refactor complete; accepted SPEC-0010 implementation lane
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt
device-callable execution:  base item-parallel dense Tensor program; gather/concat child still open
native qualification:       prior exact Windows evidence remains historical; alpha.17 pair not silently requalified
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.17 at bc2700f2e5c654567c2e17bf8d67b882351b8681
current local target:       #32 ordinary erf/gather/concat implementation
next Tensor child:          #37 device-callable gather/concat item semantics
cross-repo outcome:         #22 evaluator readiness with CUDA-MCGS #124/#125 and a frozen real model
```

## Stable CUDA-JS/Tensor boundary

The CUDA-JS/Tensor physical-boundary refactor is complete. Tensor #40/#44/#45 are protected-integrated and closed. Tensor owns mathematics, dtype/shape/layout/stride, TensorProgram/TensorPlan normalization, liveness/material/workspace meaning, deterministic Tensor Device-JS generation, selected `blockSize` and deterministic `workItems -> grid/block` policy, and the explicit `ptx | lto-ir` expert compatibility choice. CUDA-JS owns generic provider/resource lifecycle, lower validity/compatibility facts, compiler/module/function/prepared-DAG mechanisms and native CUDA execution.

No unresolved generic CUDA ownership inversion remains in Tensor. A downstream desire for native code is still a library-gap diagnostic: generic CUDA mechanism gaps route to CUDA-JS; generic Tensor mathematical/planning gaps belong here; search/model/chess/product meaning remains downstream.

## Active Tensor implementation lane

Accepted SPEC-0010 defines ordinary Tensor semantics for `unary:erf`, bounded static indexed `gather`, and finite ordered materialized `concat`.

- **#32** now owns only the ordinary Tensor normalization/planning/reference/SIMT/package implementation. Static gather and concat are locally actionable. `unary:erf` realization additionally depends on CUDA-JS #157's public helper.
- **#37** owns the separate device-callable SPEC-0009 child for item-preserving non-axis-0 gather/concat. Axis-0 and shared-input cases remain fail-closed in the first child.
- **#22** owns the first-real-model/evaluator-consumer readiness outcome and coordinates with CUDA-MCGS #124/#125. It is not a substitute for #32/#37 implementation.

The optional result-owned arena (#17) and FFT/sparse/solver roadmaps (#41/#42/#43) are closed `not planned` until representative measurement or a concrete consumer activates a bounded profile. Their design provenance remains searchable; they are not active backlog.

## CUDA-MCGS relationship

CUDA-MCGS #122 is protected-accepted. CUDA-MCGS #125 is now the base production adapter owner and is currently blocked by demonstrated public lower gaps such as CUDA-JS #193 ordinary base-allocation alignment, not by held semantic acceptance.

CUDA-MCGS #124 owns evaluator request/batch/incarnation/scatter/publication/search lifecycle. Tensor remains replaceable: evaluator-free and materially different non-Tensor evaluators must remain possible downstream.

## Qualification limits

The exact lower dependency remains CUDA-JS alpha.17 at `bc2700f2e5c654567c2e17bf8d67b882351b8681`. Prior native cuBLASLt/device-callable evidence predates this exact pair and remains historical. Portable/repository/package evidence does not silently promote native support.

Completed capability issues should close when their implementation/package gates pass; additional hardware/provider cells belong to exact qualification campaigns rather than keeping implementation issues open indefinitely.

## Current-state governance

Protected `STATUS.md` and `next_step.yaml` own the live Tensor execution seam. Issues own durable implementation/outcome/governance obligations rather than live SHA dashboards. Issue #31 owns recurrence prevention for stale target/priority declarations.

`the_restaurant` integration remains deferred and its retained plan remains under `docs/integrations/the_restaurant.md`.
