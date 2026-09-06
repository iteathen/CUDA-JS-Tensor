# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      SPEC-0010 ordinary Tensor implementation protected-complete; device-callable consumer lane active
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt + ordinary SPEC-0010 erf/gather/concat
device-callable execution:  base SPEC-0009 item-parallel dense Tensor program; unary:erf implementation and gather/concat child remain open
native qualification:       prior exact Windows evidence remains historical; alpha.18 pair not silently requalified
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.18 at 30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60
current local target:       #52 device-callable unary:erf lowering/evidence
next Tensor child:          #37 fresh SPEC-0009 child + implementation for device-callable gather/concat
cross-repo outcome:         #22 evaluator readiness with CUDA-MCGS #124 and a frozen real model
```

## Stable CUDA-JS/Tensor boundary

The CUDA-JS/Tensor ownership refactor remains complete. Tensor owns mathematics, dtype/shape/layout/stride, TensorProgram/TensorPlan normalization, liveness/material/workspace meaning, deterministic Tensor Device-JS generation, selected `blockSize` and deterministic `workItems -> grid/block` policy, and the explicit `ptx | lto-ir` expert compatibility choice. CUDA-JS owns generic provider/resource lifecycle, lower validity/compatibility facts, compiler/module/function/prepared-DAG mechanisms and native CUDA execution.

The exact public lower peer consumed by protected Tensor is now `cuda-js@0.1.0-alpha.18@30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60`. This transition was qualified through Tensor #32 / PR #53 and keeps exact-peer admission fail-closed; it is not a broad compatible-version claim.

No unresolved generic CUDA ownership inversion is currently demonstrated by the active Tensor lane. A downstream desire for native code remains a library-gap diagnostic: generic CUDA mechanism gaps route to CUDA-JS; generic Tensor mathematical/item semantics belong here; search/model/chess/product meaning remains downstream.

## Protected SPEC-0010 ordinary implementation

Tensor #32 / PR #53 is protected-complete at merge `02ae07777632cfa8b527bd5482a59dbff472e4d5`, tree `801b480740100422745b56819e957a3ff7d356f0`.

The reviewed candidate passed required PR verify run `34003437533` / job `101406276051`; the protected merge passed push verify run `34003574151` / job `101406639070`.

The protected ordinary Tensor profile now includes:

- `unary:erf` for exactly `f32`/`f64`, lowered only through public CUDA-JS `gpu.math.erf`;
- bounded canonical static `gather` with exact order/duplicates, empty-index support and pre-execution validation;
- finite ordered materialized `concat` over an explicit axis;
- additive SPEC-0010 TensorProgram identity while base SPEC-0004 programs retain exact legacy contract/limits/identity;
- ordinary resolved-SIMT mapping with lower source/binding/kernel pressure kept execution-owned.

This is portable/software/package evidence. It does not promote native provider support or device-callable gather/concat/erf.

## Active Tensor dependency lane

The next dependency-ready leaf is **#52**, not #37:

- **#52** is implementation/evidence-only. SPEC-0009 already owns unary item propagation, item-axis independence, callable ABI and per-item workspace/output isolation. The remaining work is to lower a valid SPEC-0010 `unary:erf` through the existing device-item unary path and public CUDA-JS helper, with exact legacy-identity/package evidence. No new Tensor item semantics or ABI are authorized.
- **#37** is the next semantic child. Ordinary gather/concat mathematics is complete, but device-callable non-axis-0 gather/concat still requires a fresh current-main accepted SPEC-0009 child defining item propagation, shared-input rejection, no-cross-item-write and compatibility identity before implementation.
- **#22** remains the downstream first-real-model/evaluator-consumer readiness outcome. It does not substitute for #52/#37 implementation and additionally requires a frozen real model/resource/oracle campaign plus CUDA-MCGS #124.

Optional result arena and dormant FFT/sparse/solver roadmaps remain closed/not planned until representative measurement or a concrete consumer activates a bounded profile.

## CUDA-MCGS relationship

CUDA-MCGS #122 semantic acceptance and #125 public CUDA-JS runtime adapter are protected-complete. CUDA-MCGS #124 remains the evaluator request/batch/incarnation/scatter/publication/search-lifecycle owner and is currently downstream of Tensor #52/#37/#22 plus the frozen real-model coverage gate.

CUDA-MCGS #123 has been reopened because its previous completed state had no acceptance evidence. Its evaluator-free CUDA-free Vector public-package falsifier is a parallel external-consumer lane and does not move Tensor semantics into CUDA-MCGS or block completion of local #52/#37 work. Physical CUDA-JS/CUDA-MCGS qualification remains a separate hardware evidence gate.

## Qualification limits

The exact lower dependency is CUDA-JS alpha.18 at `30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60`. Prior native cuBLASLt/device-callable evidence predates this exact pair and remains historical. Portable/repository/package evidence does not silently promote native support.

Completed capability issues close when their own implementation/package gates pass; broader hardware/provider cells and downstream product readiness stay with their natural outcome owners.

## Current-state governance

Protected `STATUS.md` and `next_step.yaml` own the live Tensor execution seam. Issues own durable implementation/outcome/governance obligations rather than a live SHA dashboard. #31 owns recurrence prevention for stale target/priority declarations; the post-SPEC-0010 reconciliation is intended to satisfy its current-state consistency requirement.

`the_restaurant` implementation remains deferred. Its retained integration plan remains documentation only under `docs/integrations/the_restaurant.md`.
