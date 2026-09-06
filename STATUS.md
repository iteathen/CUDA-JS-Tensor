# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      SPEC-0010 ordinary + device-callable erf protected-complete; gather/concat item child active
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt + ordinary SPEC-0010 erf/gather/concat
device-callable execution:  base SPEC-0009 + protected unary:erf; gather/concat child authority in current transaction
native qualification:       prior exact Windows evidence remains historical; alpha.18 pair not silently requalified
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.18 at 30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60
current local target:       #37 device-callable non-axis-0 static gather/concat child + implementation
cross-repo outcome:         #22 evaluator readiness with CUDA-MCGS #124 and a frozen real model
```

## Stable CUDA-JS/Tensor boundary

Tensor owns mathematics, dtype/shape/layout/stride, TensorProgram/TensorPlan normalization, liveness/material/workspace meaning, deterministic Tensor Device-JS generation and SPEC-0009 item semantics. CUDA-JS owns generic provider/resource lifecycle, lower validity/compatibility, compiler/module/function/library mechanisms and native CUDA execution. Search/model/chess/product meaning remains downstream.

The exact public lower peer is `cuda-js@0.1.0-alpha.18@30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60`. This is an exact prerelease peer, not a broad compatible-version claim.

## Protected completed Tensor leaves

Tensor #32 / PR #53 is protected-complete at merge `02ae07777632cfa8b527bd5482a59dbff472e4d5`, tree `801b480740100422745b56819e957a3ff7d356f0`. It implements ordinary SPEC-0010 `unary:erf`, static gather and ordered concat with additive TensorProgram identity and portable/package evidence.

Tensor #52 / PR #55 is protected-complete at merge `23e3795b3f22162e3705f0f2c659400f69f86ed7`, tree `05ffa1771009955e833775a0893612387476b4b7`. The reviewed candidate passed verify run `34004498401`; protected push verify run `34004532472` / job `101409231676` also passed. Device-callable `unary:erf` now reuses the existing SPEC-0009 unary item propagation/ABI/workspace path and public CUDA-JS `gpu.math.erf`. No new item contract or native/provider claim was introduced.

## Active Tensor dependency lane

The current dependency-ready child is **#37**.

Fresh current-main reassessment confirmed the old closed/unmerged PR #39 was stale transport but its narrow semantic core remains valid. The current authority transaction accepts a SPEC-0009 addendum with only:

- non-axis-0 static gather from an already item-varying source;
- non-axis-0 ordered concat only when every input is already item-varying;
- axis-0 and shared-source/shared-input cases rejected before compiler work;
- no new callable parameters, scheduler, runtime table, hidden allocation, host progression or CUDA-private mechanism;
- existing item-major outputs and dtype-partitioned per-item workspace retained;
- additive `SPEC-0009-gather-concat-v1` identity only when gather/concat item semantics are selected; base SPEC-0009 profiles remain exact.

After this authority transaction is protected and verified, #37 implementation is the next local action. #22 remains downstream and additionally requires a frozen real model/resource/oracle campaign plus CUDA-MCGS #124.

## CUDA-MCGS relationship

CUDA-MCGS #122 semantic acceptance and #125 public CUDA-JS runtime adapter are protected-complete. CUDA-MCGS #124 remains the evaluator request/batch/incarnation/scatter/publication/search-lifecycle owner and is downstream of Tensor #37/#22 plus the frozen real-model gate.

CUDA-MCGS #123 is a separate evaluator-free external-consumer falsifier lane. It neither blocks #37 nor authorizes Tensor/product semantics in CUDA-MCGS. Physical CUDA-JS/CUDA-MCGS and Tensor native/provider qualification remain separate hardware evidence gates.

## Qualification limits

Portable/repository/package evidence does not silently promote native/provider support. Prior recorded native evidence belongs only to its exact historical pair. No performance recommendation or production-stability claim exists.

## Current-state governance

Protected `STATUS.md` and `next_step.yaml` own the live execution seam. Issues own durable obligations and evidence, not a live SHA dashboard. Historical branches/PRs are research/evidence only unless requalified against current protected authority.

`the_restaurant` implementation remains deferred; its retained integration plan is documentation only.
