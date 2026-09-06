# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      ordinary + required device-callable Tensor primitives protected-complete; consumer-readiness gate active
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt + ordinary SPEC-0010 erf/gather/concat
device-callable execution:  SPEC-0009 base + unary:erf + accepted non-axis-0 static gather/ordered concat child
native qualification:       prior exact Windows evidence remains historical; alpha.18 pair not silently requalified
performance claims:         none
exact CUDA-JS pair:         0.1.0-alpha.18 at 30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60
current local target:       no generic Tensor implementation leaf; #22 waits on one frozen real model/public-contract coverage campaign
external coverage owner:    UCI-Arena-Vector #3
cross-repo runtime owner:    CUDA-MCGS #124
```

## Stable ownership boundary

CUDA-JS-Tensor owns generic Tensor mathematics, dtype/shape/layout/stride semantics, TensorProgram/TensorPlan normalization, material/liveness/workspace meaning, deterministic Tensor Device-JS generation and SPEC-0009 item semantics. CUDA-JS owns generic provider/resource lifecycle, lower validity/compatibility, compiler/module/function/library mechanisms and native CUDA execution. CUDA-MCGS owns evaluator request/batch/scatter/publication/search lifecycle. Model/chess/head/product meaning remains downstream.

The exact public lower peer is `cuda-js@0.1.0-alpha.18@30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60`. This is an exact prerelease peer, not a broad compatible-version claim.

## Protected completed Tensor leaves

- **#32 / PR #53** — ordinary SPEC-0010 `unary:erf`, bounded static gather and ordered concat; protected merge `02ae07777632cfa8b527bd5482a59dbff472e4d5`.
- **#52 / PR #55** — device-callable f32/f64 `unary:erf` through the existing SPEC-0009 unary path and public CUDA-JS `gpu.math.erf`; protected merge `23e3795b3f22162e3705f0f2c659400f69f86ed7`.
- **#37 / PRs #58/#59** — accepted and implemented SPEC-0009 device-callable non-axis-0 static gather/ordered concat child; protected implementation merge `999f8b56077f051b2e90a64eebc0a0cf59a659bc`, tree `85a768b521ff493e07fd9f4a6a65a0210d2e67f6`, protected verify `34005276462` / job `101411188269` success.

The #37 child admits gather only from an already item-varying source and concat only when every input is item-varying. Axis 0 and shared-source/shared-input cases fail closed before compiler work. It reuses the existing item-major output ABI and dtype-partitioned per-item workspace and adds no scheduler, host progression loop, runtime index tensor, hidden allocation or CUDA-private path.

## Active dependency lane

There is currently **no demonstrated missing generic Tensor primitive** on the first evaluator path.

Tensor #22 remains the cross-repository evaluator-readiness outcome. Its next evidence comes from one exact frozen real model owned by `iteathen/UCI-Arena-Vector#3`, which must provide:

- complete input/output TensorSpecs, dtypes, layouts and item-axis meaning;
- complete post-translation operation inventory;
- mapping of every operation to protected public Tensor/CUDA-JS authority or one explicit natural-owner gap;
- normalized f32 TensorProgram and exact model/parameter/per-item/workspace resource accounting;
- typed multi-head output ABI;
- full/partial item-batch comparison against an independent oracle; and
- exact package/target identity plus terminal cleanup.

If that campaign exposes a genuinely generic Tensor capability gap, open/accept the smallest Tensor child here and keep product/model meaning downstream. Do not invent Tensor breadth merely because #22 remains open.

CUDA-MCGS #124 owns the separate evaluator queue/request-incarnation/batch/scatter/publication/search-lifecycle connector after the public model coverage gate is sufficiently closed. CUDA-MCGS #123 remains a parallel evaluator-free external-consumer/public-package falsifier and does not authorize Tensor or product semantics in CUDA-MCGS.

## Qualification limits

Portable/repository/package evidence does not promote native/provider support. Prior native evidence belongs only to its exact historical pair. No performance recommendation, broad provider accuracy claim, Tensor-Core claim or production-stability claim exists.

## Current-state governance

Protected `STATUS.md` and `next_step.yaml` own the live Tensor execution seam. Issues own durable obligations/evidence, not a live SHA timeline. Historical branches and PRs are research/evidence only unless requalified against current protected authority.

`the_restaurant` implementation remains deferred; its retained integration plan is documentation only.
