# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      consumer-backed unary:tanh authority accepted; implementation/evidence next
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt + ordinary SPEC-0010 erf/gather/concat
device-callable execution:  SPEC-0009 base + unary:erf + accepted non-axis-0 static gather/ordered concat child
native qualification:       prior exact Windows evidence remains historical; current pair not silently requalified
performance claims:         none
current package CUDA-JS:    0.1.0-alpha.18 at 30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60
next lower dependency:      cuda-js@0.1.0-alpha.18 at d1a8edef5bd06c402a5c14c8945269f206520174 for accepted SPEC-0030-tanh-v1
current local target:       #61 implement accepted SPEC-0011 unary:tanh through public CUDA-JS only
external coverage owner:    UCI-Arena-Vector #3
cross-repo runtime owner:    CUDA-MCGS #124
```

## Stable ownership boundary

CUDA-JS-Tensor owns generic Tensor mathematics, dtype/shape/layout/stride semantics, TensorProgram/TensorPlan normalization, material/liveness/workspace meaning, deterministic Tensor Device-JS generation and SPEC-0009 item semantics. CUDA-JS owns generic provider/resource lifecycle, lower validity/compatibility, compiler/module/function/library mechanisms and native CUDA execution. CUDA-MCGS owns evaluator request/batch/scatter/publication/search lifecycle. Model/chess/head/product meaning remains downstream.

The currently integrated package dependency remains `cuda-js@0.1.0-alpha.18@30d11a5d38dd7b9987bc8bac4ac67c2fcf8fee60`. Protected CUDA-JS now also contains accepted/implemented `SPEC-0030-tanh-v1` at `d1a8edef5bd06c402a5c14c8945269f206520174`; #61 must select that exact lower revision as part of implementation before Tensor may claim tanh support. Neither exact prerelease reference is a broad compatible-version claim.

## Protected completed Tensor leaves

- **#32 / PR #53** — ordinary SPEC-0010 `unary:erf`, bounded static gather and ordered concat; protected merge `02ae07777632cfa8b527bd5482a59dbff472e4d5`.
- **#52 / PR #55** — device-callable f32/f64 `unary:erf` through the existing SPEC-0009 unary path and public CUDA-JS `gpu.math.erf`; protected merge `23e3795b3f22162e3705f0f2c659400f69f86ed7`.
- **#37 / PRs #58/#59** — accepted and implemented SPEC-0009 device-callable non-axis-0 static gather/ordered concat child; protected implementation merge `999f8b56077f051b2e90a64eebc0a0cf59a659bc`, tree `85a768b521ff493e07fd9f4a6a65a0210d2e67f6`, protected verify `34005276462` / job `101411188269` success.

The #37 child admits gather only from an already item-varying source and concat only when every input is item-varying. Axis 0 and shared-source/shared-input cases fail closed before compiler work. It reuses the existing item-major output ABI and dtype-partitioned per-item workspace and adds no scheduler, host progression loop, runtime index tensor, hidden allocation or CUDA-private path.

## Active dependency lane — #61 unary:tanh

Protected UCI-Arena-Vector #3 / PR #17 corrected the frozen LatticeKnight operation inventory and proved that current Tensor closes the old `erf`/gather/concat gaps while exactly one generic mathematical capability remains uncovered: `unary:tanh`. Supplying tanh in its test-only capability snapshot advances the consumer to `VECTOR_MODEL_WORKSPACE_UNRESOLVED`, so the gap is concrete and dependency-blocking rather than speculative breadth.

Accepted SPEC-0011 owns only the new f32/f64 Tensor unary mathematics and canonical TensorProgram compatibility composition. It preserves exact base and SPEC-0010-only identities and adds:

- `SPEC-0004-tensor-program-v1+SPEC-0011-tanh-v1`;
- `SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1+SPEC-0011-tanh-v1`.

Tanh adds no new semantic pressure limits; a tanh-only program keeps base limits and a mixed SPEC-0010+tanh program keeps the existing SPEC-0010 limits. The requirement is derived from normalized operation use, not caller assertion.

SPEC-0009 already owns the generic rule that `unary` preserves an item-varying input. Therefore #61 requires explicit device-callable lowering/evidence but **no new item-axis, callable ABI, workspace, scheduler or TensorDeviceProgram semantic child**. Ordinary and device-callable generators must use public `gpu.math.tanh` only. No exp identity, private CUDA-JS import, CUDA/PTX/native path, activation/model vocabulary or lower-precision widening is authorized.

## After #61

Once #61 is protected-qualified, Vector #3 should refresh its exact Tensor capability snapshot and prove the semantic gate advances from `unary:tanh` to the already-predicted complete TensorProgram/TensorPlan workspace/resource gate. Tensor #22 then owns the generic real-model coverage/oracle result; CUDA-MCGS #124 consumes only the resulting generic callable/resource facts while retaining evaluator/search lifecycle ownership.

CUDA-MCGS #123 remains a parallel evaluator-free public-package falsifier and does not authorize Tensor or product semantics in CUDA-MCGS.

## Qualification limits

Portable/repository/package evidence does not promote native/provider support. Prior native evidence belongs only to its exact historical pair. No performance recommendation, broad provider accuracy claim, Tensor-Core claim or production-stability claim exists.

## Current-state governance

Protected `STATUS.md` and `next_step.yaml` own the live Tensor execution seam. Issues own durable obligations/evidence, not a live SHA timeline. Historical branches and PRs are research/evidence only unless requalified against current protected authority.

`the_restaurant` implementation remains deferred; its retained integration plan is documentation only.
