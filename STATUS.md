# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      protected real-model TensorProgram/callable/resource facts complete; external independent oracle next
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt + ordinary SPEC-0010 erf/gather/concat + SPEC-0011 tanh
device-callable execution:  SPEC-0009 base + f32/f64 unary:erf/tanh + accepted non-axis-0 static gather/ordered concat child
native qualification:       prior exact Windows evidence remains historical; current pair not silently requalified
performance claims:         none
current package CUDA-JS:    0.1.0-alpha.18 at 45a9ef15537b52d6fd7c615b7e596676dfd00587
protected pair refresh:     #68 merge 0da2c70a0a10df908a33e842aa4ba3dbd7605c48, tree 1330080e112870b696c05dcd2b6d48aece91b1dc
protected tanh provenance:  #61 / PR #63 merge 3a62bc47017aa10198eb1640b66f6b71a608b562
current local target:       #22 retain generic real-model callable/workspace facts and route only any oracle-demonstrated generic defect
external oracle owner:      UCI-Arena-Vector #3
cross-repo runtime owner:    CUDA-MCGS #124
```

## Stable ownership boundary

CUDA-JS-Tensor owns generic Tensor mathematics, dtype/shape/layout/stride semantics, TensorProgram/TensorPlan normalization, material/liveness/workspace meaning, deterministic Tensor Device-JS generation and SPEC-0009 item semantics. CUDA-JS owns generic provider/resource lifecycle, lower validity/compatibility, compiler/module/function/library mechanisms and native CUDA execution. CUDA-MCGS owns evaluator request/batch/scatter/publication/search lifecycle. Model/checkpoint/chess/head/product numerical meaning remains downstream in UCI-Arena-Vector.

The protected Tensor package now selects exactly `cuda-js@0.1.0-alpha.18@45a9ef15537b52d6fd7c615b7e596676dfd00587`. Package dependency, lockfile, verifier, public compatibility identity and installed-package evidence agree on that revision. The f32/f64 public `SPEC-0030-tanh-v1` implementation provenance remains the earlier CUDA-JS protected `d1a8edef5bd06c402a5c14c8945269f206520174`; the later selected revision includes the generic bounded Device-JS/compiler admission correction from CUDA-JS #213/#216. This exact prerelease pair is not a broad compatible-version, native/provider or hardware-support claim.

## Protected completed Tensor leaves

- **#32 / PR #53** — ordinary SPEC-0010 `unary:erf`, bounded static gather and ordered concat; protected merge `02ae07777632cfa8b527bd5482a59dbff472e4d5`.
- **#52 / PR #55** — device-callable f32/f64 `unary:erf` through the existing SPEC-0009 unary path and public CUDA-JS `gpu.math.erf`; protected merge `23e3795b3f22162e3705f0f2c659400f69f86ed7`.
- **#37 / PRs #58/#59** — accepted and implemented SPEC-0009 device-callable non-axis-0 static gather/ordered concat child; protected implementation merge `999f8b56077f051b2e90a64eebc0a0cf59a659bc`, tree `85a768b521ff493e07fd9f4a6a65a0210d2e67f6`, protected verify `34005276462` success.
- **#61 / PR #63** — accepted SPEC-0011 f32/f64 `unary:tanh`, exact TensorProgram contract composition, ordinary/fused lowering and device-callable realization through public `gpu.math.tanh`; protected implementation merge `3a62bc47017aa10198eb1640b66f6b71a608b562`, tree `6b8ca9dd802c0f24f93b8cf612d8ba576e932857`, protected post-merge verify `34012270778` success.
- **#68** — exact lower-pair refresh after the generic CUDA-JS source-admission correction; protected merge `0da2c70a0a10df908a33e842aa4ba3dbd7605c48`, tree `1330080e112870b696c05dcd2b6d48aece91b1dc`, protected post-merge verify `34020688142` success.

SPEC-0009 remains the sole owner of generic unary item-varying propagation, callable ABI, item-major outputs and per-item workspace. Tanh adds no TensorDeviceProgram semantic child, scheduler, queue, shared mutable channel or lifecycle owner. Portable evidence uses public CUDA-JS only; no exp reconstruction, private CUDA-JS path, CUDA/PTX/native Tensor implementation, activation/model vocabulary or lower-precision widening was introduced.

## Active dependency lane — first real model / #22 and Vector #3

UCI-Arena-Vector #3 / PR #24 is now protected-complete for the public TensorProgram/TensorPlan/callable/resource gate. Vector merge `ca3cce162a73a664a789f6a27a819097ec994bd6`, tree `2182527aa9cd058824d3658a549a2f8224d18db5`, passed protected post-merge `Repository quality` `34023725885` and `Model Tensor Coverage` `34023725845`.

The frozen LatticeKnight-4M FP32 program demonstrates these consumer-neutral Tensor facts on the current exact pair:

- canonical mixed TensorProgram contract `SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1+SPEC-0011-tanh-v1`;
- 2,216 normalized nodes / 1,340 materialized nodes;
- device-callable contract `SPEC-0009-item-parallel-device-tensor-program-v1+SPEC-0009-gather-concat-v1`;
- seven-parameter callable ABI with one item-varying input, two shared inputs, two item-major outputs and one f32 workspace partition;
- per-item workspace: **8,298,631 f32 elements / 33,194,524 bytes**;
- item capacity 2: 66,389,048 workspace bytes, below the accepted default 67,108,864-byte limit;
- item capacity 3: 99,583,572 required bytes and fail-closed `TENSOR_DEVICE_WORKSPACE_LIMIT` / `pressure` before compiler work;
- portable public-package compilation and TensorSession/runtime cleanup are graceful.

No new generic Tensor mathematics, dtype, item-axis, ABI, workspace or lower CUDA mechanism gap is demonstrated by this real model. The old pre-tanh `VECTOR_MODEL_WORKSPACE_UNRESOLVED` result remains historical evidence only; current Vector v2/v3 evidence now carries the exact finite workspace.

Vector's `covered_real_model` / `real_model_ready=true` fields are its capability/resource coverage predicate. They do not establish checkpoint output parity, partial-batch numerical execution, evaluator readiness, native/provider support or release readiness.

The remaining first-real-model gate is an **independent checkpoint-bound numerical oracle**, owned by Vector #3. The immutable producer evidence available in GitHub does not contain checkpoint-bound policy/value reference vectors or a persisted parity receipt for checkpoint SHA `62dec13c22a4414db6b78ea9b6ca76bcf6f29a16a963c01d13d947d158b09c7e`. Tensor must not manufacture product expected outputs or absorb model semantics to close that gap.

If the independent oracle later demonstrates a consumer-neutral Tensor math/item/ABI/workspace defect, route that specific defect to #22. If it demonstrates a lower compiler/runtime mechanism defect, route it to CUDA-JS. Otherwise the protected generic resource/callable record stands unchanged.

CUDA-MCGS #124 may consume only public generic callable/resource facts after the first-real-model correctness/oracle gate is complete, while retaining evaluator request identity, queueing, batching/scatter, readiness/publication and search lifecycle ownership. CUDA-MCGS #123 remains a parallel evaluator-free public-package falsifier.

## Qualification limits

Portable/repository/package evidence does not promote native/provider support. Prior native evidence belongs only to its exact historical pair. No performance recommendation, broad provider accuracy/ULP claim, Tensor-Core claim, lower-precision tanh claim or production-stability claim exists. A product numerical oracle is not Tensor evidence unless it independently exposes a generic Tensor defect.

## Current-state governance

Protected `STATUS.md` and `next_step.yaml` own the live Tensor execution seam. Issues own durable obligations/evidence, not a live SHA timeline. Historical branches and PRs are research/evidence only unless requalified against current protected authority.

`the_restaurant` implementation remains deferred; its retained integration plan is documentation only.
