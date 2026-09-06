# Status

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.6 (publication-guarded)
phase:                      protected portable/package f32/f64 unary:tanh complete; frozen-model exact-pair requalification next
host-planned execution:     complete dense SIMT + optional exact fusion + bounded optional cuBLASLt + ordinary SPEC-0010 erf/gather/concat + SPEC-0011 tanh
device-callable execution:  SPEC-0009 base + f32/f64 unary:erf/tanh + accepted non-axis-0 static gather/ordered concat child
native qualification:       prior exact Windows evidence remains historical; current pair not silently requalified
performance claims:         none
current package CUDA-JS:    0.1.0-alpha.18 at d1a8edef5bd06c402a5c14c8945269f206520174
protected tanh result:      #61 / PR #63 merge 3a62bc47017aa10198eb1640b66f6b71a608b562, tree 6b8ca9dd802c0f24f93b8cf612d8ba576e932857
current local target:       #22 consume refreshed exact Vector #3 coverage/resource/oracle evidence; no model semantics in Tensor
external coverage owner:    UCI-Arena-Vector #3
cross-repo runtime owner:    CUDA-MCGS #124
```

## Stable ownership boundary

CUDA-JS-Tensor owns generic Tensor mathematics, dtype/shape/layout/stride semantics, TensorProgram/TensorPlan normalization, material/liveness/workspace meaning, deterministic Tensor Device-JS generation and SPEC-0009 item semantics. CUDA-JS owns generic provider/resource lifecycle, lower validity/compatibility, compiler/module/function/library mechanisms and native CUDA execution. CUDA-MCGS owns evaluator request/batch/scatter/publication/search lifecycle. Model/chess/head/product meaning remains downstream.

The protected Tensor package now selects exactly `cuda-js@0.1.0-alpha.18@d1a8edef5bd06c402a5c14c8945269f206520174`, which provides accepted/implemented public `SPEC-0030-tanh-v1` for f32/f64. The Tensor public compatibility manifest, package dependency, lockfile, verifier and installed-package evidence agree on that revision. This exact prerelease pair is not a broad compatible-version, native/provider or hardware-support claim.

## Protected completed Tensor leaves

- **#32 / PR #53** — ordinary SPEC-0010 `unary:erf`, bounded static gather and ordered concat; protected merge `02ae07777632cfa8b527bd5482a59dbff472e4d5`.
- **#52 / PR #55** — device-callable f32/f64 `unary:erf` through the existing SPEC-0009 unary path and public CUDA-JS `gpu.math.erf`; protected merge `23e3795b3f22162e3705f0f2c659400f69f86ed7`.
- **#37 / PRs #58/#59** — accepted and implemented SPEC-0009 device-callable non-axis-0 static gather/ordered concat child; protected implementation merge `999f8b56077f051b2e90a64eebc0a0cf59a659bc`, tree `85a768b521ff493e07fd9f4a6a65a0210d2e67f6`, protected verify `34005276462` / job `101411188269` success.
- **#61 / PR #63** — accepted SPEC-0011 f32/f64 `unary:tanh`, exact TensorProgram contract composition, ordinary/fused lowering and device-callable realization through public `gpu.math.tanh`; protected implementation merge `3a62bc47017aa10198eb1640b66f6b71a608b562`, tree `6b8ca9dd802c0f24f93b8cf612d8ba576e932857`, protected post-merge verify `34012270778` success.

The #37 child admits gather only from an already item-varying source and concat only when every input is item-varying. Axis 0 and shared-source/shared-input cases fail closed before compiler work. It reuses the existing item-major output ABI and dtype-partitioned per-item workspace and adds no scheduler, host progression loop, runtime index tensor, hidden allocation or CUDA-private path.

The #61 protected merge tree exactly equals the reviewed candidate tree. Required PR run `34012158868` passed 107/107 tests on reviewed head `963ec6ade0c64d1d870b92fadf4b45b5712fc34a` against the then-protected base. The existing base and SPEC-0010-only program identities remain unchanged; canonical tanh states are `SPEC-0004-tensor-program-v1+SPEC-0011-tanh-v1` and `SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1+SPEC-0011-tanh-v1`, with SPEC-0010 before SPEC-0011. Tanh adds no semantic limits.

SPEC-0009 remains the sole owner of generic unary item-varying propagation, callable ABI, item-major outputs and per-item workspace. Device-callable tanh therefore adds no TensorDeviceProgram semantic child, scheduler, queue, shared mutable channel or lifecycle owner. Portable evidence uses public CUDA-JS only; no exp reconstruction, private CUDA-JS path, CUDA/PTX/native Tensor implementation, activation/model vocabulary or lower-precision widening was introduced.

## Active dependency lane — first real model / #22 and Vector #3

Protected UCI-Arena-Vector #3 / PR #17 froze the LatticeKnight-4M operation inventory against an older exact Tensor implementation and demonstrated that the old erf/gather/concat gaps were closed while `unary:tanh` remained the sole mathematical gap. A test-only tanh capability snapshot moved that older verifier to `VECTOR_MODEL_WORKSPACE_UNRESOLVED`; that result was useful dependency evidence, not proof of the current exact pair.

Now that #61 is protected-complete, Vector #3 must rebind its capability snapshot to protected Tensor merge `3a62bc47017aa10198eb1640b66f6b71a608b562` and rerun the frozen real-model gate. Do **not** promote the earlier predicted workspace blocker until that exact-pair rerun reproduces it. Vector retains model/package/checkpoint, chess/head and product numerical-oracle meaning.

Tensor #22 owns the resulting generic complete TensorProgram/TensorPlan coverage, workspace/resource and callable/oracle facts once the refreshed Vector evidence reaches that boundary. Any newly demonstrated generic Tensor gap routes back to CUDA-JS-Tensor; any missing consumer-neutral CUDA mechanism routes to CUDA-JS. CUDA-MCGS #124 consumes only the final generic callable/resource facts while retaining evaluator/search lifecycle ownership.

CUDA-MCGS #123 remains a parallel evaluator-free public-package falsifier and does not authorize Tensor or product semantics in CUDA-MCGS.

## Qualification limits

Portable/repository/package evidence does not promote native/provider support. Prior native evidence belongs only to its exact historical pair. No performance recommendation, broad provider accuracy/ULP claim, Tensor-Core claim, lower-precision tanh claim or production-stability claim exists.

## Current-state governance

Protected `STATUS.md` and `next_step.yaml` own the live Tensor execution seam. Issues own durable obligations/evidence, not a live SHA timeline. Historical branches and PRs are research/evidence only unless requalified against current protected authority.

`the_restaurant` implementation remains deferred; its retained integration plan is documentation only.
