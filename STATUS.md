# CUDA-JS-Tensor Status

**Updated:** 2026-10-08

```text
repository:                 public protected pre-release
package name:               cuda-js-tensor (reserved by intent, unpublished)
package version:            0.1.0-alpha.8 (publication-guarded candidate; independent review pending)
current CUDA-JS dependency: cuda-js@0.1.0-alpha.21 at 2bff226b752d3c0af8b9185274d411e5990008d4
native qualification:       current candidate Windows single-DAG Node 26.7.0 and resident sequence Node 26.11.1 receipts
performance claims:         none
current generic Tensor gap: whole-plan prepared bounds addressed by explicit resident-sequence candidate
product numerical oracle:   UCI-Arena-Vector-owned and independent of Tensor readiness for other consumers
```

## Stable ownership

CUDA-JS-Tensor owns generic Tensor mathematics, dtype/shape/layout/stride semantics, TensorProgram/TensorPlan normalization, material/liveness/workspace meaning, deterministic Tensor Device-JS generation and item-axis callable semantics. CUDA-JS owns consumer-neutral provider/resource/compiler/runtime/native mechanisms. CUDA-MCGS owns evaluator request/batch/scatter/publication/search lifecycle. Model/checkpoint/chess/head/product numerical meaning remains downstream.

## Protected generic callable/resource record

The first frozen real model demonstrated complete current public Tensor operation/spec/callable/resource coverage without a new generic Tensor or CUDA-JS gap. The relevant consumer-neutral record remains:

- mixed TensorProgram contract `SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1+SPEC-0011-tanh-v1`;
- device-callable contract `SPEC-0009-item-parallel-device-tensor-program-v1+SPEC-0009-gather-concat-v1`;
- per-item workspace 33,194,524 bytes;
- item capacity 2 requires 66,389,048 bytes, below the accepted 67,108,864-byte default ceiling;
- item capacity 3 requires 99,583,572 bytes and fails closed with `TENSOR_DEVICE_WORKSPACE_LIMIT` before compiler work.

Those are Tensor facts. They are not checkpoint numerical parity, evaluator readiness, native/provider support or a performance claim.

## Product oracle boundary

UCI-Arena-Vector owns the independent checkpoint-bound policy/value numerical oracle for its frozen model. Tensor must not manufacture expected product outputs or absorb model/checkpoint/head meaning to close that product gate.

That product oracle is **not** a prerequisite owned by Tensor for CUDA-MCGS #124 or any other unrelated consumer. Tensor publishes its generic public facts and qualification; each consumer owns whether those facts are sufficient for its own work. If a consumer later demonstrates a genuine generic Tensor math/item/ABI/workspace defect, route only that defect here. A lower compiler/runtime defect routes to CUDA-JS.

This ownership correction changes no Tensor mathematics, callable ABI, workspace rule or public package behavior.

## Current work discipline

Issue #22 retains the protected generic real-model callable/resource record. No proactive Tensor mutation is justified while the downstream product oracle is pending. New Tensor work requires a concrete consumer-backed generic gap or separately accepted bounded Tensor capability.

An exact consumer-selected CUDA-JS alpha.21 runtime demonstrated `TENSOR_CUDA_JS_INCOMPATIBLE` against Tensor's alpha.18 package projection. The bounded alpha.7 refresh selects that exact public cohort and preserves strict version checks, Tensor mathematics, callable ABI, and workspace rules. This interoperability gap is independent of downstream product numerical evidence; it adds no product semantics or performance claim.

The installed-package native conformance runner passed on exact Node 26.7.0 / Windows x64 / recorded cc-7.5 device with CUDA-JS alpha.21 and cuBLASLt 13.5.1. It covers dense SIMT and mixed execution, fused special values, fixed-tree reduction, public item callable outputs and out-of-range guards, and graceful session/runtime cleanup. [Current receipt](conformance/native/receipts/2026-10-08-alpha7-alpha21-node267-win32.json) qualifies only that invocation profile; it adds no product numerical or performance claim.

Alpha.8 adds an explicitly selected candidate `execution: 'resident-sequence'` host qualification profile. A whole generic plan exceeding one prepared DAG is split into ordered resident chunks, each obeying CUDA-JS's unchanged 32-node/64-binding bounds. Original plan material layout and math remain unchanged; the default single-DAG profile and its identities remain intact. The [candidate addendum](docs/specs/SPEC-0005-resident-sequence-addendum.md) requires independent review before integration. [Native candidate evidence](conformance/native/receipts/2026-10-08-alpha8-resident-sequence.json) covers 80-kernel/81-binding cross-chunk views and binding-driven cuts at 105 global bindings, two replays, and zero-live/orphaned terminal resources on Node 26.11.1/Windows/compute_75. The unchanged single-DAG native fixture also passed on Node 26.7.0. Host sequence qualification is not a device-closed engine inference profile or product numerical evidence.

`the_restaurant` remains deferred. Multi-device Tensor semantics, NN/training/autodiff, publication and performance recommendations remain independently gated.

## Qualification limits

Portable/repository/package evidence does not promote native/provider support. Historical physical evidence belongs only to its exact historical pair. Consumer readiness is not authoritative merely because a sibling repository repeats it.
