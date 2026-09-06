# SPEC-0013: Optional CUDA-MM Physical-Policy Composition

**Status:** Accepted

**Version:** 1.0.0

**Owner:** CUDA-JS-Tensor

**Related authorities:** `iteathen/CUDA-JS` SPEC-0032/SPEC-0033; `iteathen/CUDA-MM` SPEC-0001

## Purpose

Recognize CUDA-MM as the reserved owner for reusable cross-domain physical memory-management policy while preserving Tensor's ownership of Tensor mathematics, semantic liveness, material schedules, alias constraints, item workspace and Tensor-specific planning.

This specification selects no production CUDA-MM dependency and does not move the current Tensor arena/workspace implementation merely because CUDA-MM exists.

## Tensor ownership retained

CUDA-JS-Tensor continues to own:

- Tensor material/value identity;
- dtype/shape/layout/stride/view/alias semantics;
- TensorProgram/TensorPlan meaning;
- semantic material liveness and observability;
- Tensor-specific allocation requirements and workspace semantics;
- item-axis/callable ABI/workspace meaning;
- Tensor-specific provider/backend eligibility and fallback;
- Tensor numerical/reference evidence.

Tensor computes these facts. CUDA-MM must not infer them.

## Optional CUDA-MM projection

Only after CUDA-MM #3/#4 positively activate a bounded production contract, Tensor may optionally project domain-neutral facts such as:

- opaque material/resource identity;
- bytes and alignment;
- Tensor-computed lifetime constraints;
- read/write/access compatibility;
- alias/reuse restrictions;
- persistence/movability permissions explicitly derived from Tensor semantics;
- allowed physical placement classes;
- device-affinity and finite-budget constraints.

A CUDA-MM physical plan may return generic allocation/subrange/reuse/placement decisions and pressure/rejection explanations. Tensor remains responsible for proving that the projected constraints faithfully represent Tensor semantics.

## Boundary test

The composition is invalid if CUDA-MM needs Tensor operation names, node kinds, activations, gradients, shape algebra, item-axis semantics or other Tensor vocabulary to make its generic physical policy decision.

A Tensor-specific arena/planner may remain Tensor-owned when its decisions are inseparable from Tensor semantic scheduling. Shared physical policy moves to CUDA-MM only when it survives materially different consumers unchanged.

## Lower native boundary

CUDA-JS remains the sole native CUDA/provider owner. CUDA-MM and Tensor use only public CUDA-JS mechanisms; neither may maintain native allocation/managed/P2P/prefetch/provider code.

## Optional dependency and deletion

CUDA-MM is optional. Deleting CUDA-MM leaves Tensor semantically complete and leaves the explicit/public CUDA-JS realization path available. Deleting Tensor leaves CUDA-MM's generic contract coherent.

## Current critical path

This addendum does not alter the frozen LatticeKnight FP32 numerical-oracle path, Tensor #22, CUDA-MCGS #124, mixed-precision work or any current native qualification gate.

## Non-goals

No immediate migration of SPEC-0008/Tensor arena work, no generic allocator API in Tensor, no CUDA-MM production activation, no automatic managed-memory placement, no semantic-liveness transfer and no performance claim.
