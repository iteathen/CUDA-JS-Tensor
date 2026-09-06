# SPEC-0012: Native Boundary and JavaScript/TypeScript Implementation

**Status:** Accepted

**Version:** 1.0.0

**Owner:** CUDA-JS-Tensor

**Lower authority:** `iteathen/CUDA-JS` SPEC-0032

## Purpose

This specification fixes CUDA-JS-Tensor's implementation boundary relative to the CUDA-JS native layer.

CUDA-JS-Tensor owns generic Tensor mathematics, TensorProgram/TensorPlan semantics, Tensor execution planning, item-axis semantics, callable ABI/workspace meaning, provider eligibility and Tensor numerical evidence. It does not own native CUDA/provider integration.

## Implementation-language rule

Maintained CUDA-JS-Tensor repository source is JavaScript/TypeScript. Restricted Device-JS source or deterministic Device-JS generation is permitted only through accepted public CUDA-JS language/compiler contracts.

CUDA-JS-Tensor does not maintain:

- C or C++ host implementation;
- CUDA C++ or hand-authored PTX;
- N-API/native addons;
- direct native FFI;
- CUDA Driver/provider bindings;
- native handles, pointers, ABI structs or platform discovery code.

A missing native mechanism is a lower-layer CUDA-JS gap, not permission for a Tensor-local native path.

Independent native evidence may be produced outside this repository and recorded against an exact profile. Repository-maintained numerical/reference code remains JavaScript/TypeScript and must be independent of the Tensor execution implementation it tests.

## Tensor-owned abstraction

Tensor may own high-level semantics and planning that are meaningful without CUDA, including:

- dtype, shape, layout, stride, view, alias and active-extent meaning;
- TensorProgram operation semantics and TensorPlan validation;
- mathematical broadcast, reduction, matmul, gather, concat and other Tensor operations;
- exact accumulation/order/rounding policy where specified;
- fusion semantics;
- material liveness and semantic observability;
- item independence, callable parameter meaning and per-item workspace semantics;
- mapping Tensor semantics to an eligible public lower provider/mechanism;
- backend-neutral CPU/reference realization in JavaScript/TypeScript when separately accepted.

These semantics do not move down merely because CUDA-JS realizes them.

## Lower mechanism boundary

CUDA-JS owns the native mechanisms consumed by Tensor, including as selected:

- device/context selection and compatibility;
- allocations, views, transfers, mapped/managed/peer memory mechanisms and native memory pools;
- compiler/linker/artifact/module/function machinery;
- streams/events/operations/prepared execution/CUDA Graph realization;
- Device-JS lowering and native device artifacts;
- cuBLASLt and other native provider resources/operations;
- native errors, health, leases and teardown.

Tensor may choose among accepted mechanisms only as part of explicit Tensor backend/eligibility policy. It does not reproduce their native validity or lifecycle rules.

## Memory-planning boundary

Tensor owns semantic liveness, alias constraints, material schedules and Tensor-specific allocation requirements.

Tensor does not own generic native allocation APIs or native memory placement mechanisms. It also must not absorb broadly reusable physical memory-management strategy merely because Tensor is the first consumer.

For example, Tensor may state that two materials have non-overlapping lifetimes and particular size/alignment/access requirements. A Tensor-specific arena may remain here when its plan is inseparable from Tensor semantic/material scheduling. If a later physical planner can consume generic lifetime/size/alignment/access/placement constraints unchanged across materially different domains, that reusable policy requires its own natural JavaScript/TypeScript owner and must consume public CUDA-JS rather than native APIs.

No new memory-management repository is authorized by this specification.

## Provider and optimization boundary

Tensor owns mathematical eligibility, equivalence, fallback and Tensor-visible backend identity. CUDA-JS owns native provider execution.

Tensor may not select or emulate a provider by inspecting private native facts. Lower capability/resource requirements must come from public CUDA-JS contracts.

Performance policy remains evidence-driven and Tensor-specific where it depends on shapes, dtypes, layouts or mathematical semantics. Generic native scheduling/graph/stream mechanisms remain CUDA-JS-owned.

## Numerical evidence boundary

Generic Tensor numerical parity belongs here. Concrete model/checkpoint/head expected outputs remain product-owned.

Portable reference implementations and oracles maintained in this repository are JavaScript/TypeScript only. They must not reuse the generated Device-JS implementation, provider lowering, fused schedule or subject intermediate values as their reference path.

External native evidence may supplement exact native/provider qualification but cannot become repository-maintained native source or substitute for product-specific numerical authority.

## Supersession

This specification is additive over every accepted CUDA-JS-Tensor specification. Existing Tensor semantic contracts remain unchanged except that any ambiguous native-implementation escape is closed by this specification and CUDA-JS SPEC-0032.

Historical native evidence remains provenance; new maintained Tensor source and successor specifications must obey this boundary.

## Non-goals

No native Tensor backend, no Tensor-local CUDA/provider FFI, no generic GPU memory manager, no model/NN/search semantics, no automatic hidden backend fallback, and no support/performance promotion from this ownership rule alone.
