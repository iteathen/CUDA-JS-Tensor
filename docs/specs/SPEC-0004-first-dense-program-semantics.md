# SPEC-0004: First dense program semantics

**Status:** Accepted implementation profile
**Date:** 2026-08-26
**Parent:** SPEC-0002
**Issue:** #4

## Outcome

Define the first immutable, backend-neutral dense operation DAG and static plan without inventing execution facts that belong to a backend. This profile owns mathematical shape/dtype/layout meaning, dependency order, view/materialization classification, declared aliasing, liveness and bounded distinct-allocation planning.

`ResolvedTensorPlan` remains dependency-gated on `TENSOR-SIMT-012`: exact generated program identities, backend workspace, prepared execution products, session/device compatibility and cleanup graph cannot be truthfully resolved before a concrete executor exists.

## Construction

Canonical construction is:

```text
TensorProgram.create({
  inputs: [{ name, spec }],
  nodes: [{ id?, op, inputs, options }],
  outputs: [{ name, value }]
})
```

Nodes are declared in topological order and may reference only an input or earlier node. Names/IDs are bounded, unique strings. The convenience `TensorProgram.define(builder => outputs)` facade produces the same canonical record. The temporary builder is assembly only; the resulting program and every copied record are immutable.

The normalized `program.canonical` record is JSON-safe and round-trippable through `TensorProgram.create(program.canonical)`. Rehydration re-runs semantic inference and rejects any changed derived output spec, arithmetic policy, materialization or limit fact.

V1 admits at most 256 inputs, 4,096 nodes and 256 outputs. A concrete program may be smaller but never unbounded. Unknown fields, forward references, duplicate names/IDs, foreign builder values and asynchronous builder callbacks reject.

## Functional operation model

Every operation is functional. No node mutates an input or aliases a newly materialized output in place. View operations share the input alias class; materializing operations receive distinct planned storage. Runtime input capabilities may still alias and require exact binding-time validation in a later resolved plan.

The first node families are:

| `op` | Inputs | Required `options` / defaults | Result |
|---|---:|---|---|
| `fill` | 0 | `spec`, `value` | materialized requested read-write spec |
| `copy` | 1 | none | materialized contiguous copy |
| `cast` | 1 | `dtype` | materialized contiguous explicit cast |
| `reshape` | 1 | `capacityShape` | order-preserving view |
| `permute` | 1 | `axes` | stride-permuted view |
| `slice` | 1 | one null or `{start,length,step?}` per axis | positive-step bounded view |
| `broadcast` | 1 | `capacityShape` | trailing-axis finite broadcast view |
| `contiguous` | 1 | none | materialized row-major value |
| `unary` | 1 | `operator` | materialized elementwise value |
| `binary` | 2 | `operator` | materialized trailing-broadcast value |
| `reduce` | 1 | `operator`, optional axes/keepDimensions/accumulatorDtype/order/identity | materialized reduction |
| `matmul` | 2 | optional transposeA/transposeB/accumulatorDtype | materialized rank-2 or rank-3 product |

`copy`, `cast`, `contiguous`, arithmetic, reduction and matmul outputs have offset zero, row-major contiguous strides and read-write access. Inputs consumed by an operation require read authority. `fill` requires a read-write spec because its backend must initialize the output.

## Views and active extent

`reshape` requires contiguous non-broadcast element order and equal capacity element count. An active-axis tensor may only use an identity-capacity reshape in v1 because a general dynamic reshape cannot be represented by active axis 0 alone.

`permute` requires an exact permutation. If active axis 0 exists, it must remain output axis 0. `slice` uses positive steps and proves the final reached element is in capacity bounds. An active axis-0 slice must be the full capacity axis so the dynamic bound is preserved unchanged.

`broadcast` uses trailing-axis alignment: a source dimension must equal the target or be one. Added/broadcast axes receive stride zero and the result becomes read-only when multiple logical coordinates alias. An active input must retain rank and unchanged capacity axis 0.

## Elementwise arithmetic

No implicit dtype promotion occurs; binary inputs must have the same dtype and `cast` is explicit. Unary operators are `neg`, `abs`, `exp`, `log`, and `sqrt`. `neg` admits signed integer and floating dtypes; `abs` admits every dtype; `exp`, `log`, and `sqrt` admit floating dtypes only.

Explicit casts use round-to-nearest-even for floating destinations, modulo-width interpretation for integer-to-integer destinations, and saturating round-toward-zero for floating-to-integer destinations; NaN becomes zero and infinities clamp to the destination range. Integer-to-floating conversion uses round-to-nearest-even.

Binary operators are `add`, `sub`, `mul`, `div`, `minimum`, and `maximum`. All except `div` admit every dtype; `div` admits floating dtypes only. Integer add/sub/mul use modulo-width arithmetic, with `i32` interpreted as two's-complement. Floating elementwise results round to the declared output dtype after each operation. Floating operations propagate NaN; minimum of signed zeros is negative zero and maximum is positive zero. No fast-math reassociation is implied.

Elementwise shape inference uses finite trailing-axis broadcasting. Active extents must map to output axis 0 and, when both operands are active, have identical extent and maximum.

## Reductions

Reduction operators are `sum`, `product`, `minimum`, and `maximum`. Axes default to all axes and normalize to a sorted unique set. `keepDimensions` defaults false. Empty reductions return the declared identity.

Default accumulator dtype is `f32` for `f16`/`bf16` and otherwise the input dtype. Accepted widening is `u32 -> u64`, `f16/bf16 -> f32/f64`, and `f32 -> f64`; other changes reject. Output dtype equals accumulator dtype.

Default identities are zero, one, the dtype maximum/+infinity, and the dtype minimum/-infinity respectively. An explicit identity is normalized to the accumulator dtype. `order` defaults to `fixed-tree-v1`: adjacent row-major reduction elements form a balanced binary tree with identity padding. `backend-defined` is representable but compatibility-identity-affecting and does not authorize an exact/deterministic claim.

If active axis 0 is reduced it disappears. Otherwise it remains output axis 0 with the same maximum and current extent.

## Matrix multiplication

Matmul admits matching floating dtypes only. Rank must be two for both inputs or three for both. `transposeA` and `transposeB` apply to the final two dimensions. Contracting capacity dimensions must match.

Rank-3 batch capacity dimensions must match or one may broadcast. Two active batch axes require equal maximum and extent; one active batch axis propagates when the other batch is one or the same capacity. Rank-2 active extent is admitted only for the non-transposed A row dimension, where it propagates to output axis 0; active contraction or output-axis-1 dimensions reject.

Accumulator widening follows the reduction rule. Output dtype remains the input dtype after accumulation and one final rounding. This profile makes no tensor-core, reduced-precision, cuBLASLt or performance promise.

## Static TensorPlan

`TensorPlan.create(program)` records:

- exact topological operation schedule and observable outputs;
- each value's definition and last-use index;
- view alias inheritance, distinct material alias classes and unresolved runtime input aliasing;
- one explicit physical allocation per material node, including empty-tensor minimum storage, alignment and lifetime;
- no reuse, fusion or hidden workspace in this first planner;
- exact unresolved facts that prevent backend/session resolution.

The distinct-allocation policy is intentionally complete before optimization. Arena reuse and fusion require separate equivalence and lifecycle evidence.

## Falsifiers

Stop or narrow the operation when inference depends on runtime data, an active dimension cannot remain axis 0, alias subset cannot be proved, a scalar cannot be represented canonically, integer failure requires a hidden device-host callback, or backend facts would be guessed. No executor may reinterpret a rejected case as supported.
