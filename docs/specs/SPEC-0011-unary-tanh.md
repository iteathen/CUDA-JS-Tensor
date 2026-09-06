# SPEC-0011: Consumer-backed unary tanh semantics

**Status:** Accepted

**Date:** 2026-09-05

**Parent:** SPEC-0004

**Issue:** #61

**Consumer evidence:** `iteathen/UCI-Arena-Vector#3`, protected PR #17 / merge `67b2512794c4389abdea22e7f353dac712f6c03d`

**Upstream mechanism dependency:** public CUDA-JS `SPEC-0030-tanh-v1`, protected merge `d1a8edef5bd06c402a5c14c8945269f206520174`

## Outcome

Add exactly one consumer-neutral TensorProgram mathematical operation to the existing `unary` family:

```text
{ op: "unary", inputs: [source], options: { operator: "tanh" } }
```

The first profile admits exactly `f32` and `f64`. It preserves dtype, capacity shape and active-axis meaning, produces one distinct materialized contiguous read-write output, and uses the existing TensorProgram/TensorPlan material/liveness owners.

This specification adds no neural-network layer, activation-layer, model, chess, evaluator, search, batching, scheduler, provider, native or performance semantics. The frozen LatticeKnight model is evidence that generic `tanh` mathematics is required; it is not semantic authority for Tensor.

## Assessment and ownership

Protected UCI-Arena-Vector #3 proves that, after protected Tensor `erf`, static gather and ordered concat support, its complete frozen-model operation inventory has exactly one uncovered mathematical operation: `unary:tanh`. A test-only capability snapshot containing `tanh` advances that consumer to its TensorProgram/TensorPlan workspace gate.

SPEC-0004 owns the existing unary mathematical family but its exact `SPEC-0004-tensor-program-v1` identity admits only `neg`, `abs`, `exp`, `log` and `sqrt`. Silently adding `tanh` under that identity would change accepted language semantics without changing compatibility identity. SPEC-0010 already establishes the additive child pattern for new operation-catalog semantics. The smallest correct owner is therefore this additive SPEC-0011 child.

CUDA-JS separately owns the lower restricted Device-JS scalar mechanism. Protected CUDA-JS `SPEC-0030-tanh-v1` exposes same-kind `gpu.math.tanh(x)` for `f32`/`f64` and private ordinary `tanhf`/`tanh` lowering. Tensor consumes that public contract; it does not reproduce CUDA provider/header/cache or numerical-provider ownership.

## Mathematical semantics

For every logical element `x`, `unary:tanh` computes the real hyperbolic tangent:

```text
tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
```

The equation defines the mathematical function, not an implementation recipe. A Tensor backend must not realize this contract by substituting an exp identity, fast approximate intrinsic, GELU approximation or consumer-specific formula merely because it is available.

The first accepted dtype table is:

| Input | Output | Status |
|---|---|---|
| `f32` | `f32` | admitted |
| `f64` | `f64` | admitted |
| `f16` | — | rejected |
| `bf16` | — | rejected |
| integer | — | rejected |

No implicit cast, widening or narrowing occurs.

Special values are exact semantic requirements:

- `tanh(+0) = +0`;
- `tanh(-0) = -0`;
- `tanh(+Infinity) = +1`;
- `tanh(-Infinity) = -1`;
- `tanh(NaN)` returns NaN.

Finite results are represented in the declared output dtype under the existing Tensor floating-result policy. Tensor does not invent a provider-independent correctly-rounded, bit-identical or global-ULP guarantee. Resolved realization consumes the exact public CUDA-JS `SPEC-0030-tanh-v1` provider-bound semantics and may not claim more accuracy than that selected lower profile proves.

## Shape, access and materialization

The operation requires exactly one readable Tensor input. Its output has:

- the same dtype;
- the same capacity shape;
- the same `activeAxis0` maximum/current extent when present;
- offset zero and contiguous row-major strides;
- read-write access;
- a distinct material alias class.

The canonical normalized options are:

```text
{
  operator: "tanh",
  arithmetic: "round-to-output-dtype-v1",
  specialValues: "ieee-nan-propagate-signed-zero-v1"
}
```

Unknown fields, wrong arity and unsupported dtypes reject during TensorProgram normalization before backend/compiler work.

## Additive TensorProgram compatibility

This child must preserve every pre-SPEC-0011 exact contract and canonical record.

A program using neither SPEC-0010 nor SPEC-0011 remains exactly:

```text
SPEC-0004-tensor-program-v1
```

with the existing base limits:

```text
{ maxInputs: 256, maxNodes: 4096, maxOutputs: 256 }
```

A program using SPEC-0010 operations but no `tanh` remains exactly:

```text
SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1
```

with the existing SPEC-0010 limits unchanged.

A program using `unary:tanh` but no SPEC-0010 operation selects exactly:

```text
SPEC-0004-tensor-program-v1+SPEC-0011-tanh-v1
```

and retains the base limits because this child adds no new metadata/resource pressure limit.

A program using both SPEC-0010 and `unary:tanh` selects exactly:

```text
SPEC-0004-tensor-program-v1+SPEC-0010-erf-gather-concat-v1+SPEC-0011-tanh-v1
```

and carries the existing SPEC-0010 limits unchanged.

Child order is canonical: SPEC-0010 before SPEC-0011. Requirements are derived only from normalized semantic use. Callers cannot assert a child contract independently of the operation graph. Rehydration must reconstruct semantics and reject any contract/limits/operation mismatch, including forged, omitted, reordered or unnecessary children.

The existing TensorProgram compatibility-identity owner remains sufficient; no second child registry or identity manager is authorized.

## Ordinary resolved-SIMT realization

Once the exact public CUDA-JS tanh dependency is selected by the package, ordinary SIMT lowering must map this Tensor operator only through:

```text
gpu.math.tanh(value)
```

No private CUDA-JS import, CUDA C++/PTX/native code or provider symbol is permitted in Tensor. Existing Device-JS/source/AST/binding/workspace pressure remains lower execution admission, not Tensor mathematical rejection.

The existing exact elementwise-fusion profile may fuse `unary:tanh` only when its already accepted material-observability, dtype, shape and exactness criteria admit the node. Fusion does not change this operation's semantics or contract identity; the complete unfused SIMT realization remains available.

## Device-callable SPEC-0009 relation

SPEC-0009 already owns the generic item rule:

> `copy`, `cast`, `contiguous` and `unary` preserve an item-varying input.

Adding a new accepted unary operator therefore does **not** create a new item-axis, callable ABI, output layout, workspace or scheduler semantic. No SPEC-0009 child is required for tanh.

Device-callable support remains an implementation/evidence claim. The existing device-item unary generator must explicitly map `tanh` through public `gpu.math.tanh`, and the resulting item program must pass the existing SPEC-0009 classification, workspace isolation, out-of-range no-write, package-composition and cleanup evidence before support is claimed.

The TensorDeviceProgram contract does not gain a tanh-specific item child merely because its underlying TensorProgram compatibility identity contains SPEC-0011; the program/plan identity and copied CUDA-JS library semantic identity already bind the changed mathematics.

## Dependency and package rule

Production implementation may proceed only against an exact protected CUDA-JS revision that contains accepted public `SPEC-0030-tanh-v1`. The current qualifying lower input is:

```text
cuda-js@0.1.0-alpha.18
d1a8edef5bd06c402a5c14c8945269f206520174
```

Updating the package dependency is part of the implementation transaction, not this authority-only acceptance. Until that implementation integrates, the current protected Tensor package remains bound to its earlier exact CUDA-JS revision and must not claim tanh support.

## Required evidence

Portable/software/package implementation evidence must prove at least:

- f32 and f64 TensorProgram inference, normalization and round-trip rehydration;
- f16/bf16/integer rejection;
- exact base, SPEC-0010-only, SPEC-0011-only and SPEC-0010+SPEC-0011 contract/limit selection;
- forged/mutated/reordered/unnecessary child rejection;
- byte/canonical/compatibility identity preservation for representative pre-tanh base and SPEC-0010 programs;
- ordinary unfused SIMT lowering through public `gpu.math.tanh` only;
- exact-fusion behavior under the existing fusion owner, with unfused fallback retained;
- device-callable item classification and lowering through the existing SPEC-0009 unary rule;
- installed-package consumption through the public Tensor and CUDA-JS surfaces;
- no private/native escape path, hidden allocation, new scheduler or host-progress mechanism.

Native/provider numerical support remains separately evidence-gated. Portable generated-source/package tests are not native CUDA qualification.

## Non-goals

`f16`/`bf16` tanh, activation-layer APIs, GELU, model semantics, autodiff/training, fast-math or approximate tanh, provider-independent finite error guarantees, new item semantics, cooperative execution, new workspace policy, Tensor-Core/provider acceleration, performance claims, CUDA-MCGS integration and product-specific adapters are outside this child.
