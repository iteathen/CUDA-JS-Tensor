# SPEC-0008: Consumer-backed erf, static gather and concat semantics

**Status:** Candidate for review — no implementation authority
**Date:** 2026-09-01
**Parent:** SPEC-0004
**Issue:** #32
**Consumer evidence:** `iteathen/UCI-Arena-Vector#3`, PR #16
**Upstream mechanism dependency:** `iteathen/CUDA-JS#157` for public Device-JS `erf`

## Outcome

Define the smallest backend-neutral TensorProgram semantic extension proven necessary by the first frozen real downstream model:

1. add `erf` to the existing `unary` operator family;
2. add one bounded **static indexed gather** operation whose complete index set is canonical program metadata and therefore range-checkable before execution;
3. add one finite ordered `concat` operation over an explicit axis.

This candidate adds no model, neural-network, chess, policy-head, underpromotion, Restaurant or CUDA mechanism meaning to CUDA-JS-Tensor. The downstream model is evidence that these generic operations are needed; it is not specification authority for their semantics.

This document is deliberately **not accepted implementation authority**. SPEC-0004 remains the accepted operation catalog until this candidate is independently reviewed and accepted through the repository's normal specification process.

## Design disposition

The strongest credible failure is satisfying one consumer by smuggling its shape/mapping conventions into Tensor, or by adding a nominally generic gather whose runtime index failures require a new host callback or hidden synchronization path.

The selected first profile is therefore conservative:

- `erf` is ordinary materialized unary mathematics owned by Tensor semantics but requires a separately accepted public Device-JS helper from CUDA-JS #157 for the complete SIMT realization;
- `gather` carries a finite static `indices` list in the canonical TensorProgram node, so every index is validated against capacity before execution and no runtime out-of-range protocol is invented;
- `concat` is a finite materializing operation with exact dtype/rank/shape and active-extent rules;
- no in-place variant, runtime index tensor, scatter, dynamic concatenation, padding convention or consumer-specific fused form is admitted.

This keeps LEGO ownership intact: Tensor owns tensor mathematics/indexing; CUDA-JS owns generic Device-JS numeric mechanism; consumers own model/application meaning.

## Candidate operation catalog extension

If accepted, SPEC-0004's functional operation model gains only the following additions:

| `op` | Inputs | Required `options` / defaults | Result |
|---|---:|---|---|
| existing `unary` | 1 | `operator: "erf"` | materialized elementwise error function |
| `gather` | 1 | `axis`, `indices` | materialized static indexed selection |
| `concat` | 2..256 | `axis` | materialized ordered concatenation |

All three outputs have offset zero, row-major contiguous strides and read-write access. Inputs require read authority. Each result receives a distinct material alias class and participates in the same liveness/allocation planning rules as existing SPEC-0004 material nodes.

Unknown fields, wrong arity, unsupported dtypes, invalid axes, invalid static indices, incompatible shapes, incompatible active extents and unsafe arithmetic reject during TensorProgram normalization before backend/native work.

## `unary: erf`

### Type and shape

`erf` admits floating dtypes only:

```text
f16 bf16 f32 f64
```

The output has the same dtype, capacity shape and active-axis identity as the input, but is a new materialized contiguous value.

### Mathematical semantics

For each logical element `x`, the mathematical result is the Gaussian error function:

```text
erf(x) = (2 / sqrt(pi)) * integral_0^x exp(-t^2) dt
```

The operation is semantic, not an implementation recipe. A backend may use an accepted provider helper, but may not substitute a different function merely because it is faster or easier to lower.

The mathematical result is rounded to the declared output dtype under the same per-operation floating result rule already owned by SPEC-0004. No fast-math reassociation or consumer-selected approximation is implied.

Special values are:

- NaN propagates to NaN;
- `erf(+Infinity) = +1`;
- `erf(-Infinity) = -1`;
- positive zero remains positive zero;
- negative zero remains negative zero.

These rules are part of Tensor semantic identity and backend equivalence.

### Upstream realization gate

Current accepted CUDA-JS SPEC-0030 exposes `sqrt`, `log`, `exp`, `abs`, classification and min/max helpers but does not expose `erf`. CUDA-JS issue #157 owns the consumer-neutral Device-JS numeric mechanism.

Acceptance of this Tensor spec would authorize Tensor semantics only. **Production SIMT/device-callable implementation of `unary:erf` remains blocked until a compatible public CUDA-JS `erf` contract is accepted and available.** CUDA-JS-Tensor must not deep-import CUDA-JS source, expose CUDA intrinsic/header names, embed private CUDA source or add native/FFI code to bypass that gate.

## Static indexed `gather`

### Canonical record

The first profile is:

```text
{
  op: "gather",
  inputs: [source],
  options: {
    axis: <integer>,
    indices: [<integer>, ...]
  }
}
```

`axis` is normalized to one nonnegative axis index in `[0, rank)`; negative-axis shorthand is not admitted in this first profile.

`indices` is copied into canonical JSON-safe program metadata. It is an ordered finite array of nonnegative safe integers. Empty `indices` is valid and produces a zero-sized gathered axis. Duplicates are valid and repeat the selected source element in the corresponding output position.

The first candidate pressure limit is:

```text
maxStaticGatherIndices = 65536
```

This is a **profile pressure bound**, not mathematical meaning. It bounds canonical metadata and normalization memory rather than limiting tensor dimensions generally. A later accepted profile may raise or replace this bound without changing gather semantics. The bound must appear in canonical program limits/compatibility identity when implementation is accepted.

Every index must satisfy:

```text
0 <= index < source.capacityShape[axis]
```

Because all indices are static canonical metadata, out-of-range selection rejects during program normalization. There is no device-time index failure, host callback, hidden synchronization or clamping semantics in this profile.

### Shape

Output rank equals source rank. Output capacity shape copies every source dimension except:

```text
output.capacityShape[axis] = indices.length
```

The output element at coordinate `o` reads the source coordinate obtained by replacing `o[axis]` with `indices[o[axis]]`. Index order is observable and preserved exactly.

No implicit dtype conversion occurs. Output dtype equals source dtype.

### Active extent

If the source has no `activeAxis0`, ordinary gather semantics apply on any axis.

If the source has `activeAxis0`:

- gathering `axis === 0` rejects in the first profile because a static capacity index list cannot generally prove validity against every runtime logical extent;
- gathering any `axis !== 0` preserves the source `activeAxis0` maximum and current extent unchanged.

This restriction avoids fabricating runtime index validity or changing the one-active-axis contract. A later profile may generalize active-axis gather only with explicit accepted semantics.

### Aliasing and materialization

`gather` is always materialized. It never returns a strided view, even when the indices happen to form an identity or monotonic slice. This avoids data-dependent/view-subset alias proofs and keeps the first operation independently replaceable. Existing `slice` remains the efficient view operation for positive-step contiguous/index-regular cases.

## Ordered `concat`

### Canonical record

The first profile is:

```text
{
  op: "concat",
  inputs: [first, second, ...],
  options: { axis: <integer> }
}
```

The operation requires from 2 through 256 source values. The 256-source limit is an explicit first-profile per-node pressure bound and must be present in canonical program limits/identity when implementation is accepted. Larger compositions can be represented by multiple concat nodes without changing mathematical meaning.

`axis` is one nonnegative axis index in `[0, rank)`.

### Compatibility and shape

Every input must have:

- the same dtype;
- the same rank;
- identical capacity dimensions on every axis other than `axis`.

The output dtype/rank are unchanged. The output capacity dimension on `axis` is the checked finite sum of all input capacity dimensions on that axis. Overflow, unsafe-integer arithmetic or an output specification outside ordinary TensorSpec/session bounds rejects.

Input order is semantic. The output coordinate space is partitioned on `axis` in exact input-list order. No sorting, set semantics, deduplication or backend reordering is permitted.

`concat` is always materialized contiguous output and never aliases an input.

### Active extent

If all inputs have no `activeAxis0`, concat is admitted on any valid axis.

If any input has `activeAxis0`, the first profile requires **all** inputs to have `activeAxis0` with identical maximum and current extent.

For active inputs:

- `axis === 0` rejects in this first profile; dynamic axis-0 concatenation would require a separately accepted extent-sum/runtime-layout contract;
- `axis !== 0` preserves the identical active-axis maximum and current extent on output axis 0.

Mixed active/inactive inputs reject even when their current logical lengths happen to match. This keeps compatibility identity structural rather than value-coincidental.

## TensorPlan and backend obligations

If accepted, the three operations integrate into the existing SPEC-0004 planning boundary:

- `erf`, `gather` and `concat` are material nodes with exact definitions, last use, distinct allocation and alignment facts;
- no hidden workspace is authorized by their semantic records;
- the first complete backend remains the existing SIMT path unless a separately accepted exact accelerator profile replaces an eligible node;
- unsupported accelerated shapes/dtypes must fall back to the complete semantic path or reject according to accepted backend authority; they may not compute different mathematics.

Static gather and concat require only existing generic Device-JS pointer/index/arithmetic mechanisms for a complete SIMT lowering. `erf` additionally requires the public CUDA-JS #157 mechanism described above.

SPEC-0005/SPEC-0009 execution lifecycle, item isolation, cancellation, session/device compatibility and cleanup ownership remain unchanged. This candidate introduces no scheduler, queue, host callback, runtime allocation growth or native resource owner.

## Canonical compatibility and deletion

If accepted:

- the new operation/operator records participate in TensorProgram canonical identity exactly like existing operations;
- programs not using SPEC-0008 operations retain their previous canonical contract/identity and behavior;
- a future contract-version change must make use of this extension explicit rather than silently reinterpret an old SPEC-0004-only program;
- deleting the first consumer must leave only generic Tensor/CUDA-JS tests and no model/application vocabulary in production owners.

The accepted implementation must define the exact additive program contract/version transition before code is merged. This candidate does **not** silently mutate `SPEC-0004-tensor-program-v1` in place.

## Falsifiers required before acceptance

### `erf`

- integer/bool input rejects;
- output shape/dtype/active extent are preserved exactly;
- NaN, infinities and signed zeros match the declared semantics;
- a backend/tanh approximation cannot satisfy `erf` equivalence;
- absence of an accepted public CUDA-JS realization cannot be bypassed by private source/native code.

### `gather`

- unknown options, negative/noninteger axis, axis >= rank reject;
- more than 65,536 indices reject as pressure;
- negative, noninteger, unsafe or out-of-capacity indices reject before execution;
- empty list produces a zero-sized gathered axis;
- duplicates and nonmonotonic order are preserved;
- active-axis-0 gather rejects; non-axis-0 gather preserves active extent;
- result is materialized and does not inherit source alias class;
- canonical round-trip preserves index order and exact values.

### `concat`

- fewer than 2 or more than 256 inputs reject;
- dtype/rank/non-axis shape mismatch rejects;
- invalid axis and unsafe axis-sum arithmetic reject;
- input order changes canonical identity and output order;
- active-axis concat on axis 0 rejects;
- mixed active/inactive or differing active extents reject;
- compatible non-axis concat preserves active extent exactly;
- result is materialized with no input aliasing.

### Cross-operation / deletion

- existing SPEC-0004-only programs retain previous identities/round trips;
- all three operations remain consumer-neutral under first-consumer deletion;
- static plan resource/liveness accounting remains finite;
- complete SIMT lowering is independent of optional accelerators;
- no private CUDA-JS/native mechanism appears in public Tensor products.

## Consumer evidence — non-normative

UCI-Arena-Vector PR #16 freezes LatticeKnight-4M against protected `cuda-js-tensor@0.1.0-alpha.6@44376e151ab854c81d65df79db1717478ae8ce5b` and reports exactly these missing capabilities:

```text
concat
gather
unary:erf
```

The consumer uses static model-owned index maps, which is why static preflightable gather is sufficient for the first generic profile. This paragraph explains the evidence trail only; no model shape, policy mapping or application semantics are part of this specification.

## Acceptance gate

This candidate may become implementation authority only after independent review confirms:

1. the operation set is the smallest reusable coverage of the proven consumer gap;
2. the static gather choice avoids unnecessary runtime failure/synchronization authority;
3. the pressure limits are explicit profile limits rather than hidden semantic restrictions;
4. active-extent behavior is complete and conservative;
5. public CUDA-JS #157 is an explicit `erf` implementation dependency rather than a private workaround;
6. the additive TensorProgram contract/version transition is explicit;
7. required conformance/failure/deletion/cleanup evidence is specified before production code.

Until that acceptance, production TensorProgram code must continue rejecting `erf`, `gather` and `concat` exactly as it does today.

## Non-goals

Runtime index tensors, scatter/update, masks, negative indices, wrap/clamp indexing, dynamic gather failure protocols, dynamic axis-0 concat, stack, split, arbitrary advanced indexing, in-place variants, fused GELU, model-specific activation names, policy heads, chess mappings, generated CUDA source exposure, native Tensor code, private CUDA-JS imports, performance claims or accelerator-specific semantics.