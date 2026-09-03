# SPEC-0005 Addendum: CUDA-JS physical-boundary normalization

**Status:** Accepted

**Date:** 2026-09-03

**Owner:** `tensor.execution`

**Parents:** accepted `SPEC-0005-resolved-simt-execution.md`, `SPEC-0006-host-planned-cublaslt-matmul.md`, `SPEC-0006-provider-boundary-addendum.md`, and `SPEC-0009-item-parallel-device-callable-tensor-program.md`

**Issue owner:** #45

**Exact lower dependency:** `cuda-js@0.1.0-alpha.17` from protected `iteathen/CUDA-JS main@bc2700f2e5c654567c2e17bf8d67b882351b8681`

## Outcome

Normalize Tensor physical lowering around the final CUDA-JS ownership decisions without moving Tensor mathematics or selected physical-profile policy into CUDA-JS.

This addendum consumes three completed lower dispositions:

- CUDA-JS #180: no new higher-level executable-preparation transaction is justified; the existing compiler/artifact/module/function/provider/prepared-DAG capabilities are the correct composable LEGO pieces;
- CUDA-JS #181: explicit grid/block geometry remains the canonical expert launch contract; selected upper profiles may choose materially different physical geometry while CUDA-JS remains the final launch-validity owner;
- CUDA-JS #186 / SPEC-0008 capability-limit projection: accepted lower prepared-DAG and Device-JS parameter ceilings are now available through immutable public `CUDA_JS_COMPATIBILITY` records.

Tensor therefore keeps its semantic and selected-profile choices while deleting copied CUDA-JS compatibility facts.

## Tensor-owned physical-profile policy

The following remain Tensor-owned because they derive from the selected Tensor execution profile rather than from generic CUDA validity:

- the logical work-item count produced by Tensor lowering;
- the v1 `maxLogicalWorkItems = 2^32-1` bound used by the generated u32 work-item index profile;
- the resolved `blockSize` choice, including its current finite accepted set and default;
- `grid.x = ceil(workItems / blockSize)` and `block.x = blockSize` for the current one-dimensional Tensor SIMT profile;
- Tensor liveness, material/storage assignment, workspace sizing, access/dependency meaning and semantic node ordering;
- Tensor workspace ceilings and any independently justified stricter resource-count gates;
- Tensor Device-JS source generation and the accepted device-callable item ABI;
- the explicit Tensor device-library output family `"ptx" | "lto-ir"`, default `"ptx"`, because the selected artifact family is part of the Tensor device-callable compatibility identity and consumer composition contract.

CUDA-JS still validates the resulting launch geometry, source request, artifact family, resources and prepared DAG. Tensor owning a selected physical profile does not make Tensor the lower validity authority.

## Lower compatibility facts to consume

Tensor must no longer own numeric mirrors whose only rationale is equality with CUDA-JS ceilings.

For host-planned prepared execution, consume:

```text
CUDA_JS_COMPATIBILITY.capabilities.preparedOperationDagLimits = {
  nodes,
  edges,
  bindings,
  predecessorsPerNode
}
```

The historical Tensor `maxKernels: 32` and `maxBindings: 64` values are deleted as Tensor-owned limits. Where Tensor can reject predictable prepared incompatibility before expensive compilation, it uses the public lower values. CUDA-JS `prepareOperationDag()` remains the final validator for topology, edges, predecessor counts, binding sets, launch facts and resource hazards.

For device-callable Tensor programs, consume:

```text
CUDA_JS_COMPATIBILITY.capabilities.deviceJsLimits.parametersPerFunction
```

The historical Tensor `maxParameters: 64` mirror is deleted. Tensor still owns which parameters its item ABI requires and may impose an independently stricter semantic bound in a future accepted contract, but v1 does not rename the CUDA-JS ceiling as a Tensor fact.

For the retained expert artifact-family control, Tensor continues to accept only its existing `ptx | lto-ir` set. It must verify that the selected output is present in public `CUDA_JS_COMPATIBILITY.capabilities.compilerOutputFormats`; a future CUDA-JS format does not automatically widen the Tensor contract.

## Compatibility adapter

Tensor may centralize public CUDA-JS compatibility parsing in one private `tensor.execution` adapter module. That adapter must:

- import only `cuda-js/compatibility`;
- validate the exact lower fields required by Tensor and fail closed if they are missing or malformed;
- expose no native/provider handles or internal CUDA-JS objects;
- avoid copying numeric lower limits into new maintained Tensor constants;
- preserve the lower records as immutable compatibility facts used by Tensor planning/identity.

This is an adapter translation boundary, not a new Tensor registry or generic compatibility framework.

## Preparation and cleanup composition

The current host-planned CUDA-JS adapter remains structurally correct:

1. Tensor generates the semantic Device-JS functions required by the resolved Tensor plan;
2. public CUDA-JS compilation produces the selected artifact;
3. Tensor loads the module and obtains the declared functions through public CUDA-JS capabilities;
4. Tensor chooses any accepted optional provider plans required by its backend policy;
5. Tensor constructs prepared nodes from Tensor-owned dependencies/accesses and lower public capabilities;
6. public `prepareOperationDag()` validates and retains the complete lower executable DAG;
7. execution uses the existing CUDA-JS operation lifecycle.

No `PreparedExecutable`, `GpuProgram`, preparation transaction or private compiler/runtime shortcut is introduced.

Tensor continues to perform reverse dependency-order rollback/close for the lower resources it selected. This is not duplicated native lifecycle ownership: each resource's validity, leases, disposal effect, failure category and orphan truth remain CUDA-JS-owned. Tensor owns only the orchestration of resources it chose for one resolved Tensor plan.

## Identity and prerelease correction

Removing copied lower limits intentionally changes physical compatibility identities where those copied records previously participated. The corrected identity should represent:

- independently owned Tensor profile facts under Tensor-owned fields; and
- public lower compatibility facts under an explicitly lower/CUDA-JS compatibility record.

This is an accepted pre-1.0 compatibility correction. It does not change Tensor mathematical results or the TensorProgram/TensorPlan semantic identity.

Borrower counts, provider-native codes, native handles, deep-import identities and other lower private state never enter Tensor identity.

## Qualification

Portable/package qualification must prove:

- Tensor no longer defines `maxKernels: 32`, `maxBindings: 64`, or device-callable `maxParameters: 64` as lower-mirroring production constants;
- host-planned node/binding preflight consumes `preparedOperationDagLimits` through the public compatibility export;
- device-callable parameter preflight consumes `deviceJsLimits.parametersPerFunction`;
- Tensor `blockSize` and the existing work-items-to-grid/block mapping remain unchanged;
- `ptx` and `lto-ir` remain the exact Tensor output choices and are checked against the lower public format set;
- compile/module/function/provider/prepared-DAG composition remains public-only with no deep import;
- mathematical/reference fixtures remain unchanged;
- cleanup/failure behavior remains truthful;
- packed-package consumers exercise the corrected public dependency seam.

The exact CUDA-JS dependency revision must advance to alpha.17 in both package and lock/control assertions.

The prior recorded native Tensor evidence remains historical evidence for its recorded earlier CUDA-JS pair. This refactor does not claim native requalification until the exact new pair is rerun on its required physical profile.

## Falsifiers

Rollback this addendum if the migration requires private CUDA-JS imports, a new native boundary, a second scheduler/operation lifecycle, moving Tensor mathematics into CUDA-JS, hiding materially selected launch topology behind a generic resolver, or widening Tensor artifact formats merely because CUDA-JS supports more formats.

If a future Tensor planner needs a lower fact not publicly projected by CUDA-JS, route that demonstrated consumer-neutral gap to CUDA-JS before adding another copied constant here.

## Non-goals

No new Tensor operation, no provider family expansion, no performance tuning, no launch autotuning, no CUDA Graph realization, no multi-device work, no NN/search semantics, no native code, no generic preparation abstraction, and no activity on CUDA-MCGS #122.
