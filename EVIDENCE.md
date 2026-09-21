# Evidence status

This repository follows the shared [iteathen evidence and validation policy](https://github.com/iteathen/.github/blob/main/EVIDENCE_POLICY.md).

## Current posture

CUDA-JS-Tensor is a public pre-release. Portable tests and repository verification are **INTERNAL-QUALIFICATION**. Native results tied to exact recorded CUDA-JS/library/hardware profiles are **HARDWARE-MEASURED** when their provenance records identify the physical environment.

## Registered claims

| Claim | Evidence class | Status |
| --- | --- | --- |
| `TENSOR-INT-001` — maintained portable/package/tensor tests pass for the tested revision | **INTERNAL-QUALIFICATION** | repository-controlled |
| `TENSOR-HW-001` — exact recorded native profiles demonstrate the behaviors recorded by their native evidence | **HARDWARE-MEASURED** | revision/environment scoped |
| `TENSOR-GENERAL-001` — general Linux, Tensor Core, multi-GPU, or performance superiority | **UNVALIDATED** | explicitly not claimed |

Machine-readable records: [`evidence/claims.json`](evidence/claims.json).

## What current evidence establishes

Internal checks establish the tested tensor semantics/package behavior. Native evidence establishes only the exact behavior, revisions, CUDA-JS pair, and hardware environment recorded by that evidence.

## What it does not establish

It does not establish general hardware/platform support, general performance superiority, production support, or independent third-party reproduction.

## Path to stronger evidence

Broader native claims require exact-profile hardware qualification on each supported environment. Comparative performance claims require reproducible workload definitions, raw timing data, hardware/software provenance, and an appropriate external/reference baseline.

## Non-mutation rule

Evidence work may exercise Tensor's public and native qualification paths, but must not alter tensor semantics, execution behavior, or production APIs merely to obtain favorable evidence.
