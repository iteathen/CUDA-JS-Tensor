# CUDA-JS-Tensor Agent Entry Point

This file is the mandatory first read for every developer. Continue with [`agent_files/AGENTS.md`](agent_files/AGENTS.md), [`agent_files/AI_RULES.md`](agent_files/AI_RULES.md), [`agent_files/DESIGN_ALIGNMENT_CARD.md`](agent_files/DESIGN_ALIGNMENT_CARD.md), and [`agent_files/general_foundation/PRINCIPLES.md`](agent_files/general_foundation/PRINCIPLES.md) before changing anything.

## Current phase

CUDA-JS-Tensor is a public pre-release repository in specification-first foundation work. It will provide consumer-neutral tensor planning and accelerated tensor execution through versioned public CUDA-JS contracts. It is not a neural-network framework, training system, CUDA binding, or CUDA-MCGS subsystem.

Production implementation begins only for an accepted owned boundary with dependency-ready public CUDA-JS contracts. Experiments must be explicitly disposable and cannot become production by drift.

## Authority order

1. Current explicit project-owner instruction.
2. This file and `agent_files/AGENTS.md`.
3. Accepted ADRs in `docs/decisions/`.
4. Accepted specifications in `docs/specs/`.
5. Agent rules, registry, validation policy, and charter.
6. Architecture, research, plans, and status material.

## Non-negotiable invariants

- Apply domain truth -> LEGO ownership -> SOLID responsibility -> CUPID quality -> simplest sufficient total system -> measured validation.
- One fact, state, resource, lifecycle, or compatibility identity has one visible owner.
- Host code is ordinary JavaScript. Device programs are restricted Device-JS. Native CUDA implementation belongs behind public CUDA-JS contracts.
- Do not add C/C++, CUDA C++, native addons, direct FFI/Driver calls, hand-written PTX, or embedded CUDA source.
- Do not deep-import CUDA-JS internals or invent a workaround for a missing CUDA-JS primitive. Stop and request a consumer-neutral public capability.
- Public tensor contracts contain no neural-network, MCGS, chess, model, loss, optimizer, checkpoint, or training-policy assumption.
- One v1 session owns exactly one selected CUDA-JS runtime/device. Multi-device work composes sessions; it does not smuggle cross-device ownership into one tensor.
- Shapes, dtypes, strides, active extents, aliasing, precision, workspace, synchronization, failure, and cleanup are explicit and bounded.
- Convenience overloads normalize to the same canonical contract as expert forms. Defaults are documented, inspectable, overridable, and identity-affecting where relevant.
- Acceleration is optional but strongly recommended when qualified evidence shows an end-to-end benefit. The portable/SIMT path remains complete.
- No production `.cu` or `.ptx` files. Generated implementation is private, deterministic, and attributable to maintained JavaScript plus versioned dependencies.
- `the_restaurant` integration is deferred. Preserve its plan without changing that repository until separately authorized.
- Do not weaken validation, protections, cleanup, or evidence to make a change pass.

## Required validation

```bash
npm run verify
```

No implementation, support, performance, compatibility, publication, or cleanup claim exists beyond the exact evidence run.
