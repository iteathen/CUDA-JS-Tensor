# SPEC-0009: Block32 item participation addendum

**Status:** Candidate implementation profile; independent review and exact native qualification required before integration
**Date:** 2026-10-08
**Authority:** Explicit owner instruction to qualify a device-closed cooperative item callable without changing scalar mathematics

`compileTensorDeviceProgram(..., { participation: 'block32', ... })` selects an additive `SPEC-0009-block32-v1` child after any gather/concat child. Omission or explicit `'scalar'` preserves the accepted scalar source, profile identity, ABI and workspace. Unknown participation rejects before compilation.

Exactly one 32x1x1 CUDA block participates in one invocation for one item. All 32 threads invoke the callable exactly once, on a uniform caller branch, with the same itemIndex. The caller owns item scheduling and must prove these participation facts before ignition using the callable's public metadata and `requireParticipation({ block, uniformItemIndex: true, uniformCall: true })`. Partial participation, multiple items in one block or divergent caller invocation is outside this profile and must be rejected by the caller's admission owner.

The library uses only existing public `gpu.thread.x()` and `gpu.barrier.block()` helpers. Each material/output element uses its exact previous scalar formula, including serial matmul contraction and the existing balanced reduction order. Only the outer independent element loop starts at the participant index and strides by 32. Each nonempty material stage ends with an unconditional block barrier outside all participant-divergent loops. Views perform no work. Output copy stages likewise synchronize before return. No warp helper, shared allocation, native import or host progress is introduced.

The uniform item index makes the existing out-of-range early return uniform over the block and therefore preserves no-read/no-write behavior. Every item's existing workspace range, layout, byte count and ABI remain unchanged and disjoint. Participation changes the physical compatibility identity and is declared in the public descriptor; it cannot silently replace scalar or host-resident qualification.

Native qualification requires two concurrent item blocks, out-of-range blocks and guarded outputs/workspace, mixed matmul/reduce/erf/gather/tanh math, exact cleanup and bounded measured workloads before a larger invocation. A measurement qualifies only its exact program/inputs/cohort/hardware; no unmeasured throughput or broader performance claim follows.
