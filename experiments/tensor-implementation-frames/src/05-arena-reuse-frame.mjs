import { frameRecord } from './frame-contract.mjs';

function align(value, alignment) {
  return Math.ceil(value / alignment) * alignment;
}

// Chunk 5 creates a deterministic first-fit candidate arena from already proved
// TensorPlan lifetimes.  The result is not used by execution.  Production reuse
// still needs view/suballocation capabilities, binding-range proof, result
// ownership, cancellation, rollback, and high-water evidence.  This prototype
// exists so those obligations attach to an actual proposed layout algorithm.
export function frameArenaReuse(plan) {
  const allocations = [...plan.allocations].sort((left, right) => (
    left.lifetime.definedAt - right.lifetime.definedAt || left.id.localeCompare(right.id)
  ));
  const slots = [];
  const assignments = [];
  let arenaBytes = 0;

  for (const allocation of allocations) {
    let slot = slots.find((candidate) => (
      candidate.lastUse < allocation.lifetime.definedAt
      && candidate.byteLength >= allocation.byteLength
      && candidate.offset % allocation.alignment === 0
    ));
    if (!slot) {
      const offset = align(arenaBytes, allocation.alignment);
      slot = { id: `slot:${slots.length}`, offset, byteLength: allocation.byteLength, lastUse: -1 };
      slots.push(slot);
      arenaBytes = offset + allocation.byteLength;
    }
    assignments.push({ allocation: allocation.id, slot: slot.id, offset: slot.offset, byteLength: allocation.byteLength });
    slot.lastUse = allocation.lifetime.lastUse;
  }

  return frameRecord({
    kind: 'arena-reuse',
    plan,
    scope: 'Deterministic first-fit reuse of non-overlapping material lifetimes inside one resolved-plan run.',
    candidates: [{
      distinctBytes: plan.totalDistinctBytes,
      candidateArenaBytes: arenaBytes,
      slots: slots.map(({ id, offset, byteLength }) => ({ id, offset, byteLength })),
      assignments,
    }],
    blockers: [
      'TensorSession has no accepted arena/suballocation execution contract for result-owned material views.',
      'Alias, binding-range, cancellation, rollback, child-view, and cleanup proof is incomplete.',
      'The candidate layout has not been compared against allocation pressure or fragmentation on representative plans.',
    ],
    completion: [
      'Accept arena ownership and suballocation lifecycle semantics.',
      'Bind every assignment through public bounded CUDA-JS views with exact access ranges.',
      'Prove result teardown, rollback, and first-consumer deletion before enabling reuse.',
    ],
  });
}
