import { frameRecord } from './frame-contract.mjs';

function align(value, alignment) {
  return (Math.floor((value + alignment - 1) / alignment)) * alignment;
}

function firstFitAssign(allocations) {
  const ordered = [...allocations].sort((left, right) => (
    left.lifetime.definedAt - right.lifetime.definedAt || left.id.localeCompare(right.id)
  ));
  const slots = [];
  const assignments = [];
  let arenaBytes = 0;

  for (const allocation of ordered) {
    let selected = null;
    for (const slot of slots) {
      if (slot.lastUse < allocation.lifetime.definedAt
          && slot.byteLength >= allocation.byteLength
          && slot.offset % allocation.alignment === 0) {
        selected = slot;
        break;
      }
    }

    if (!selected) {
      const offset = align(arenaBytes, allocation.alignment);
      selected = {
        id: `slot:${slots.length}`,
        offset,
        byteLength: allocation.byteLength,
        lastUse: -1,
      };
      slots.push(selected);
      arenaBytes = offset + allocation.byteLength;
    }

    assignments.push(Object.freeze({
      allocation: allocation.id,
      slotId: selected.id,
      byteOffset: selected.offset,
      byteLength: allocation.byteLength,
    }));
    selected.lastUse = allocation.lifetime.lastUse;
    selected.byteLength = Math.max(selected.byteLength, allocation.byteLength);
  }

  const utilization = arenaBytes > 0 ? ((arenaBytes === 0) ? 0 : (ordered.reduce((sum, allocation) => sum + allocation.byteLength, 0) / arenaBytes)) : 1;
  return Object.freeze({
    ordered: Object.freeze(ordered.map((allocation) => allocation.id)),
    slots: Object.freeze(slots.map((slot) => Object.freeze({ id: slot.id, offset: slot.offset, byteLength: slot.byteLength }))),
    assignments: Object.freeze(assignments),
    arenaBytes,
    potentialSavingBytes: ordered.reduce((sum, allocation) => sum + allocation.byteLength, 0) - arenaBytes,
    utilization,
  });
}

export function frameArenaReuse(plan) {
  const assignment = firstFitAssign(plan.allocations);
  return frameRecord({
    kind: 'arena-reuse',
    plan,
    scope: 'Deterministic first-fit reuse of non-overlapping material lifetimes in one resolved execution.',
    candidates: Object.freeze([{
      totalMaterialBytes: plan.totalDistinctBytes,
      ...assignment,
    }]),
    blockers: [
      'TensorSession currently allocates each materialized TensorPlan node independently; arena reuse ownership is not bound.',
      'Alias boundaries and rollback rules must be proven before suballocation is enabled for run products.',
      'Cross-run arena persistence is intentionally out of scope for current execution contracts.',
    ],
    completion: [
      'Accept arena/suballocation ownership and child-view cleanup proof.',
      'Bind allocations to one bounded session arena with exact access ranges.',
      'Measure fragmentation and peak-memory effects before recommending policy by default.',
    ],
  });
}
