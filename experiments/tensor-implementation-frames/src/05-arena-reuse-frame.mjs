import { assertTensorPlan, frameRecord } from './frame-contract.mjs';

function safeAdd(left, right, field) {
  const result = left + right;
  if (!Number.isSafeInteger(result)) throw new RangeError(`${field} exceeds the safe integer range.`);
  return result;
}

function align(value, alignment) {
  const remainder = value % alignment;
  return remainder === 0 ? value : safeAdd(value, alignment - remainder, 'arena alignment');
}

function aliasClosedAllocations(plan) {
  const aliasByValue = new Map(plan.aliases.map((entry) => [entry.value, entry.aliasClass]));
  const lastUseByAlias = new Map();
  for (const lifetime of plan.liveness) {
    const aliasClass = aliasByValue.get(lifetime.value);
    if (aliasClass === undefined) throw new TypeError(`TensorPlan alias class is missing for ${lifetime.value}.`);
    lastUseByAlias.set(aliasClass, Math.max(lastUseByAlias.get(aliasClass) ?? -1, lifetime.lastUse));
  }

  return Object.freeze(plan.allocations.map((allocation) => {
    const aliasClass = aliasByValue.get(allocation.value);
    const aliasLastUse = lastUseByAlias.get(aliasClass);
    if (aliasClass === undefined || aliasLastUse === undefined) {
      throw new TypeError(`TensorPlan allocation alias lifetime is missing for ${allocation.id}.`);
    }
    return Object.freeze({
      ...allocation,
      aliasClass,
      directLastUse: allocation.lifetime.lastUse,
      lifetime: Object.freeze({
        definedAt: allocation.lifetime.definedAt,
        lastUse: aliasLastUse,
      }),
    });
  }));
}

function firstFitAssign(allocations) {
  const ordered = [...allocations].sort((left, right) => (
    left.lifetime.definedAt - right.lifetime.definedAt || left.id.localeCompare(right.id)
  ));
  const slots = [];
  const assignments = [];
  let arenaBytes = 0;
  let alignmentPaddingBytes = 0;

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

    const reused = selected !== null;
    if (!selected) {
      const offset = align(arenaBytes, allocation.alignment);
      alignmentPaddingBytes = safeAdd(alignmentPaddingBytes, offset - arenaBytes, 'arena alignment padding');
      selected = {
        id: `slot:${slots.length}`,
        offset,
        byteLength: allocation.byteLength,
        lastUse: -1,
      };
      slots.push(selected);
      arenaBytes = safeAdd(offset, allocation.byteLength, 'arena byte length');
    }

    assignments.push(Object.freeze({
      allocation: allocation.id,
      aliasClass: allocation.aliasClass,
      slotId: selected.id,
      byteOffset: selected.offset,
      byteLength: allocation.byteLength,
      definedAt: allocation.lifetime.definedAt,
      directLastUse: allocation.directLastUse,
      effectiveLastUse: allocation.lifetime.lastUse,
      reused,
    }));
    selected.lastUse = allocation.lifetime.lastUse;
  }

  const distinctMaterialBytes = ordered.reduce(
    (sum, allocation) => safeAdd(sum, allocation.byteLength, 'distinct material byte length'),
    0,
  );
  return Object.freeze({
    ordered: Object.freeze(ordered.map((allocation) => allocation.id)),
    slots: Object.freeze(slots.map((slot) => Object.freeze({
      id: slot.id,
      offset: slot.offset,
      byteLength: slot.byteLength,
    }))),
    assignments: Object.freeze(assignments),
    distinctMaterialBytes,
    arenaBytes,
    alignmentPaddingBytes,
    netByteDelta: distinctMaterialBytes - arenaBytes,
    reuseFactor: arenaBytes === 0 ? 1 : distinctMaterialBytes / arenaBytes,
  });
}

export function frameArenaReuse(plan) {
  assertTensorPlan(plan, 'frameArenaReuse');
  const allocations = aliasClosedAllocations(plan);
  const assignment = firstFitAssign(allocations);
  if (assignment.distinctMaterialBytes !== plan.totalDistinctBytes) {
    throw new TypeError('TensorPlan distinct material byte total does not match its allocations.');
  }
  return frameRecord({
    kind: 'arena-reuse',
    plan,
    scope: 'Deterministic first-fit reuse after closing each material lifetime over its complete TensorPlan alias class.',
    candidates: Object.freeze([assignment]),
    blockers: [
      'TensorSession currently allocates each materialized TensorPlan node independently; arena reuse ownership is not bound.',
      'Suballocation access ranges, rollback, and child-view cleanup must be accepted before run products may use this layout.',
      'Backend workspace and cross-run arena persistence remain outside this static material-allocation analysis.',
    ],
    completion: [
      'Accept arena/suballocation ownership and child-view cleanup proof.',
      'Bind alias-closed material allocations to one bounded session arena with exact access ranges.',
      'Measure alignment, fragmentation, and peak-memory effects before recommending policy by default.',
    ],
  });
}
