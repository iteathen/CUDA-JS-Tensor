import { createHash } from 'node:crypto';

import { TensorPlan } from '../../../components/tensor-program/index.mjs';

// These records are deterministic analysis snapshots. They do not make support
// claims, but they are fully executable analysis with an explicit identity.
export const TENSOR_IMPLEMENTATION_FRAME_CONTRACT = 'TENSOR-IMPLEMENTATION-FRAME-ANALYSIS-v1';
export function assertTensorPlan(plan, caller) {
  if (!(plan instanceof TensorPlan)) {
    throw new TypeError(`${caller} requires an accepted TensorPlan.`);
  }
}

export function deepFreeze(value) {
  if (value && typeof value === 'object') {
    for (const child of Array.isArray(value) ? value : Object.values(value)) deepFreeze(child);
    if (!Object.isFrozen(value)) Object.freeze(value);
  }
  return value;
}

export function frameRecord({ kind, plan, scope, candidates, blockers, completion }) {
  assertTensorPlan(plan, kind);
  const canonical = {
    contract: TENSOR_IMPLEMENTATION_FRAME_CONTRACT,
    kind,
    planIdentity: plan.compatibilityIdentity,
    status: 'analysis-only',
    scope,
    candidates,
    blockers,
    completion,
  };
  const compatibilityIdentity = `tensor-implementation-frame-v1:${createHash('sha256')
    .update(JSON.stringify(canonical))
    .digest('hex')}`;
  return deepFreeze({ ...canonical, compatibilityIdentity });
}
