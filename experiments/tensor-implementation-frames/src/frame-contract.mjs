import { createHash } from 'node:crypto';

import { TensorPlan } from '../../../components/tensor-program/index.mjs';

// These records are intentionally impossible to execute.  The experiment exists to
// preserve implementation-shaped work without accidentally adding a public backend.
// A future production chunk must replace its frame with an accepted specification,
// public CUDA-JS dependency, lifecycle implementation, and evidence.
export function assertTensorPlan(plan, caller) {
  if (!(plan instanceof TensorPlan)) {
    throw new TypeError(`${caller} requires an accepted TensorPlan.`);
  }
}

export function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const child of Array.isArray(value) ? value : Object.values(value)) deepFreeze(child);
    Object.freeze(value);
  }
  return value;
}

export function frameRecord({ kind, plan, scope, candidates, blockers, completion }) {
  assertTensorPlan(plan, kind);
  const canonical = {
    contract: 'INCOMPLETE-tensor-implementation-frame-v1',
    kind,
    planIdentity: plan.compatibilityIdentity,
    scope,
    candidates,
    blockers,
    completion,
    executable: false,
    supportClaim: false,
  };
  const compatibilityIdentity = `tensor-implementation-frame:${createHash('sha256')
    .update(JSON.stringify(canonical))
    .digest('hex')}`;
  return deepFreeze({ ...canonical, compatibilityIdentity });
}

