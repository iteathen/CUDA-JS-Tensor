import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import test from 'node:test';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { createDeviceItemProfile } from '../testing.mjs';

function itemUnary(operator, dtype = 'f32') {
  return TensorProgram.define((graph) => graph.unary(
    operator,
    graph.input('items', { dtype, capacityShape: [4, 2], access: 'read' }),
  ));
}

test('capture protected legacy device-item identity and demonstrate missing unary:erf lowering', () => {
  const legacy = createDeviceItemProfile(TensorPlan.create(itemUnary('neg')), {
    itemCapacity: 4,
    itemInputs: ['items'],
  });
  console.log(`DEVICE_ERF_LEGACY_IDENTITY=${legacy.compatibilityIdentity}`);
  console.log(`DEVICE_ERF_LEGACY_SOURCE_SHA256=${createHash('sha256').update(legacy.lowering.source).digest('hex')}`);

  const erf = createDeviceItemProfile(TensorPlan.create(itemUnary('erf')), {
    itemCapacity: 4,
    itemInputs: ['items'],
  });
  assert.match(erf.lowering.source, /gpu\.math\.erf/u);
});
