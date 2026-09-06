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

function profile(program) {
  return createDeviceItemProfile(TensorPlan.create(program), {
    itemCapacity: 4,
    itemInputs: ['items'],
  });
}

test('device-callable unary erf reuses the existing item ABI/workspace and public helper for f32/f64', () => {
  for (const dtype of ['f32', 'f64']) {
    const erf = profile(itemUnary('erf', dtype));
    assert.match(erf.lowering.source, /gpu\.math\.erf/u);
    assert.doesNotMatch(erf.lowering.source, /tanh|gelu|approx|__device__|#include|cuda[A-Z]/iu);
    assert.deepEqual(erf.parameters.map((entry) => entry.role), ['item-index', 'input', 'output', 'workspace']);
    assert.deepEqual(erf.inputs.map((entry) => [entry.name, entry.itemVarying]), [['items', true]]);
    assert.equal(erf.outputs[0].perItemElements, 2);
    assert.equal(erf.workspace.length, 1);
    assert.equal(erf.workspace[0].dtype, dtype);
    assert.equal(erf.workspace[0].perItemElements, 2);
    assert.equal(erf.totalWorkspaceBytes, 4 * 2 * (dtype === 'f32' ? 4 : 8));
  }
});

test('device-callable erf item profile is deterministic and retains existing out-of-range item guard', () => {
  const first = profile(itemUnary('erf'));
  const second = profile(itemUnary('erf'));
  assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
  assert.equal(first.lowering.source, second.lowering.source);
  assert.match(first.lowering.source, /if \(itemIndex >= gpu\.u32\(4\)\) \{\n    return gpu\.u32\(1\);/u);
});

test('legacy device-item identity and generated source remain exact when erf is unused', () => {
  const legacy = profile(itemUnary('neg'));
  const sourceSha256 = createHash('sha256').update(legacy.lowering.source).digest('hex');
  // These values are captured from protected main before the #52 production edit and are
  // replaced with literal assertions after the first exact-head qualification run.
  console.log(`DEVICE_ERF_LEGACY_IDENTITY=${legacy.compatibilityIdentity}`);
  console.log(`DEVICE_ERF_LEGACY_SOURCE_SHA256=${sourceSha256}`);
  assert.doesNotMatch(legacy.lowering.source, /gpu\.math\.erf/u);
});
