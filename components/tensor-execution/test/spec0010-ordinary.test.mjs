import assert from 'node:assert/strict';
import test from 'node:test';

import { compileDeviceProgram } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { createDeviceItemProfile, lowerSimtPlan } from '../testing.mjs';

function extensionProgram(dtype = 'f32') {
  return TensorProgram.define((graph) => {
    const source = graph.input('source', { dtype, capacityShape: [2, 5], access: 'read' });
    const gathered = graph.gather(source, 1, [4, 1, 4, 0]);
    const gaussian = graph.unary('erf', gathered);
    const tail = graph.input('tail', { dtype, capacityShape: [2, 2], access: 'read' });
    return graph.concat([gaussian, tail], 1);
  });
}

test('ordinary SPEC-0010 lowering is deterministic and uses only public Device-JS source semantics', { timeout: 15_000 }, async () => {
  const first = lowerSimtPlan(TensorPlan.create(extensionProgram('f32')));
  const second = lowerSimtPlan(TensorPlan.create(extensionProgram('f32')));
  assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
  assert.equal(first.source, second.source);
  assert.equal(first.kernels.length, 3);
  assert.deepEqual(first.kernels.map((entry) => entry.semanticNode), ['node:0', 'node:1', 'node:2']);
  assert.match(first.source, /gpu\.math\.erf/u);
  assert.doesNotMatch(first.source, /tanh|gelu|#include|__device__|cuda[A-Z]/iu);

  const gatherSource = first.kernels[0].source;
  const staticAssignments = [...gatherSource.matchAll(/gatheredAxis = gpu\.u64\((\d+)n\)/gu)].map((match) => Number(match[1]));
  assert.deepEqual(staticAssignments, [0, 4, 1, 4, 0]);
  assert.match(gatherSource, /if \(c1 === gpu\.u64\(0n\)\)/u);
  assert.match(gatherSource, /if \(c1 === gpu\.u64\(3n\)\)/u);

  const concatSource = first.kernels[2].source;
  assert.match(concatSource, /if \(c1 >= gpu\.u64\(0n\) && c1 < gpu\.u64\(4n\)\)/u);
  assert.match(concatSource, /if \(c1 >= gpu\.u64\(4n\) && c1 < gpu\.u64\(6n\)\)/u);
  assert.match(concatSource, /concatValue = p0\[concatOffset0\]/u);
  assert.match(concatSource, /concatValue = p1\[concatOffset1\]/u);

  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  try {
    const compiled = await compileDeviceProgram(runtime, { source: first.source, functions: first.functions });
    assert.match(compiled.deviceProgram.contract, /SPEC-0030-dense-numeric-v1\+SPEC-0030-erf-v1$/u);
    assert.equal(compiled.compiler.headerProfile, 'cuda-numeric');
  } finally {
    assert.equal((await runtime.close()).graceful, true);
  }
});

test('f64 ordinary erf lowers through the same public helper without Tensor-owned approximation', { timeout: 15_000 }, async () => {
  const lowering = lowerSimtPlan(TensorPlan.create(extensionProgram('f64')));
  assert.match(lowering.kernels[1].source, /gpu\.math\.erf/u);
  assert.doesNotMatch(lowering.source, /tanh|gelu|approx/iu);
  const runtime = await openCudaRuntimeForTesting({ compiler: true });
  try {
    const compiled = await compileDeviceProgram(runtime, { source: lowering.source, functions: lowering.functions });
    assert.match(compiled.deviceProgram.contract, /SPEC-0030-dense-numeric-v1\+SPEC-0030-erf-v1$/u);
  } finally {
    assert.equal((await runtime.close()).graceful, true);
  }
});

test('ordinary concat pressure remains a resolved binding limit rather than a semantic rejection', () => {
  const program = TensorProgram.define((graph) => {
    const inputs = Array.from({ length: 64 }, (_, index) => graph.input(`input${index}`, { dtype: 'f32', capacityShape: [1], access: 'read' }));
    return graph.concat(inputs, 0);
  });
  assert.equal(program.nodes[0].inputIds.length, 64);
  assert.throws(() => lowerSimtPlan(TensorPlan.create(program)), (error) => error?.code === 'TENSOR_SIMT_BINDING_LIMIT' && error?.category === 'pressure');
});

test('SPEC-0009 device-callable admission remains fail-closed for gather and concat', () => {
  const gathered = TensorProgram.define((graph) => {
    const items = graph.input('items', { dtype: 'f32', capacityShape: [4, 3], access: 'read' });
    return graph.gather(items, 1, [2, 0]);
  });
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(gathered), { itemCapacity: 4, itemInputs: ['items'] }), (error) => error?.category === 'unsupported');

  const concatenated = TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4, 1], access: 'read' });
    return graph.concat([left, right], 1);
  });
  assert.throws(() => createDeviceItemProfile(TensorPlan.create(concatenated), { itemCapacity: 4, itemInputs: ['left', 'right'] }), (error) => error?.category === 'unsupported');
});
