import assert from 'node:assert/strict';
import test from 'node:test';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TENSOR_DEVICE_GATHER_CONCAT_CONTRACT, TENSOR_DEVICE_PROGRAM_CONTRACT } from '../index.mjs';
import { createDeviceItemProfile } from '../testing.mjs';

function profile(program, itemInputs) {
  return createDeviceItemProfile(TensorPlan.create(program), { itemCapacity: 4, itemInputs });
}

function gatherProgram(indices = [2, 0, 2], axis = 1) {
  return TensorProgram.define((graph) => {
    const items = graph.input('items', { dtype: 'f32', capacityShape: [4, 3, 4], access: 'read' });
    return graph.gather(items, axis, indices);
  });
}

function concatProgram(axis = 1) {
  return TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4, 3], access: 'read' });
    return graph.concat([left, right], axis);
  });
}

test('non-axis gather preserves item ownership, duplicate/nonmonotonic order, ABI and per-item workspace', () => {
  const result = profile(gatherProgram(), ['items']);
  assert.equal(result.contract, TENSOR_DEVICE_GATHER_CONCAT_CONTRACT);
  assert.equal(result.canonical.contract, TENSOR_DEVICE_GATHER_CONCAT_CONTRACT);
  assert.deepEqual(result.inputs.map((entry) => [entry.name, entry.itemVarying]), [['items', true]]);
  assert.deepEqual(result.parameters.map((entry) => entry.role), ['item-index', 'input', 'output', 'workspace']);
  assert.equal(result.outputs[0].perItemElements, 12);
  assert.equal(result.workspace.length, 1);
  assert.equal(result.workspace[0].perItemElements, 12);
  assert.match(result.lowering.source, /let n0GatherIndex = gpu\.u64\(0n\)/u);
  assert.match(result.lowering.source, /n0GatherIndex = gpu\.u64\(2n\)/u);
  assert.match(result.lowering.source, /n0GatherIndex = gpu\.u64\(0n\)/u);
  assert.match(result.lowering.source, /n0InputOffset \+= item \* gpu\.u64\(12n\)/u);
  assert.match(result.lowering.source, /let n0OutputOffset = item \* gpu\.u64\(12n\)/u);
  assert.doesNotMatch(result.lowering.source, /gpu\.(?:thread|block|atomic|barrier|mailbox)|#include|__device__|cuda[A-Z]/u);
});

test('empty gather keeps child identity and zero logical output without generated cross-item work', () => {
  const result = profile(gatherProgram([]), ['items']);
  assert.equal(result.contract, TENSOR_DEVICE_GATHER_CONCAT_CONTRACT);
  assert.equal(result.outputs[0].perItemElements, 0);
  assert.doesNotMatch(result.lowering.source, /GatherIndex/u);
  assert.match(result.lowering.source, /if \(itemIndex >= gpu\.u32\(4\)\)/u);
});

test('ordered concat preserves item ownership, canonical input order and existing ABI/workspace owner', () => {
  const result = profile(concatProgram(), ['left', 'right']);
  assert.equal(result.contract, TENSOR_DEVICE_GATHER_CONCAT_CONTRACT);
  assert.deepEqual(result.inputs.map((entry) => [entry.name, entry.itemVarying]), [['left', true], ['right', true]]);
  assert.deepEqual(result.parameters.map((entry) => entry.role), ['item-index', 'input', 'input', 'output', 'workspace']);
  assert.equal(result.outputs[0].perItemElements, 5);
  assert.equal(result.workspace[0].perItemElements, 5);
  assert.match(result.lowering.source, /if \(n0c0 < gpu\.u64\(2n\)\)/u);
  assert.match(result.lowering.source, /else if \(n0c0 < gpu\.u64\(5n\)\)/u);
  assert.match(result.lowering.source, /n0Input0Offset \+= item \* gpu\.u64\(2n\)/u);
  assert.match(result.lowering.source, /n0Input1Offset \+= item \* gpu\.u64\(3n\)/u);
  assert.match(result.lowering.source, /let n0OutputOffset = item \* gpu\.u64\(5n\)/u);
  assert.doesNotMatch(result.lowering.source, /gpu\.(?:thread|block|atomic|barrier|mailbox)|#include|__device__|cuda[A-Z]/u);
});

test('axis-0 gather and concat reject before device compilation', () => {
  const gather = TensorProgram.define((graph) => graph.gather(
    graph.input('items', { dtype: 'f32', capacityShape: [4, 3], access: 'read' }),
    0,
    [3, 1],
  ));
  assert.throws(() => profile(gather, ['items']), (error) => error.code === 'TENSOR_DEVICE_GATHER_ITEM_AXIS_UNSUPPORTED');

  const concat = TensorProgram.define((graph) => {
    const left = graph.input('left', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    const right = graph.input('right', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    return graph.concat([left, right], 0);
  });
  assert.throws(() => profile(concat, ['left', 'right']), (error) => error.code === 'TENSOR_DEVICE_CONCAT_ITEM_AXIS_UNSUPPORTED');
});

test('shared gather source and any shared concat input reject before compiler work', () => {
  const gather = TensorProgram.define((graph) => {
    graph.input('items', { dtype: 'f32', capacityShape: [4, 1], access: 'read' });
    const shared = graph.input('shared', { dtype: 'f32', capacityShape: [3, 4], access: 'read' });
    return graph.gather(shared, 1, [3, 0]);
  });
  assert.throws(() => profile(gather, ['items']), (error) => error.code === 'TENSOR_DEVICE_GATHER_SHARED_SOURCE_UNSUPPORTED');

  const concat = TensorProgram.define((graph) => {
    const items = graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' });
    const shared = graph.input('shared', { dtype: 'f32', capacityShape: [4, 3], access: 'read' });
    return graph.concat([items, shared], 1);
  });
  assert.throws(() => profile(concat, ['items']), (error) => error.code === 'TENSOR_DEVICE_CONCAT_SHARED_INPUT_UNSUPPORTED');
});

test('child is additive: base and unary-erf device-item profiles keep the exact base contract', () => {
  const negProgram = TensorProgram.define((graph) => graph.unary('neg', graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' })));
  const erfProgram = TensorProgram.define((graph) => graph.unary('erf', graph.input('items', { dtype: 'f32', capacityShape: [4, 2], access: 'read' })));
  const first = profile(negProgram, ['items']);
  const second = profile(negProgram, ['items']);
  const erf = profile(erfProgram, ['items']);
  assert.equal(first.contract, TENSOR_DEVICE_PROGRAM_CONTRACT);
  assert.equal(first.canonical.contract, TENSOR_DEVICE_PROGRAM_CONTRACT);
  assert.equal(first.compatibilityIdentity, second.compatibilityIdentity);
  assert.equal(first.lowering.source, second.lowering.source);
  assert.equal(erf.contract, TENSOR_DEVICE_PROGRAM_CONTRACT);
  assert.match(erf.lowering.source, /gpu\.math\.erf/u);
});

test('out-of-range guard precedes every generated read/write and item-local storage has no global mutable channel', () => {
  for (const result of [profile(gatherProgram(), ['items']), profile(concatProgram(), ['left', 'right'])]) {
    const source = result.lowering.source;
    const guard = source.indexOf('if (itemIndex >= gpu.u32(4))');
    const firstInput = source.search(/input\d+\[/u);
    const firstWorkspace = source.search(/workspace\d+\[/u);
    const firstOutput = source.search(/output\d+\[/u);
    assert(guard >= 0);
    for (const firstAccess of [firstInput, firstWorkspace, firstOutput]) if (firstAccess >= 0) assert(guard < firstAccess);
    assert.doesNotMatch(source, /gpu\.(?:thread|block|atomic|barrier|mailbox)|static |global|shared/u);
    assert.match(source, /item \* gpu\.u64\(/u);
  }
});
