import assert from 'node:assert/strict';
import test from 'node:test';

import { compileDeviceProgram } from 'cuda-js';
import { openCudaRuntimeForTesting } from 'cuda-js/testing';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { TensorSession } from '../../tensor-value/index.mjs';
import { compileTensorDeviceProgram } from '../index.mjs';
import { createDeviceItemProfile } from '../testing.mjs';

const ITEM_CAPACITY = 8;
const BOARD_CELLS = 42;
const FRONTIER_CAPACITY = 16;
const MAX_WORKSPACE_BYTES = 8 * 1024 * 1024;

function dominanceProgram(direction) {
  return TensorProgram.define((graph) => {
    const candidates = graph.input('candidates', {
      dtype: 'f32',
      capacityShape: [ITEM_CAPACITY, BOARD_CELLS],
      access: 'read',
    });
    const frontierTransposed = graph.input('frontierTransposed', {
      dtype: 'f32',
      capacityShape: [BOARD_CELLS, FRONTIER_CAPACITY],
      access: 'read',
    });
    const ones = graph.input('ones', {
      dtype: 'f32',
      capacityShape: [FRONTIER_CAPACITY],
      access: 'read',
    });
    const overlap = graph.matmul(candidates, frontierTransposed);

    let subsetCardinality;
    if (direction === 'minimal') {
      subsetCardinality = graph.input('frontierPopcounts', {
        dtype: 'f32',
        capacityShape: [FRONTIER_CAPACITY],
        access: 'read',
      });
    } else {
      subsetCardinality = graph.reduce('sum', candidates, {
        axes: [1],
        keepDimensions: true,
        order: 'fixed-tree-v1',
      });
    }

    // For 0/1 bit rows, popcount(A) - dot(A, B) is zero iff A is a subset of B.
    // Values are exact integers in [0, 42] in f32. Clamp non-zero deficits to one,
    // invert to a 0/1 subset predicate, then OR all frontier comparisons with max.
    const deficit = graph.binary('sub', subsetCardinality, overlap);
    const nonZero = graph.binary('minimum', deficit, ones);
    const subset = graph.binary('sub', ones, nonZero);
    return graph.reduce('maximum', subset, {
      axes: [1],
      order: 'fixed-tree-v1',
    });
  });
}

for (const direction of ['minimal', 'maximal']) {
  test(`Connect4 BSFP ${direction} dominance maps to one item-parallel Tensor leaf`, () => {
    const plan = TensorPlan.create(dominanceProgram(direction));
    const profile = createDeviceItemProfile(plan, {
      itemCapacity: ITEM_CAPACITY,
      itemInputs: ['candidates'],
      maxWorkspaceBytes: MAX_WORKSPACE_BYTES,
    });

    assert.equal(profile.outputs.length, 1);
    assert.equal(profile.outputs[0].perItemElements, 1);
    assert(profile.totalWorkspaceBytes > 0);
    assert(profile.totalWorkspaceBytes <= MAX_WORKSPACE_BYTES);
    assert.match(profile.lowering.source, /for \(let k = gpu\.u64\(0n\)/u);
    assert.match(profile.lowering.source, /gpu\.math\.minimum/u);
    assert.doesNotMatch(profile.lowering.source, /gpu\.thread|gpu\.block|gpu\.barrier|gpu\.atomic/u);
  });

  test(`Connect4 BSFP ${direction} dominance Tensor leaf composes through public CUDA-JS imports`, { timeout: 15_000 }, async () => {
    const runtime = await openCudaRuntimeForTesting({ compiler: true });
    const session = await TensorSession.open(runtime);
    try {
      const deviceProgram = await compileTensorDeviceProgram(session, dominanceProgram(direction), {
        itemCapacity: ITEM_CAPACITY,
        itemInputs: ['candidates'],
        maxWorkspaceBytes: MAX_WORKSPACE_BYTES,
      });
      const alias = direction === 'minimal' ? 'bsfpDominatesMinimal' : 'bsfpDominatesMaximal';
      const parameters = [
        ...deviceProgram.function.parameters,
        { name: 'status', type: 'ptr<u32>' },
      ];
      const argumentsList = deviceProgram.function.parameters.map((entry) => entry.name).join(', ');
      const source = `function consumer(${parameters.map((entry) => entry.name).join(', ')}) { status[gpu.u32(0)] = ${alias}(${argumentsList}); }`;
      const composed = await compileDeviceProgram(runtime, {
        source,
        functions: [{ name: 'consumer', kind: 'kernel', parameters, returns: 'void' }],
        imports: [deviceProgram.importAs(alias)],
      });

      assert.equal(deviceProgram.function.name, 'tensorRunItem');
      assert.equal(deviceProgram.function.returns, 'u32');
      assert.equal(composed.deviceProgram.imports[0].exportName, 'tensorRunItem');
      assert.equal(composed.linker.artifact.format, 'cubin');
    } finally {
      assert.equal((await session.close()).graceful, true);
      assert.equal((await runtime.close()).graceful, true);
    }
  });
}
