import { frameRecord } from './frame-contract.mjs';

const FUSIBLE_OPS = new Set(['cast', 'unary', 'binary']);

function buildConsumerMap(program) {
  const byInput = new Map();
  for (const node of program.nodes) {
    for (const inputId of node.inputIds) {
      if (!byInput.has(inputId)) byInput.set(inputId, []);
      byInput.get(inputId).push(node.id);
    }
  }
  return byInput;
}

function indexById(nodes) {
  const byId = new Map();
  for (const node of nodes) byId.set(node.id, node);
  return byId;
}

function chainsFromProgram(program) {
  const byId = indexById(program.nodes);
  const outputs = new Set(program.outputs.map((entry) => entry.valueId));
  const uses = buildConsumerMap(program);
  const used = new Set();
  const chains = [];

  for (const node of program.nodes) {
    if (used.has(node.id) || node.materialization !== 'materialize' || !FUSIBLE_OPS.has(node.op)) continue;
    const nodes = [node];
    used.add(node.id);

    let cursor = node;
    while (!outputs.has(cursor.id)) {
      const downstream = uses.get(cursor.id) ?? [];
      if (downstream.length !== 1) break;
      const next = byId.get(downstream[0]);
      if (!next || used.has(next.id) || next.materialization !== 'materialize' || !FUSIBLE_OPS.has(next.op)) break;
      if (next.inputIds.filter((inputId) => inputId === cursor.id).length !== 1) break;
      nodes.push(next);
      used.add(next.id);
      cursor = next;
    }

    if (nodes.length <= 1) {
      used.delete(node.id);
      continue;
    }

    const hasBinary = nodes.some((entry) => entry.op === 'binary');
    const boundaryInputCount = new Set();
    const boundaryOutputCount = new Set();
    const regionIds = new Set(nodes.map((entry) => entry.id));
    for (const entry of nodes) {
      for (const inputId of entry.inputIds) if (!regionIds.has(inputId)) boundaryInputCount.add(inputId);
      const downstream = uses.get(entry.id) ?? [];
      if (outputs.has(entry.id) || downstream.some((id) => !regionIds.has(id))) boundaryOutputCount.add(entry.id);
    }

    chains.push(Object.freeze({
      output: nodes.at(-1).id,
      input: nodes[0].id,
      candidateSize: nodes.length,
      hasBinary,
      nodes: Object.freeze(nodes.map((entry) => Object.freeze({
        id: entry.id,
        op: entry.op,
        dtype: entry.outputSpec.dtype,
      }))),
      boundaryInputs: Object.freeze([...boundaryInputCount]),
      boundaryOutputs: Object.freeze([...boundaryOutputCount]),
    }));
  }

  return Object.freeze(chains);
}

export function frameElementwiseFusion(plan) {
  const chains = chainsFromProgram(plan.program);
  return frameRecord({
    kind: 'elementwise-fusion',
    plan,
    scope: 'Conservative single-consumer cast/unary/binary chains with no public intermediate outputs.',
    candidates: chains,
    blockers: [
      'Fused rounding/error attribution equivalence is not yet represented by a public profile.',
      'Generated kernel source, launch bounds, and access ranges are not yet implemented.',
      'Representative evidence is required before recommending launch-count reduction over compilation cost.',
    ],
    completion: [
      'Accept exact fusion equivalence and observability child profile.',
      'Generate one specialized kernel while preserving public outputs and SIMT fallback.',
      'Measure total execution and compilation/cache cost before recommendation.',
    ],
  });
}
