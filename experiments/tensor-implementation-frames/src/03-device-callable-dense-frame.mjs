import { assertTensorPlan, frameRecord } from './frame-contract.mjs';

const CANDIDATE_OPS = new Set(['cast', 'contiguous', 'unary', 'binary', 'reduce', 'matmul']);

function isCandidateNode(node) {
  return node.materialization === 'materialize' && CANDIDATE_OPS.has(node.op);
}

function candidateComponents(program) {
  const candidates = program.nodes.filter(isCandidateNode);
  const byId = new Map(candidates.map((node) => [node.id, node]));
  const indexById = new Map(program.nodes.map((node, index) => [node.id, index]));
  const adjacent = new Map(candidates.map((node) => [node.id, new Set()]));
  const consumers = new Map();

  for (const node of program.nodes) {
    for (const inputId of new Set(node.inputIds)) {
      if (!consumers.has(inputId)) consumers.set(inputId, new Set());
      consumers.get(inputId).add(node.id);
      if (byId.has(node.id) && byId.has(inputId)) {
        adjacent.get(node.id).add(inputId);
        adjacent.get(inputId).add(node.id);
      }
    }
  }

  const publicOutputs = new Set(program.outputs.map((entry) => entry.valueId));
  const visited = new Set();
  const regions = [];

  for (const start of candidates) {
    if (visited.has(start.id)) continue;
    const pending = [start.id];
    const componentIds = [];
    visited.add(start.id);

    while (pending.length > 0) {
      const currentId = pending.pop();
      componentIds.push(currentId);
      for (const adjacentId of adjacent.get(currentId)) {
        if (visited.has(adjacentId)) continue;
        visited.add(adjacentId);
        pending.push(adjacentId);
      }
    }

    componentIds.sort((left, right) => indexById.get(left) - indexById.get(right));
    const nodes = componentIds.map((id) => byId.get(id));
    const matmulCount = nodes.filter((node) => node.op === 'matmul').length;
    if (matmulCount === 0) continue;

    const regionIds = new Set(componentIds);
    const boundaryInputs = new Set();
    const boundaryOutputs = new Set();
    for (const node of nodes) {
      for (const inputId of node.inputIds) {
        if (!regionIds.has(inputId)) boundaryInputs.add(inputId);
      }
      const downstream = consumers.get(node.id) ?? new Set();
      if ([...downstream].some((consumerId) => !regionIds.has(consumerId)) || publicOutputs.has(node.id)) {
        boundaryOutputs.add(node.id);
      }
    }

    regions.push(Object.freeze({
      firstTopologicalNode: nodes[0].id,
      lastTopologicalNode: nodes.at(-1).id,
      candidateSize: nodes.length,
      matmulCount,
      nodes: Object.freeze(nodes.map((node) => Object.freeze({
        id: node.id,
        op: node.op,
        rank: node.outputSpec.rank,
        dtype: node.outputSpec.dtype,
      }))),
      boundaryInputs: Object.freeze([...boundaryInputs]),
      boundaryOutputs: Object.freeze([...boundaryOutputs]),
    }));
  }

  return Object.freeze(regions);
}

export function frameDeviceCallableDense(plan) {
  assertTensorPlan(plan, 'frameDeviceCallableDense');
  const regions = candidateComponents(plan.program);
  return frameRecord({
    kind: 'device-callable-dense-subgraph',
    plan,
    scope: 'Maximal connected material regions anchored by matmul with every external input and observable/external output exposed.',
    candidates: regions,
    blockers: [
      'Exact callable profile and complete participation contract are not yet accepted.',
      'Shared-memory, architecture, alignment, and link requirements are not specified.',
      'No validated lifecycle model exists for public output materialization with mixed callable + SIMT edges.',
    ],
    completion: [
      'Accept Tensor device-callable leaf profile after second-instance review.',
      'Resolve finite ABI identity, resources, participation, and failure attribution.',
      'Generate and link callable source through public CUDA-JS contracts only.',
    ],
  });
}
