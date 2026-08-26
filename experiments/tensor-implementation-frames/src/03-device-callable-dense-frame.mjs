import { frameRecord } from './frame-contract.mjs';

const CANDIDATE_OPS = new Set(['cast', 'contiguous', 'unary', 'binary', 'reduce', 'matmul']);

function consumersByValue(program) {
  const byInput = new Map();
  for (const node of program.nodes) {
    for (const inputId of node.inputIds) {
      if (!byInput.has(inputId)) byInput.set(inputId, []);
      byInput.get(inputId).push(node.id);
    }
  }
  return byInput;
}

function isCandidateNode(node) {
  return node.materialization === 'materialize' && CANDIDATE_OPS.has(node.op);
}

function collectCandidates(program, planInputs) {
  const byId = new Map(program.nodes.map((node) => [node.id, node]));
  const consumerByInput = consumersByValue(program);
  const publicOutputs = new Set(planInputs.map((entry) => entry.valueId));
  const used = new Set();
  const regions = [];

  for (const start of program.nodes) {
    if (!isCandidateNode(start) || used.has(start.id)) continue;
    const nodes = [];
    let current = start;
    while (current && isCandidateNode(current) && !used.has(current.id)) {
      nodes.push(current);
      const downstream = consumerByInput.get(current.id) ?? [];
      if (downstream.length !== 1) break;
      const next = byId.get(downstream[0]);
      if (!next || !isCandidateNode(next) || used.has(next.id)) break;
      current = next;
    }
    if (nodes.length <= 1) continue;
    const regionIds = new Set(nodes.map((entry) => entry.id));
    for (const node of nodes) used.add(node.id);

    const boundaryInputs = new Set();
    const boundaryOutputs = new Set();
    for (const node of nodes) {
      for (const inputId of node.inputIds) {
        if (!regionIds.has(inputId)) boundaryInputs.add(inputId);
      }
      const outgoing = consumerByInput.get(node.id) ?? [];
      if (outgoing.some((consumerId) => !regionIds.has(consumerId)) || publicOutputs.has(node.id)) {
        boundaryOutputs.add(node.id);
      }
    }
    regions.push(Object.freeze({
      firstNode: nodes[0].id,
      lastNode: nodes.at(-1).id,
      nodes: Object.freeze(nodes.map((node) => Object.freeze({
        id: node.id,
        op: node.op,
        rank: node.outputSpec.rank,
        dtype: node.outputSpec.dtype,
      }))),
      candidateSize: nodes.length,
      boundaryInputs: Object.freeze([...boundaryInputs]),
      boundaryOutputs: Object.freeze([...boundaryOutputs]),
    }));
  }

  return Object.freeze(regions);
}

export function frameDeviceCallableDense(plan) {
  const regions = collectCandidates(plan.program, plan.program.outputs);
  return frameRecord({
    kind: 'device-callable-dense-subgraph',
    plan,
    scope: 'Finite dense regions with matmul and adjacent typed operations for one injected device leaf.',
    candidates: regions,
    blockers: [
      'Exact callable profile and complete participation contract are not yet accepted.',
      'Shared-memory, architecture, alignment, and link requirements are not specified.',
      'No validated lifecycle model exists for public output materialization with mixed callable + SIMT edges.',
    ],
    completion: [
      'Accept Tensor device-callable leaf profile after second-instance review.',
      'Resolve finite ABI/ABI identity and failure attribution contract.',
      'Generate and link callable source through public CUDA-JS contracts only.',
    ],
  });
}
