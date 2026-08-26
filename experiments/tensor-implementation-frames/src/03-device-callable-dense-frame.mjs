import { frameRecord } from './frame-contract.mjs';

const CANDIDATE_OPS = new Set(['cast', 'contiguous', 'unary', 'binary', 'reduce', 'matmul']);

function consumers(program) {
  const result = new Map();
  for (const node of program.nodes) {
    for (const input of node.inputIds) {
      if (!result.has(input)) result.set(input, []);
      result.get(input).push(node.id);
    }
  }
  return result;
}

// Chunk 3 finds maximal implementation-shaped regions that could become one
// typed device-callable dense leaf.  Region discovery is useful code, but it is
// deliberately not lowering: participation, shared memory, ABI, provider
// identity, device function linking, and caller-kernel integration still belong
// to the accepted CUDA-JS SPEC-0028 boundary and a future Tensor child spec.
export function frameDeviceCallableDense(plan) {
  const program = plan.program;
  const uses = consumers(program);
  const publicOutputs = new Set(program.outputs.map((entry) => entry.valueId));
  const regions = [];
  let current = [];

  function finish() {
    if (current.length === 0) return;
    const ids = new Set(current.map((node) => node.id));
    const externalInputs = new Set();
    const externalOutputs = new Set();
    for (const node of current) {
      for (const input of node.inputIds) if (!ids.has(input)) externalInputs.add(input);
      const outsideUse = (uses.get(node.id) ?? []).some((consumer) => !ids.has(consumer));
      if (outsideUse || publicOutputs.has(node.id)) externalOutputs.add(node.id);
    }
    if (current.some((node) => node.op === 'matmul')) {
      regions.push({
        firstNode: current[0].id,
        lastNode: current.at(-1).id,
        nodes: current.map((node) => ({ id: node.id, op: node.op })),
        externalInputs: [...externalInputs],
        externalOutputs: [...externalOutputs],
      });
    }
    current = [];
  }

  for (const node of program.nodes) {
    if (node.materialization === 'materialize' && CANDIDATE_OPS.has(node.op)) current.push(node);
    else finish();
  }
  finish();

  return frameRecord({
    kind: 'device-callable-dense-subgraph',
    plan,
    scope: 'Finite dense regions containing matmul and adjacent typed operations, callable as one injected device leaf.',
    candidates: regions,
    blockers: [
      'Exact provider/profile value over the complete SIMT or host-planned path has not been established.',
      'Shared-memory, participation, alignment, architecture, dtype, toolkit, license, and link gates are not resolved.',
      'No accepted Tensor callable-leaf ABI or caller-owned resource contribution exists.',
    ],
    completion: [
      'Accept the Tensor child profile after second-instance and deletion assessment.',
      'Resolve one CUDA-JS SPEC-0028 typed leaf package with finite resources and exact participation.',
      'Generate callable source and public link requests without adding a scheduler or native Tensor boundary.',
    ],
  });
}

