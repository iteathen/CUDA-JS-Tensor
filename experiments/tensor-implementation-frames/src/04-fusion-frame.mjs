import { frameRecord } from './frame-contract.mjs';

const FUSIBLE_OPS = new Set(['cast', 'unary', 'binary']);

// Chunk 4 identifies conservative single-consumer elementwise chains.  It does
// not emit a fused kernel: numeric step rounding, observable intermediates,
// access ranges, failure attribution, launch bounds, and source identity must be
// specified first.  Keeping discovery separate makes deletion easy if fusion is
// not profitable on representative workloads.
export function frameElementwiseFusion(plan) {
  const program = plan.program;
  const uses = new Map();
  for (const node of program.nodes) {
    for (const input of node.inputIds) {
      if (!uses.has(input)) uses.set(input, []);
      uses.get(input).push(node.id);
    }
  }
  const publicOutputs = new Set(program.outputs.map((entry) => entry.valueId));
  const byId = new Map(program.nodes.map((node) => [node.id, node]));
  const claimed = new Set();
  const chains = [];

  for (const start of program.nodes) {
    if (claimed.has(start.id) || !FUSIBLE_OPS.has(start.op) || start.materialization !== 'materialize') continue;
    const chain = [start];
    let cursor = start;
    while (!publicOutputs.has(cursor.id)) {
      const nextIds = uses.get(cursor.id) ?? [];
      if (nextIds.length !== 1) break;
      const next = byId.get(nextIds[0]);
      if (!next || claimed.has(next.id) || !FUSIBLE_OPS.has(next.op) || next.materialization !== 'materialize') break;
      chain.push(next);
      cursor = next;
    }
    if (chain.length > 1) {
      for (const node of chain) claimed.add(node.id);
      chains.push({ nodes: chain.map((node) => ({ id: node.id, op: node.op })), output: chain.at(-1).id });
    }
  }

  return frameRecord({
    kind: 'elementwise-fusion',
    plan,
    scope: 'Conservative single-consumer cast/unary/binary chains with no public intermediate output.',
    candidates: chains,
    blockers: [
      'SPEC-0004 per-operation rounding and failure attribution have no accepted fused equivalence profile.',
      'Generated fused source, access ranges, launch bounds, and identity are not implemented.',
      'No representative evidence shows that removed launches outweigh larger generated kernels.',
    ],
    completion: [
      'Accept an exact fusion equivalence and observability child profile.',
      'Generate one specialized kernel while preserving public outputs and SIMT fallback.',
      'Measure total execution and compilation/cache cost before recommendation.',
    ],
  });
}
