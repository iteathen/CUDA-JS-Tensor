import { TensorPlan } from '../../tensor-program/index.mjs';

import { deepFreeze, fail, identity } from './contract.mjs';

export const TENSOR_FUSION_POLICIES = Object.freeze(['none', 'exact-elementwise']);

const FUSIBLE_OPERATIONS = new Set(['cast', 'unary', 'binary']);

function sameArray(left, right) {
  return left.length === right.length && left.every((value, index) => value === right[index]);
}

function sameActiveExtent(left, right) {
  if (left === null || right === null) return left === right;
  return left.maximum === right.maximum && left.extent === right.extent;
}

function sameElementDomain(left, right) {
  return left.rank === right.rank
    && sameArray(left.capacityShape, right.capacityShape)
    && sameArray(left.logicalShape, right.logicalShape)
    && sameActiveExtent(left.activeAxis0, right.activeAxis0);
}

function consumerMap(program) {
  const result = new Map();
  for (const node of program.nodes) {
    for (const inputId of new Set(node.inputIds)) {
      if (!result.has(inputId)) result.set(inputId, new Set());
      result.get(inputId).add(node.id);
    }
  }
  return result;
}

function candidate(node) {
  return node.materialization === 'materialize'
    && FUSIBLE_OPERATIONS.has(node.op)
    && node.outputSpec.logicalElementCount > 0;
}

function externalInputs(nodes) {
  const regionIds = new Set(nodes.map((node) => node.id));
  const seen = new Set();
  const result = [];
  for (const node of nodes) {
    for (const inputId of node.inputIds) {
      if (regionIds.has(inputId) || seen.has(inputId)) continue;
      seen.add(inputId);
      result.push(inputId);
    }
  }
  return Object.freeze(result);
}

function discoverRegions(plan) {
  const program = plan.program;
  const byId = new Map(program.nodes.map((node) => [node.id, node]));
  const consumers = consumerMap(program);
  const publicOutputs = new Set(program.outputs.map((output) => output.valueId));
  const assigned = new Set();
  const regions = [];

  for (const start of program.nodes) {
    if (!candidate(start) || assigned.has(start.id)) continue;
    const nodes = [start];
    const regionIds = new Set([start.id]);
    let current = start;

    while (!publicOutputs.has(current.id)) {
      const downstream = [...(consumers.get(current.id) ?? [])];
      if (downstream.length !== 1) break;
      const next = byId.get(downstream[0]);
      if (!next || assigned.has(next.id) || !candidate(next) || !next.inputIds.includes(current.id)) break;
      if (!sameElementDomain(start.outputSpec, next.outputSpec)) break;
      if (next.inputIds.some((inputId) => regionIds.has(inputId) && inputId !== current.id)) break;
      nodes.push(next);
      regionIds.add(next.id);
      current = next;
    }

    if (nodes.length < 2) continue;
    for (const node of nodes) assigned.add(node.id);
    const output = nodes.at(-1);
    regions.push(Object.freeze({
      id: `fusion:${regions.length}`,
      firstNode: nodes[0].id,
      outputNode: output.id,
      workItems: output.outputSpec.logicalElementCount,
      nodes: Object.freeze(nodes),
      nodeIds: Object.freeze(nodes.map((node) => node.id)),
      externalInputs: externalInputs(nodes),
      removedMaterials: Object.freeze(nodes.slice(0, -1).map((node) => node.id)),
    }));
  }
  return Object.freeze(regions);
}

export function createFusionProfile(plan, policy = 'none') {
  if (!(plan instanceof TensorPlan)) fail('TENSOR_FUSION_PLAN_INVALID', 'validation', 'Fusion selection requires a TensorPlan.');
  if (!TENSOR_FUSION_POLICIES.includes(policy)) {
    fail('TENSOR_FUSION_POLICY_UNSUPPORTED', 'unsupported', 'fusion must select an accepted finite tensor fusion policy.', { policy, accepted: TENSOR_FUSION_POLICIES });
  }
  const eligible = discoverRegions(plan);
  const regions = policy === 'exact-elementwise' ? eligible : Object.freeze([]);
  const fusedNodeCount = regions.reduce((total, region) => total + region.nodes.length, 0);
  const removedMaterialCount = regions.reduce((total, region) => total + region.removedMaterials.length, 0);
  const canonical = deepFreeze({
    contract: 'SPEC-0007-exact-elementwise-fusion-profile-v1',
    planIdentity: plan.compatibilityIdentity,
    policy,
    eligibleRegionCount: eligible.length,
    selectedRegionCount: regions.length,
    fusedNodeCount,
    removedMaterialCount,
    regions: regions.map((region) => ({
      id: region.id,
      firstNode: region.firstNode,
      outputNode: region.outputNode,
      workItems: region.workItems,
      nodes: region.nodes.map((node) => ({ id: node.id, op: node.op, dtype: node.outputSpec.dtype, specIdentity: node.outputSpec.compatibilityIdentity })),
      externalInputs: [...region.externalInputs],
      removedMaterials: [...region.removedMaterials],
    })),
  });
  return Object.freeze({
    policy,
    eligibleRegionCount: eligible.length,
    regions,
    fusedNodeCount,
    removedMaterialCount,
    canonical,
    compatibilityIdentity: identity('tensor-fusion-profile-v1', canonical),
  });
}
