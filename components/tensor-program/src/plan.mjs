import { checkedAdd, deepFreeze, exactRecord, fail, identity } from './contract.mjs';
import { physicalBytes } from './semantics.mjs';
import { TensorProgram } from './program.mjs';

export const TENSOR_PLAN_CONTRACT = 'SPEC-0004-static-tensor-plan-v1';

const PLAN_FIELDS = new Set(['allocationPolicy']);
const PLAN_TOKEN = Symbol('TensorPlan');
const PLAN_DATA = new WeakMap();

function normalizeOptions(value) {
  const options = value ?? {};
  exactRecord(options, PLAN_FIELDS, 'TENSOR_PLAN_OPTIONS_INVALID', 'TensorPlan options contain unknown fields.');
  const allocationPolicy = options.allocationPolicy ?? 'distinct-v1';
  if (allocationPolicy !== 'distinct-v1') fail('TENSOR_PLAN_ALLOCATION_POLICY_UNSUPPORTED', 'unsupported', 'The first static planner supports only distinct bounded allocations.');
  return Object.freeze({ allocationPolicy });
}

function buildPlan(program, options) {
  const lastUses = new Map();
  const aliasClasses = new Map();
  for (const input of program.inputs) {
    lastUses.set(input.valueId, -1);
    aliasClasses.set(input.valueId, input.spec.aliasGroup === null ? `unresolved-input:${input.name}` : `declared:${input.spec.aliasGroup}`);
  }

  for (let index = 0; index < program.nodes.length; index += 1) {
    const node = program.nodes[index];
    lastUses.set(node.id, index);
    for (const inputId of node.inputIds) lastUses.set(inputId, index);
    aliasClasses.set(node.id, node.materialization === 'view' ? aliasClasses.get(node.inputIds[0]) : `material:${node.id}`);
  }
  for (const output of program.outputs) lastUses.set(output.valueId, program.nodes.length);

  const liveness = [];
  for (const input of program.inputs) liveness.push(Object.freeze({ value: input.valueId, definition: 'input', definedAt: -1, lastUse: lastUses.get(input.valueId), observable: program.outputs.some((entry) => entry.valueId === input.valueId) }));
  for (let index = 0; index < program.nodes.length; index += 1) {
    const node = program.nodes[index];
    liveness.push(Object.freeze({ value: node.id, definition: node.op, definedAt: index, lastUse: lastUses.get(node.id), observable: program.outputs.some((entry) => entry.valueId === node.id) }));
  }

  const aliases = liveness.map((entry) => Object.freeze({
    value: entry.value,
    aliasClass: aliasClasses.get(entry.value),
    runtimeInputAliasingUnresolved: entry.definition === 'input' && program.valueSpec(entry.value).aliasGroup === null,
  }));

  const allocations = [];
  let totalDistinctBytes = 0;
  for (let index = 0; index < program.nodes.length; index += 1) {
    const node = program.nodes[index];
    if (node.materialization !== 'materialize') continue;
    const byteLength = physicalBytes(node.outputSpec);
    totalDistinctBytes = checkedAdd(totalDistinctBytes, byteLength, 'plan.totalDistinctBytes');
    allocations.push(Object.freeze({
      id: `allocation:${node.id}`,
      value: node.id,
      byteLength,
      alignment: node.outputSpec.alignment,
      lifetime: Object.freeze({ definedAt: index, lastUse: lastUses.get(node.id) }),
      reuse: 'none',
    }));
  }

  const inputRequirements = program.inputs.map((entry) => Object.freeze({
    name: entry.name,
    value: entry.valueId,
    specIdentity: entry.spec.compatibilityIdentity,
    access: entry.spec.access,
    declaredAliasGroup: entry.spec.aliasGroup,
    runtimeAliasValidation: 'required',
  }));
  const operations = program.nodes.map((node, index) => Object.freeze({
    index,
    id: node.id,
    op: node.op,
    inputs: node.inputIds,
    output: node.id,
    outputSpecIdentity: node.outputSpec.compatibilityIdentity,
    materialization: node.materialization,
    aliasClass: aliasClasses.get(node.id),
  }));
  const outputs = program.outputs.map((entry) => Object.freeze({ name: entry.name, value: entry.valueId, specIdentity: entry.spec.compatibilityIdentity, aliasClass: aliasClasses.get(entry.valueId) }));
  const unresolved = Object.freeze([
    'runtime-input-aliasing',
    'session-device-compatibility',
    'backend-selection',
    'generated-program-identities',
    'backend-workspace',
    'prepared-execution-products',
    'cleanup-graph',
  ]);
  const canonical = deepFreeze({
    contract: TENSOR_PLAN_CONTRACT,
    programIdentity: program.compatibilityIdentity,
    allocationPolicy: options.allocationPolicy,
    inputRequirements: inputRequirements.map((entry) => ({ ...entry })),
    operations: operations.map((entry) => ({ ...entry, inputs: [...entry.inputs] })),
    liveness: liveness.map((entry) => ({ ...entry })),
    aliases: aliases.map((entry) => ({ ...entry })),
    allocations: allocations.map((entry) => ({ ...entry, lifetime: { ...entry.lifetime } })),
    totalDistinctBytes,
    outputs: outputs.map((entry) => ({ ...entry })),
    unresolved: [...unresolved],
    executable: false,
  });
  return {
    program,
    options,
    inputRequirements: Object.freeze(inputRequirements),
    operations: Object.freeze(operations),
    liveness: Object.freeze(liveness),
    aliases: Object.freeze(aliases),
    allocations: Object.freeze(allocations),
    totalDistinctBytes,
    outputs: Object.freeze(outputs),
    unresolved,
    canonical,
    compatibilityIdentity: identity('tensor-plan-v1', canonical),
  };
}

export class TensorPlan {
  constructor(token, data) {
    if (token !== PLAN_TOKEN) fail('TENSOR_PLAN_CONSTRUCTION_INVALID', 'validation', 'Use TensorPlan.create().');
    PLAN_DATA.set(this, data);
    Object.freeze(this);
  }

  static create(program, options) {
    if (!(program instanceof TensorProgram)) fail('TENSOR_PLAN_PROGRAM_INVALID', 'validation', 'TensorPlan requires a TensorProgram.');
    const normalized = normalizeOptions(options);
    return new TensorPlan(PLAN_TOKEN, buildPlan(program, normalized));
  }

  get kind() { return 'tensor-plan'; }
  get contract() { return TENSOR_PLAN_CONTRACT; }
  get program() { return PLAN_DATA.get(this).program; }
  get compatibilityIdentity() { return PLAN_DATA.get(this).compatibilityIdentity; }
  get allocationPolicy() { return PLAN_DATA.get(this).options.allocationPolicy; }
  get inputRequirements() { return PLAN_DATA.get(this).inputRequirements; }
  get operations() { return PLAN_DATA.get(this).operations; }
  get liveness() { return PLAN_DATA.get(this).liveness; }
  get aliases() { return PLAN_DATA.get(this).aliases; }
  get allocations() { return PLAN_DATA.get(this).allocations; }
  get totalDistinctBytes() { return PLAN_DATA.get(this).totalDistinctBytes; }
  get outputs() { return PLAN_DATA.get(this).outputs; }
  get unresolved() { return PLAN_DATA.get(this).unresolved; }
  get executable() { return false; }
  get canonical() { return PLAN_DATA.get(this).canonical; }
  describe() { return this.canonical; }
  toJSON() { return this.canonical; }
}
