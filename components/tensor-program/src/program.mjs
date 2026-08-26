import { createTensorSpec, TensorSpec } from '../../tensor-value/index.mjs';
import { isDeepStrictEqual } from 'node:util';
import { boundedName, deepFreeze, exactRecord, fail, identity, plainObject } from './contract.mjs';
import { inferOperation } from './semantics.mjs';

export const TENSOR_PROGRAM_CONTRACT = 'SPEC-0004-tensor-program-v1';
export const TENSOR_PROGRAM_LIMITS = Object.freeze({ maxInputs: 256, maxNodes: 4_096, maxOutputs: 256 });

const PROGRAM_FIELDS = new Set(['inputs', 'nodes', 'outputs']);
const INPUT_FIELDS = new Set(['name', 'spec']);
const NODE_FIELDS = new Set(['id', 'op', 'inputs', 'options']);
const OUTPUT_FIELDS = new Set(['name', 'value']);
const NORMALIZED_FIELDS = new Set(['contract', 'limits', 'inputs', 'nodes', 'outputs']);
const PROGRAM_TOKEN = Symbol('TensorProgram');
const REF_TOKEN = Symbol('TensorValueRef');
const PROGRAM_DATA = new WeakMap();
const REF_DATA = new WeakMap();

export class TensorValueRef {
  constructor(token, owner, id, spec) {
    if (token !== REF_TOKEN) fail('TENSOR_PROGRAM_VALUE_CONSTRUCTION_INVALID', 'validation', 'Tensor values are created inside TensorProgram.define().');
    REF_DATA.set(this, { owner, id, spec });
    Object.freeze(this);
  }

  get kind() { return 'tensor-program-value'; }
  get id() { return REF_DATA.get(this).id; }
  get spec() { return REF_DATA.get(this).spec; }
}

function refData(value, owner, field) {
  const data = REF_DATA.get(value);
  if (!data || data.owner !== owner) fail('TENSOR_PROGRAM_VALUE_INVALID', 'validation', `${field} must be a value from this program builder.`);
  return data;
}

function normalizeSpec(value, field) {
  try { return value instanceof TensorSpec ? value : createTensorSpec(value); } catch (error) {
    if (error?.code) throw error;
    fail('TENSOR_PROGRAM_SPEC_INVALID', 'validation', `${field} is not a valid TensorSpec.`);
  }
}

class ProgramBuilder {
  #owner = Object.freeze({});
  #inputs = [];
  #nodes = [];
  #outputs = [];
  #values = new Map();
  #names = new Set();
  #sealed = false;

  #assertOpen() {
    if (this.#sealed) fail('TENSOR_PROGRAM_BUILDER_CLOSED', 'closed-program', 'Program builder is already sealed.');
  }

  input(name, spec) {
    this.#assertOpen();
    boundedName(name, 'input.name');
    if (this.#names.has(name)) fail('TENSOR_PROGRAM_NAME_DUPLICATE', 'validation', 'Input name is duplicated.', { name });
    if (this.#inputs.length >= TENSOR_PROGRAM_LIMITS.maxInputs) fail('TENSOR_PROGRAM_LIMIT', 'pressure', 'Program exceeds maxInputs.');
    const normalized = normalizeSpec(spec, 'input.spec');
    const id = `input:${name}`;
    const ref = new TensorValueRef(REF_TOKEN, this.#owner, id, normalized);
    this.#inputs.push(Object.freeze({ name, valueId: id, spec: normalized }));
    this.#values.set(id, ref);
    this.#names.add(name);
    return ref;
  }

  node(op, inputs, options = {}, id = undefined) {
    this.#assertOpen();
    if (this.#nodes.length >= TENSOR_PROGRAM_LIMITS.maxNodes) fail('TENSOR_PROGRAM_LIMIT', 'pressure', 'Program exceeds maxNodes.');
    if (!Array.isArray(inputs)) fail('TENSOR_PROGRAM_OPERATION_INPUTS_INVALID', 'validation', 'Node inputs must be an array of program values.');
    const inputData = inputs.map((value, index) => refData(value, this.#owner, `node.inputs[${index}]`));
    const nodeId = id ?? `node:${this.#nodes.length}`;
    boundedName(nodeId, 'node.id');
    if (this.#values.has(nodeId)) fail('TENSOR_PROGRAM_VALUE_DUPLICATE', 'validation', 'Node ID is duplicated.', { id: nodeId });
    const normalizedOptions = op === 'fill' && options?.spec !== undefined && !(options.spec instanceof TensorSpec)
      ? { ...options, spec: normalizeSpec(options.spec, 'fill.spec') }
      : options;
    const inferred = inferOperation(op, inputData.map((entry) => entry.spec), normalizedOptions);
    const ref = new TensorValueRef(REF_TOKEN, this.#owner, nodeId, inferred.outputSpec);
    this.#nodes.push(Object.freeze({
      id: nodeId,
      op,
      inputIds: Object.freeze(inputData.map((entry) => entry.id)),
      options: inferred.options,
      outputSpec: inferred.outputSpec,
      materialization: inferred.materialization,
    }));
    this.#values.set(nodeId, ref);
    return ref;
  }

  fill(spec, value) { return this.node('fill', [], { spec, value }); }
  copy(input) { return this.node('copy', [input]); }
  cast(input, dtype) { return this.node('cast', [input], { dtype }); }
  reshape(input, capacityShape) { return this.node('reshape', [input], { capacityShape }); }
  permute(input, axes) { return this.node('permute', [input], { axes }); }
  slice(input, slices) { return this.node('slice', [input], { slices }); }
  broadcast(input, capacityShape) { return this.node('broadcast', [input], { capacityShape }); }
  contiguous(input) { return this.node('contiguous', [input]); }
  unary(operator, input) { return this.node('unary', [input], { operator }); }
  binary(operator, left, right) { return this.node('binary', [left, right], { operator }); }
  reduce(operator, input, options = {}) { return this.node('reduce', [input], { ...options, operator }); }
  matmul(left, right, options = {}) { return this.node('matmul', [left, right], options); }

  output(name, value) {
    this.#assertOpen();
    boundedName(name, 'output.name');
    if (this.#outputs.some((entry) => entry.name === name)) fail('TENSOR_PROGRAM_NAME_DUPLICATE', 'validation', 'Output name is duplicated.', { name });
    if (this.#outputs.length >= TENSOR_PROGRAM_LIMITS.maxOutputs) fail('TENSOR_PROGRAM_LIMIT', 'pressure', 'Program exceeds maxOutputs.');
    const data = refData(value, this.#owner, 'output.value');
    this.#outputs.push(Object.freeze({ name, valueId: data.id, spec: data.spec }));
    return value;
  }

  value(id) {
    const value = this.#values.get(id);
    if (!value) fail('TENSOR_PROGRAM_REFERENCE_INVALID', 'validation', 'Program references an unknown or forward value.', { id });
    return value;
  }

  acceptReturnedOutputs(value) {
    if (value === undefined) return;
    if (this.#outputs.length > 0) fail('TENSOR_PROGRAM_OUTPUTS_AMBIGUOUS', 'validation', 'Use either builder.output() or returned outputs, not both.');
    if (REF_DATA.has(value)) { this.output('output', value); return; }
    if (!plainObject(value)) fail('TENSOR_PROGRAM_OUTPUTS_INVALID', 'validation', 'Builder callback must return one program value or an output-name record.');
    for (const [name, ref] of Object.entries(value)) this.output(name, ref);
  }

  build() {
    this.#assertOpen();
    if (this.#outputs.length < 1) fail('TENSOR_PROGRAM_OUTPUTS_INVALID', 'validation', 'Program requires at least one observable output.');
    this.#sealed = true;
    return constructProgram(this.#inputs, this.#nodes, this.#outputs);
  }
}

function canonicalProgram(inputs, nodes, outputs) {
  return deepFreeze({
    contract: TENSOR_PROGRAM_CONTRACT,
    limits: { ...TENSOR_PROGRAM_LIMITS },
    inputs: inputs.map((entry) => ({ name: entry.name, value: entry.valueId, spec: entry.spec.canonical })),
    nodes: nodes.map((entry) => ({
      id: entry.id,
      op: entry.op,
      inputs: [...entry.inputIds],
      options: entry.options,
      materialization: entry.materialization,
      outputSpec: entry.outputSpec.canonical,
    })),
    outputs: outputs.map((entry) => ({ name: entry.name, value: entry.valueId, spec: entry.spec.canonical })),
  });
}

function constructProgram(inputs, nodes, outputs) {
  const canonical = canonicalProgram(inputs, nodes, outputs);
  return new TensorProgram(PROGRAM_TOKEN, {
    inputs: Object.freeze([...inputs]),
    nodes: Object.freeze([...nodes]),
    outputs: Object.freeze([...outputs]),
    canonical,
    compatibilityIdentity: identity('tensor-program-v1', canonical),
    specs: new Map([...inputs.map((entry) => [entry.valueId, entry.spec]), ...nodes.map((entry) => [entry.id, entry.outputSpec])]),
  });
}

function fromCanonicalRecord(value) {
  exactRecord(value, PROGRAM_FIELDS, 'TENSOR_PROGRAM_OPTIONS_INVALID', 'TensorProgram options contain unknown fields.');
  if (!Array.isArray(value.inputs) || !Array.isArray(value.nodes) || !Array.isArray(value.outputs)) fail('TENSOR_PROGRAM_OPTIONS_INVALID', 'validation', 'inputs, nodes and outputs must be arrays.');
  const builder = new ProgramBuilder();
  for (const input of value.inputs) {
    exactRecord(input, INPUT_FIELDS, 'TENSOR_PROGRAM_INPUT_INVALID', 'Input record contains unknown fields.');
    builder.input(input.name, input.spec);
  }
  for (const node of value.nodes) {
    exactRecord(node, NODE_FIELDS, 'TENSOR_PROGRAM_NODE_INVALID', 'Node record contains unknown fields.');
    if (!Array.isArray(node.inputs)) fail('TENSOR_PROGRAM_NODE_INVALID', 'validation', 'Node inputs must be value ID strings.');
    builder.node(node.op, node.inputs.map((id) => builder.value(id)), node.options ?? {}, node.id);
  }
  for (const output of value.outputs) {
    exactRecord(output, OUTPUT_FIELDS, 'TENSOR_PROGRAM_OUTPUT_INVALID', 'Output record contains unknown fields.');
    builder.output(output.name, builder.value(output.value));
  }
  return builder.build();
}

function specOptionsFromCanonical(spec) {
  if (!plainObject(spec)) fail('TENSOR_PROGRAM_CANONICAL_INVALID', 'validation', 'Canonical TensorSpec record is invalid.');
  return {
    dtype: spec.dtype,
    capacityShape: spec.capacityShape,
    activeAxis0: spec.activeAxis0,
    strides: spec.strides,
    byteOffset: spec.byteOffset,
    alignment: spec.alignment,
    access: spec.access,
    aliasGroup: spec.aliasGroup,
  };
}

function scalarFromCanonical(record) {
  if (!plainObject(record) || typeof record.dtype !== 'string' || !Object.hasOwn(record, 'value')) fail('TENSOR_PROGRAM_CANONICAL_INVALID', 'validation', 'Canonical scalar record is invalid.');
  if (record.dtype === 'u64') {
    try { return BigInt(record.value); } catch { fail('TENSOR_PROGRAM_CANONICAL_INVALID', 'validation', 'Canonical u64 scalar is invalid.'); }
  }
  if (record.value === 'Infinity') return Infinity;
  if (record.value === '-Infinity') return -Infinity;
  if (record.value === '-0') return -0;
  return record.value;
}

function sourceOptions(node) {
  const options = node.options;
  if (!plainObject(options)) fail('TENSOR_PROGRAM_CANONICAL_INVALID', 'validation', 'Canonical node options are invalid.');
  if (node.op === 'fill') return { spec: specOptionsFromCanonical(options.spec), value: scalarFromCanonical(options.value) };
  if (node.op === 'copy' || node.op === 'contiguous') return {};
  if (node.op === 'cast') return { dtype: options.dtype };
  if (node.op === 'reshape' || node.op === 'broadcast') return { capacityShape: options.capacityShape };
  if (node.op === 'permute') return { axes: options.axes };
  if (node.op === 'slice') return { slices: options.slices };
  if (node.op === 'unary' || node.op === 'binary') return { operator: options.operator };
  if (node.op === 'reduce') return {
    operator: options.operator,
    axes: options.axes,
    keepDimensions: options.keepDimensions,
    accumulatorDtype: options.accumulatorDtype,
    order: options.order,
    identity: scalarFromCanonical(options.identity),
  };
  if (node.op === 'matmul') return { transposeA: options.transposeA, transposeB: options.transposeB, accumulatorDtype: options.accumulatorDtype };
  fail('TENSOR_PROGRAM_CANONICAL_INVALID', 'validation', 'Canonical node operation is invalid.');
}

function fromNormalizedCanonical(value) {
  exactRecord(value, NORMALIZED_FIELDS, 'TENSOR_PROGRAM_CANONICAL_INVALID', 'Normalized canonical program contains unknown fields.');
  const source = {
    inputs: value.inputs?.map((entry) => ({ name: entry.name, spec: specOptionsFromCanonical(entry.spec) })),
    nodes: value.nodes?.map((entry) => ({ id: entry.id, op: entry.op, inputs: entry.inputs, options: sourceOptions(entry) })),
    outputs: value.outputs?.map((entry) => ({ name: entry.name, value: entry.value })),
  };
  const rebuilt = fromCanonicalRecord(source);
  if (!isDeepStrictEqual(rebuilt.canonical, value)) fail('TENSOR_PROGRAM_CANONICAL_DIVERGENCE', 'validation', 'Normalized canonical program contains derived facts that do not match semantic inference.');
  return rebuilt;
}

export class TensorProgram {
  constructor(token, data) {
    if (token !== PROGRAM_TOKEN) fail('TENSOR_PROGRAM_CONSTRUCTION_INVALID', 'validation', 'Use TensorProgram.create() or TensorProgram.define().');
    PROGRAM_DATA.set(this, data);
    Object.freeze(this);
  }

  static create(record) { return record?.contract === TENSOR_PROGRAM_CONTRACT ? fromNormalizedCanonical(record) : fromCanonicalRecord(record); }

  static define(callback) {
    if (typeof callback !== 'function') fail('TENSOR_PROGRAM_BUILDER_INVALID', 'validation', 'TensorProgram.define requires a synchronous callback.');
    if (callback.constructor?.name === 'AsyncFunction') fail('TENSOR_PROGRAM_BUILDER_ASYNC', 'unsupported', 'Program builder callbacks must be synchronous and finite.');
    const builder = new ProgramBuilder();
    const result = callback(builder);
    if (result && typeof result.then === 'function') fail('TENSOR_PROGRAM_BUILDER_ASYNC', 'unsupported', 'Program builder callbacks must be synchronous and finite.');
    builder.acceptReturnedOutputs(result);
    return builder.build();
  }

  get kind() { return 'tensor-program'; }
  get contract() { return TENSOR_PROGRAM_CONTRACT; }
  get compatibilityIdentity() { return PROGRAM_DATA.get(this).compatibilityIdentity; }
  get inputs() { return PROGRAM_DATA.get(this).inputs; }
  get nodes() { return PROGRAM_DATA.get(this).nodes; }
  get outputs() { return PROGRAM_DATA.get(this).outputs; }
  get canonical() { return PROGRAM_DATA.get(this).canonical; }

  valueSpec(id) {
    const spec = PROGRAM_DATA.get(this).specs.get(id);
    if (!spec) fail('TENSOR_PROGRAM_REFERENCE_INVALID', 'validation', 'Program value ID is unknown.', { id });
    return spec;
  }

  describe() { return this.canonical; }
  toJSON() { return this.canonical; }
}
