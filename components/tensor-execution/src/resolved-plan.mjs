import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { inspectTensorForSession, inspectTensorSessionForExecution, reserveTensorSessionExecution } from '../../tensor-value/internal.mjs';

import { createBackendProfileRequest, realizeBackendProfile, TENSOR_BACKEND_POLICIES } from './backend-profile.mjs';
import { createCudaJsTensorBackend } from './cuda-js-adapter.mjs';
import { deepFreeze, exactRecord, fail, failureSummary, identity, plainObject, RESOLVED_TENSOR_PLAN_CONTRACT, TENSOR_EXECUTION_RESULT_CONTRACT, TENSOR_SIMT_LIMITS } from './contract.mjs';
import { TENSOR_FUSION_POLICIES } from './fusion-profile.mjs';
import { lowerSimtPlan } from './lowering.mjs';

const RESOLVE_FIELDS = new Set(['backend', 'blockSize', 'maxWorkspaceBytes', 'fusion']);
const BLOCK_SIZES = new Set([32, 64, 128, 256, 512, 1024]);
const RESOLVED_TOKEN = Symbol('ResolvedTensorPlan');
const RESULT_TOKEN = Symbol('TensorExecutionResult');
const RESOLVED_DATA = new WeakMap();
const RESULT_DATA = new WeakMap();

function normalizeOptions(value) {
  const options = value ?? {};
  exactRecord(options, RESOLVE_FIELDS, 'TENSOR_RESOLVE_OPTIONS_INVALID', 'ResolvedTensorPlan options contain unknown fields.');
  const backend = options.backend ?? 'simt';
  if (!TENSOR_BACKEND_POLICIES.includes(backend)) fail('TENSOR_BACKEND_UNSUPPORTED', 'unsupported', 'backend must select an accepted finite tensor backend policy.', { backend, accepted: TENSOR_BACKEND_POLICIES });
  const blockSize = options.blockSize ?? 256;
  if (!BLOCK_SIZES.has(blockSize)) fail('TENSOR_SIMT_BLOCK_SIZE_INVALID', 'validation', 'blockSize must be one of the finite CUDA block sizes.', { blockSize });
  const maxWorkspaceBytes = options.maxWorkspaceBytes ?? TENSOR_SIMT_LIMITS.maxWorkspaceBytes;
  if (!Number.isSafeInteger(maxWorkspaceBytes) || maxWorkspaceBytes < 1 || maxWorkspaceBytes > TENSOR_SIMT_LIMITS.maxWorkspaceBytes) {
    fail('TENSOR_SIMT_WORKSPACE_LIMIT_INVALID', 'validation', 'maxWorkspaceBytes must be a positive safe integer within the first SIMT profile.', { maximum: TENSOR_SIMT_LIMITS.maxWorkspaceBytes });
  }
  const fusion = options.fusion ?? 'none';
  if (!TENSOR_FUSION_POLICIES.includes(fusion)) fail('TENSOR_FUSION_POLICY_UNSUPPORTED', 'unsupported', 'fusion must select an accepted finite tensor fusion policy.', { fusion, accepted: TENSOR_FUSION_POLICIES });
  return Object.freeze({ backend, blockSize, maxWorkspaceBytes, fusion });
}

function staticPlan(value) {
  if (value instanceof TensorPlan) return value;
  if (value instanceof TensorProgram) return TensorPlan.create(value);
  fail('TENSOR_RESOLVE_PLAN_INVALID', 'validation', 'ResolvedTensorPlan requires a TensorPlan or TensorProgram.');
}

function ensurePlanFitsSession(lowering, sessionInspection, acceleratorWorkspaces = []) {
  const acceleratorWorkspaceBytes = acceleratorWorkspaces.reduce((total, workspace) => total + workspace.byteLength, 0);
  const required = lowering.materialBytes + lowering.totalWorkspaceBytes + acceleratorWorkspaceBytes;
  if (!Number.isSafeInteger(required) || required > sessionInspection.limits.maxSessionBytes) {
    fail('TENSOR_RESOLVE_SESSION_BYTE_LIMIT', 'pressure', 'The resolved plan cannot fit the session byte limit even with no other live tensors.', { required, maximum: sessionInspection.limits.maxSessionBytes });
  }
  const requiredLiveTensors = lowering.materials.length + lowering.workspaces.length + acceleratorWorkspaces.length;
  if (requiredLiveTensors > sessionInspection.limits.maxLiveTensors) {
    fail('TENSOR_RESOLVE_SESSION_TENSOR_LIMIT', 'pressure', 'The resolved plan cannot fit the session live-tensor limit even with no other live tensors.', { required: requiredLiveTensors, maximum: sessionInspection.limits.maxLiveTensors });
  }
  for (const material of lowering.materials) {
    if (material.byteLength > sessionInspection.limits.maxTensorBytes) fail('TENSOR_RESOLVE_TENSOR_BYTE_LIMIT', 'pressure', 'A planned tensor exceeds the session per-tensor limit.', { value: material.valueId, required: material.byteLength, maximum: sessionInspection.limits.maxTensorBytes });
  }
  for (const workspace of lowering.workspaces) {
    if (workspace.byteLength > sessionInspection.limits.maxTensorBytes) fail('TENSOR_RESOLVE_TENSOR_BYTE_LIMIT', 'pressure', 'A planned workspace exceeds the session per-tensor limit.', { value: workspace.id, required: workspace.byteLength, maximum: sessionInspection.limits.maxTensorBytes });
  }
  for (const workspace of acceleratorWorkspaces) {
    if (workspace.byteLength > sessionInspection.limits.maxTensorBytes) fail('TENSOR_RESOLVE_TENSOR_BYTE_LIMIT', 'pressure', 'A planned accelerator workspace exceeds the session per-tensor limit.', { value: workspace.id, required: workspace.byteLength, maximum: sessionInspection.limits.maxTensorBytes });
  }
  for (const output of lowering.outputs) {
    const base = lowering.references.get(output.baseValueId);
    if (base.baseValueId.startsWith('input:') && base.spec.layout !== 'row-major-contiguous' && output.spec.compatibilityIdentity !== base.spec.compatibilityIdentity) {
      fail('TENSOR_SIMT_STRIDED_INPUT_VIEW_OUTPUT_UNSUPPORTED', 'unsupported', 'A changed public output view over a strided runtime input is not representable by the current Tensor value port; materialize it first.', { output: output.name });
    }
  }
}

function resolvedData(value, operation) {
  const data = RESOLVED_DATA.get(value);
  if (!data) fail('TENSOR_RESOLVED_PLAN_INVALID', 'validation', `${operation} requires a ResolvedTensorPlan capability.`);
  return data;
}

function resultData(value, operation) {
  const data = RESULT_DATA.get(value);
  if (!data) fail('TENSOR_EXECUTION_RESULT_INVALID', 'validation', `${operation} requires a TensorExecutionResult capability.`);
  return data;
}

function normalizeBindings(program, value) {
  const inputs = program.inputs;
  if (inputs.length === 0) {
    if (value === undefined || (Array.isArray(value) && value.length === 0) || (plainObject(value) && Object.keys(value).length === 0)) return Object.freeze({});
    fail('TENSOR_EXECUTION_BINDINGS_INVALID', 'validation', 'A no-input tensor program accepts only omitted or empty bindings.');
  }
  if (inputs.length === 1 && !Array.isArray(value) && !plainObject(value)) return Object.freeze({ [inputs[0].name]: value });
  if (Array.isArray(value)) {
    if (value.length !== inputs.length) fail('TENSOR_EXECUTION_BINDINGS_INVALID', 'validation', 'Positional tensor bindings must match the program input count.', { expected: inputs.length, actual: value.length });
    return Object.freeze(Object.fromEntries(inputs.map((entry, index) => [entry.name, value[index]])));
  }
  if (!plainObject(value)) fail('TENSOR_EXECUTION_BINDINGS_INVALID', 'validation', 'Tensor bindings must be a named record, positional array, or the sole Tensor input.');
  const expected = inputs.map((entry) => entry.name);
  const actual = Object.keys(value);
  if (actual.length !== expected.length || expected.some((name) => !Object.hasOwn(value, name))) fail('TENSOR_EXECUTION_BINDINGS_INVALID', 'validation', 'Named tensor bindings must be exact.', { expected, actual });
  return Object.freeze(Object.fromEntries(expected.map((name) => [name, value[name]])));
}

function validateInputAliases(program, inspections) {
  for (let left = 0; left < program.inputs.length; left += 1) {
    const leftInput = program.inputs[left];
    if (leftInput.spec.aliasGroup === null) continue;
    for (let right = left + 1; right < program.inputs.length; right += 1) {
      const rightInput = program.inputs[right];
      if (rightInput.spec.aliasGroup === null) continue;
      const declaredSame = leftInput.spec.aliasGroup === rightInput.spec.aliasGroup;
      const actualSame = inspections.get(leftInput.valueId).storageIdentity === inspections.get(rightInput.valueId).storageIdentity;
      if (declaredSame !== actualSame) fail('TENSOR_EXECUTION_ALIAS_MISMATCH', 'validation', 'Runtime tensor aliases do not match the program declaration.', { left: leftInput.name, right: rightInput.name, declaredSame, actualSame });
    }
  }
}

async function closeTensorList(tensors) {
  const failures = [];
  for (const tensor of [...tensors].reverse()) {
    if (!tensor || ['closed', 'orphaned'].includes(tensor.state)) continue;
    try {
      const report = await tensor.close();
      if (report?.disposition !== 'closed') failures.push(Object.freeze({ code: 'TENSOR_CLEANUP_UNPROVED', category: 'cleanup-unproved', name: 'TensorError' }));
    } catch (error) {
      failures.push(failureSummary(error));
    }
  }
  return Object.freeze({ failures: Object.freeze(failures), retryable: failures.length > 0 && failures.every((entry) => entry.category === 'backpressure') });
}

async function rollbackRun(outputViews, workspaces, materials, primary) {
  const reports = [await closeTensorList(outputViews), await closeTensorList(workspaces), await closeTensorList(materials)];
  const failures = reports.flatMap((entry) => entry.failures);
  if (failures.length > 0) fail('TENSOR_EXECUTION_ROLLBACK_UNPROVED', 'cleanup-unproved', 'Tensor execution failed and per-run resource rollback was not proved.', { primary: failureSummary(primary), cleanup: failures }, { cause: primary });
}

export class TensorExecutionResult {
  constructor(token, data) {
    if (token !== RESULT_TOKEN) fail('TENSOR_EXECUTION_RESULT_CONSTRUCTION_INVALID', 'validation', 'Tensor execution results are created by ResolvedTensorPlan.run().');
    RESULT_DATA.set(this, data);
    Object.freeze(this);
  }

  get kind() { return 'tensor-execution-result'; }
  get contract() { return TENSOR_EXECUTION_RESULT_CONTRACT; }
  get state() { return resultData(this, 'TensorExecutionResult.state').state; }
  get outputs() { return resultData(this, 'TensorExecutionResult.outputs').outputs; }
  get output() {
    const data = resultData(this, 'TensorExecutionResult.output');
    return data.outputNames.length === 1 ? data.outputs[data.outputNames[0]] : null;
  }
  get execution() { return resultData(this, 'TensorExecutionResult.execution').execution; }
  get compatibilityIdentity() { return resultData(this, 'TensorExecutionResult.compatibilityIdentity').compatibilityIdentity; }

  get(name) {
    const data = resultData(this, 'TensorExecutionResult.get');
    if (typeof name !== 'string' || !Object.hasOwn(data.outputs, name)) fail('TENSOR_EXECUTION_OUTPUT_UNKNOWN', 'validation', 'Tensor execution output name is unknown.', { name: typeof name === 'string' ? name : null });
    return data.outputs[name];
  }

  async close() {
    const data = resultData(this, 'TensorExecutionResult.close');
    if (data.state === 'closed' || data.state === 'orphaned') return data.terminal;
    data.state = 'closing';
    const reports = [await closeTensorList(data.outputViews), await closeTensorList(data.workspaces), await closeTensorList(data.materials)];
    const failures = reports.flatMap((entry) => entry.failures);
    if (failures.length > 0 && failures.every((entry) => entry.category === 'backpressure')) {
      data.state = 'closing';
      fail('TENSOR_EXECUTION_RESULT_BUSY', 'backpressure', 'TensorExecutionResult retains live child resources and may be closed again after they become terminal.', { failures });
    }
    data.state = failures.length === 0 ? 'closed' : 'orphaned';
    data.terminal = deepFreeze({ schemaVersion: 1, kind: this.kind, state: data.state, graceful: failures.length === 0, failures });
    data.owner.results.delete(this);
    return data.terminal;
  }
}

export class ResolvedTensorPlan {
  constructor(token, data) {
    if (token !== RESOLVED_TOKEN) fail('TENSOR_RESOLVED_PLAN_CONSTRUCTION_INVALID', 'validation', 'Use ResolvedTensorPlan.create().');
    RESOLVED_DATA.set(this, data);
    Object.freeze(this);
  }

  static create(session, planOrProgram, options) { return resolveTensorPlan(session, planOrProgram, options); }

  get kind() { return 'resolved-tensor-plan'; }
  get contract() { return RESOLVED_TENSOR_PLAN_CONTRACT; }
  get state() { return resolvedData(this, 'ResolvedTensorPlan.state').state; }
  get plan() { return resolvedData(this, 'ResolvedTensorPlan.plan').plan; }
  get backend() { return resolvedData(this, 'ResolvedTensorPlan.backend').profile.backend; }
  get backendPolicy() { return resolvedData(this, 'ResolvedTensorPlan.backendPolicy').options.backend; }
  get fusionPolicy() { return resolvedData(this, 'ResolvedTensorPlan.fusionPolicy').options.fusion; }
  get fusionRegionCount() { return resolvedData(this, 'ResolvedTensorPlan.fusionRegionCount').lowering.fusionProfile.regions.length; }
  get fusedNodeCount() { return resolvedData(this, 'ResolvedTensorPlan.fusedNodeCount').lowering.fusionProfile.fusedNodeCount; }
  get compatibilityIdentity() { return resolvedData(this, 'ResolvedTensorPlan.compatibilityIdentity').compatibilityIdentity; }
  get kernelCount() { return resolvedData(this, 'ResolvedTensorPlan.kernelCount').profile.simtNodeCount; }
  get cublasLtNodeCount() { return resolvedData(this, 'ResolvedTensorPlan.cublasLtNodeCount').profile.cublasLtNodeCount; }
  get bindingCount() { return resolvedData(this, 'ResolvedTensorPlan.bindingCount').bindingRecords.length; }
  get workspaceBytes() { return resolvedData(this, 'ResolvedTensorPlan.workspaceBytes').workspaceBytes; }
  get canonical() { return resolvedData(this, 'ResolvedTensorPlan.canonical').canonical; }
  describe() { return this.canonical; }

  async run(bindings) {
    const data = resolvedData(this, 'ResolvedTensorPlan.run');
    if (data.state !== 'open') fail('TENSOR_RESOLVED_PLAN_CLOSED', 'closed-plan', 'ResolvedTensorPlan.run requires an open plan.', { state: data.state });
    if (data.pendingRuns > 0) fail('TENSOR_RESOLVED_PLAN_BUSY', 'backpressure', 'The first resolved-plan profile permits one run at a time.');
    inspectTensorSessionForExecution(data.session);
    const normalized = normalizeBindings(data.plan.program, bindings);
    const inputInspections = new Map();
    const baseTensors = new Map();
    for (const input of data.plan.program.inputs) {
      const tensor = normalized[input.name];
      const inspection = inspectTensorForSession(data.session, tensor);
      if (inspection.spec.compatibilityIdentity !== input.spec.compatibilityIdentity) fail('TENSOR_EXECUTION_INPUT_SPEC_MISMATCH', 'validation', 'Runtime tensor spec does not match the resolved program input.', { input: input.name, expected: input.spec.compatibilityIdentity, actual: inspection.spec.compatibilityIdentity });
      inputInspections.set(input.valueId, inspection);
      baseTensors.set(input.valueId, tensor);
    }
    validateInputAliases(data.plan.program, inputInspections);
    data.pendingRuns += 1;
    const materials = [];
    const workspaces = [];
    const outputViews = [];
    try {
      for (const material of data.lowering.materials) {
        const tensor = await data.session.allocate(material.spec);
        materials.push(tensor);
        baseTensors.set(material.valueId, tensor);
      }
      const workspaceById = new Map();
      for (const workspace of data.workspaces) {
        const tensor = await data.session.allocate({ dtype: workspace.dtype, capacityShape: [workspace.elementCount], access: 'read-write' });
        workspaces.push(tensor);
        workspaceById.set(workspace.id, tensor);
      }

      const cudaBindings = {};
      for (const binding of data.bindingRecords) {
        let tensor;
        if (binding.role === 'input' || binding.role === 'material') tensor = baseTensors.get(binding.valueId);
        else tensor = workspaceById.get(binding.valueId);
        if (!tensor) continue;
        cudaBindings[binding.name] = inspectTensorForSession(data.session, tensor).deviceView;
      }
      const execution = await data.backendAdapter.execute(Object.freeze(cudaBindings));
      const outputByValue = new Map();
      const outputs = {};
      for (const output of data.lowering.outputs) {
        let tensor = outputByValue.get(output.valueId);
        if (!tensor) {
          const base = baseTensors.get(output.baseValueId);
          if (!base) fail('TENSOR_EXECUTION_OUTPUT_BASE_MISSING', 'internal', 'A resolved tensor output lost its storage owner.', { output: output.name });
          tensor = base.spec.compatibilityIdentity === output.spec.compatibilityIdentity ? base : await base.view(output.spec);
          if (tensor !== base) outputViews.push(tensor);
          outputByValue.set(output.valueId, tensor);
        }
        outputs[output.name] = tensor;
      }
      const outputNames = data.plan.program.outputs.map((entry) => entry.name);
      const canonicalResult = { plan: data.compatibilityIdentity, runSequence: ++data.runSequence, outputs: outputNames.map((name) => ({ name, specIdentity: outputs[name].spec.compatibilityIdentity })), execution };
      const result = new TensorExecutionResult(RESULT_TOKEN, {
        state: 'open',
        outputs: Object.freeze(outputs),
        outputNames: Object.freeze(outputNames),
        execution,
        materials: Object.freeze(materials),
        workspaces: Object.freeze(workspaces),
        outputViews: Object.freeze(outputViews),
        compatibilityIdentity: identity('tensor-execution-result-v1', canonicalResult),
        terminal: null,
        owner: data,
      });
      data.results.add(result);
      return result;
    } catch (error) {
      await rollbackRun(outputViews, workspaces, materials, error);
      throw error;
    } finally {
      data.pendingRuns -= 1;
    }
  }

  async close() {
    const data = resolvedData(this, 'ResolvedTensorPlan.close');
    if (data.state === 'closed' || data.state === 'orphaned') return data.terminal;
    if (data.pendingRuns > 0) fail('TENSOR_RESOLVED_PLAN_BUSY', 'backpressure', 'ResolvedTensorPlan cannot close while a run is pending.');
    const resultFailures = [];
    for (const result of [...data.results].reverse()) {
      try {
        const report = await result.close();
        if (report?.graceful !== true) resultFailures.push(Object.freeze({ code: 'TENSOR_EXECUTION_RESULT_CLEANUP_UNPROVED', category: 'cleanup-unproved', name: 'TensorError' }));
      } catch (error) {
        if (error?.category === 'backpressure') fail('TENSOR_RESOLVED_PLAN_BUSY', 'backpressure', 'ResolvedTensorPlan retains a busy execution result and may be closed again after its child resources become terminal.', { causeCode: error.code ?? null }, { cause: error });
        resultFailures.push(failureSummary(error));
      }
    }
    data.state = 'closing';
    const backend = await data.backendAdapter.close();
    const graceful = backend.graceful && resultFailures.length === 0;
    data.state = graceful ? 'closed' : 'orphaned';
    data.terminal = deepFreeze({ schemaVersion: 1, kind: this.kind, state: data.state, graceful, resultFailures, backend });
    data.sessionRegistration.release(data.terminal);
    return data.terminal;
  }
}

export async function resolveTensorPlanWithAdapter(session, planOrProgram, options, adapterFactory) {
  const plan = staticPlan(planOrProgram);
  const normalized = normalizeOptions(options);
  const reservation = reserveTensorSessionExecution(session);
  let backendAdapter = null;
  try {
    const sessionInspection = inspectTensorSessionForExecution(session);
    const lowering = lowerSimtPlan(plan, normalized);
    ensurePlanFitsSession(lowering, sessionInspection);
    const profileRequest = createBackendProfileRequest(plan, lowering, normalized);
    const resourceLimits = Object.freeze({
      acceleratorWorkspaceBytes: Math.max(0, Math.min(
        normalized.maxWorkspaceBytes - lowering.totalWorkspaceBytes,
        sessionInspection.limits.maxSessionBytes - lowering.materialBytes - lowering.totalWorkspaceBytes,
      )),
      maxAcceleratorWorkspaceBytesPerPlan: sessionInspection.limits.maxTensorBytes,
      maxAcceleratorWorkspaceCount: Math.max(0, sessionInspection.limits.maxLiveTensors - lowering.materials.length - lowering.workspaces.length),
    });
    backendAdapter = await adapterFactory(sessionInspection.runtime, lowering, profileRequest, Object.freeze({ ...normalized, ...resourceLimits }));
    const profile = backendAdapter.profile ?? realizeBackendProfile(profileRequest, lowering, []);
    const acceleratorWorkspaces = backendAdapter.workspaces ?? Object.freeze([]);
    ensurePlanFitsSession(lowering, sessionInspection, acceleratorWorkspaces);
    const bindingRecords = backendAdapter.bindingRecords ?? lowering.bindings;
    const workspaces = Object.freeze([...lowering.workspaces, ...acceleratorWorkspaces]);
    const workspaceBytes = backendAdapter.workspaceBytes ?? lowering.totalWorkspaceBytes;
    const canonical = deepFreeze({
      contract: RESOLVED_TENSOR_PLAN_CONTRACT,
      planIdentity: plan.compatibilityIdentity,
      sessionCompatibilityIdentity: sessionInspection.compatibilityIdentity,
      backend: profile.backend,
      backendPolicy: normalized.backend,
      fusionPolicy: normalized.fusion,
      options: { ...normalized },
      resourceLimits: { ...resourceLimits },
      loweringIdentity: lowering.compatibilityIdentity,
      fusionProfileIdentity: lowering.fusionProfile.compatibilityIdentity,
      fusionProfile: lowering.fusionProfile.canonical,
      backendProfileIdentity: profile.compatibilityIdentity,
      backendProfile: profile.canonical,
      kernelCount: profile.simtNodeCount,
      fusionRegionCount: lowering.fusionProfile.regions.length,
      fusedNodeCount: lowering.fusionProfile.fusedNodeCount,
      materialCount: lowering.materials.length,
      materialBytes: lowering.materialBytes,
      cublasLtNodeCount: profile.cublasLtNodeCount,
      bindingCount: bindingRecords.length,
      workspaceBytes,
      backendIdentity: backendAdapter.identity,
      backendDescriptor: backendAdapter.descriptor,
      cleanup: 'resolved-plan-owns-backend-plans-compiled-module-functions-and-prepared-dag; result-owns-run-allocations',
    });
    const data = {
      session,
      plan,
      options: normalized,
      lowering,
      profile,
      backendAdapter,
      bindingRecords,
      workspaces,
      workspaceBytes,
      canonical,
      compatibilityIdentity: identity('resolved-tensor-plan-v1', canonical),
      state: 'open',
      pendingRuns: 0,
      runSequence: 0,
      terminal: null,
      sessionRegistration: null,
      results: new Set(),
    };
    const resolved = new ResolvedTensorPlan(RESOLVED_TOKEN, data);
    data.sessionRegistration = reservation.commit(() => resolved.close());
    return resolved;
  } catch (error) {
    reservation.cancel();
    if (backendAdapter !== null) {
      const cleanup = await backendAdapter.close();
      if (cleanup.graceful !== true) fail('TENSOR_RESOLVE_ROLLBACK_UNPROVED', 'cleanup-unproved', 'Resolved-plan session registration failed and backend rollback was not proved.', { primary: failureSummary(error), cleanup }, { cause: error });
    }
    throw error;
  }
}

export function resolveTensorPlan(session, planOrProgram, options) {
  return resolveTensorPlanWithAdapter(session, planOrProgram, options, createCudaJsTensorBackend);
}
