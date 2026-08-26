import { createHash } from 'node:crypto';
import { CUDA_JS_COMPATIBILITY, openCudaRuntime } from 'cuda-js';
import { deepFreeze, fail, failureSummary, TensorError } from './error.mjs';
import { createTensorSpec, TensorSpec } from './tensor-spec.mjs';

export const CUDA_JS_TENSOR_COMPATIBILITY = deepFreeze({
  schemaVersion: 1,
  contract: 'SPEC-0001-tensor-session-value-v1',
  package: { name: 'cuda-js-tensor', version: '0.1.0-alpha.4' },
  cudaJs: {
    name: 'cuda-js',
    version: '0.1.0-alpha.15',
    publicApiSchema: 1,
    protectedMainRevision: 'af29b95e0707b36b88ee4e234c25a9e7f7ed3a1d',
  },
});

const MIB = 1_048_576;
const DEFAULT_LIMITS = Object.freeze({ maxTensorBytes: 128 * MIB, maxSessionBytes: 256 * MIB, maxLiveTensors: 1_024 });
const DEFAULT_DEFAULTS = Object.freeze({ dtype: 'f32', access: 'read-write' });
const OPEN_FIELDS = new Set(['runtime', 'runtimeOwnership', 'device', 'limits', 'defaults']);
const LIMIT_FIELDS = new Set(['maxTensorBytes', 'maxSessionBytes', 'maxLiveTensors']);
const DEFAULT_FIELDS = new Set(['dtype', 'access']);
const SESSION_TOKEN = Symbol('TensorSession');
const TENSOR_TOKEN = Symbol('Tensor');
const SESSION_DATA = new WeakMap();
const TENSOR_DATA = new WeakMap();
let sessionSequence = 0;

function plainObject(value) {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  try { return Object.getPrototypeOf(value) === Object.prototype || Object.getPrototypeOf(value) === null; } catch { return false; }
}

function runtimeLike(value) {
  try {
    return value !== null && typeof value === 'object' && typeof value.describe === 'function'
      && typeof value.allocateDevice === 'function' && typeof value.close === 'function';
  } catch { return false; }
}

function positiveSafeInteger(value, field) {
  if (!Number.isSafeInteger(value) || value < 1) fail('TENSOR_SESSION_LIMIT_INVALID', 'validation', `${field} must be a positive safe integer.`, { field });
  return value;
}

function normalizeLimits(value) {
  if (value === undefined) return DEFAULT_LIMITS;
  if (!plainObject(value) || Object.keys(value).some((key) => !LIMIT_FIELDS.has(key))) fail('TENSOR_SESSION_LIMITS_INVALID', 'validation', 'Session limits contain unknown fields.');
  const limits = Object.freeze({
    maxTensorBytes: positiveSafeInteger(value.maxTensorBytes ?? DEFAULT_LIMITS.maxTensorBytes, 'maxTensorBytes'),
    maxSessionBytes: positiveSafeInteger(value.maxSessionBytes ?? DEFAULT_LIMITS.maxSessionBytes, 'maxSessionBytes'),
    maxLiveTensors: positiveSafeInteger(value.maxLiveTensors ?? DEFAULT_LIMITS.maxLiveTensors, 'maxLiveTensors'),
  });
  if (limits.maxTensorBytes > limits.maxSessionBytes) fail('TENSOR_SESSION_LIMITS_INVALID', 'validation', 'maxTensorBytes cannot exceed maxSessionBytes.');
  return limits;
}

function normalizeDefaults(value) {
  if (value === undefined) return DEFAULT_DEFAULTS;
  if (!plainObject(value) || Object.keys(value).some((key) => !DEFAULT_FIELDS.has(key))) fail('TENSOR_SESSION_DEFAULTS_INVALID', 'validation', 'Session defaults contain unknown fields.');
  const probe = createTensorSpec({ dtype: value.dtype ?? DEFAULT_DEFAULTS.dtype, capacityShape: [], access: value.access ?? DEFAULT_DEFAULTS.access });
  return Object.freeze({ dtype: probe.dtype, access: probe.access });
}

function normalizeOpen(value) {
  if (value === undefined) return Object.freeze({ runtime: null, runtimeSource: 'created', runtimeOwnership: 'owned', device: undefined, limits: DEFAULT_LIMITS, defaults: DEFAULT_DEFAULTS });
  if (runtimeLike(value)) return Object.freeze({ runtime: value, runtimeSource: 'injected', runtimeOwnership: 'borrowed', device: undefined, limits: DEFAULT_LIMITS, defaults: DEFAULT_DEFAULTS });
  if (!plainObject(value) || Object.keys(value).some((key) => !OPEN_FIELDS.has(key))) fail('TENSOR_SESSION_OPTIONS_INVALID', 'validation', 'TensorSession options contain unknown fields.');
  const hasRuntime = Object.hasOwn(value, 'runtime');
  const hasDevice = Object.hasOwn(value, 'device');
  if (hasRuntime && hasDevice) fail('TENSOR_SESSION_OPTIONS_INVALID', 'validation', 'An injected runtime cannot be combined with a device selector.');
  if (hasRuntime && !runtimeLike(value.runtime)) fail('TENSOR_SESSION_RUNTIME_INVALID', 'validation', 'runtime must be an open public CUDA-JS runtime capability.');
  const runtimeOwnership = value.runtimeOwnership ?? (hasRuntime ? 'borrowed' : 'owned');
  if (!['owned', 'borrowed'].includes(runtimeOwnership) || (!hasRuntime && runtimeOwnership !== 'owned')) {
    fail('TENSOR_SESSION_OWNERSHIP_INVALID', 'validation', 'Created runtimes are owned; injected runtimes require borrowed or explicitly transferred owned authority.');
  }
  return Object.freeze({
    runtime: hasRuntime ? value.runtime : null,
    runtimeSource: hasRuntime ? 'injected' : 'created',
    runtimeOwnership,
    device: hasDevice ? value.device : undefined,
    limits: normalizeLimits(value.limits),
    defaults: normalizeDefaults(value.defaults),
  });
}

function assertStaticCompatibility() {
  const actual = CUDA_JS_COMPATIBILITY?.package;
  const publicApiSchema = CUDA_JS_COMPATIBILITY?.publicApi?.schemaVersion;
  if (actual?.name !== CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.name || actual?.version !== CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version || publicApiSchema !== CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.publicApiSchema) {
    fail('TENSOR_CUDA_JS_INCOMPATIBLE', 'unsupported', 'Installed CUDA-JS does not match the exact accepted CUDA-JS-Tensor compatibility identity.', {
      expectedVersion: CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version,
      actualVersion: actual?.version ?? null,
      expectedPublicApiSchema: CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.publicApiSchema,
      actualPublicApiSchema: publicApiSchema ?? null,
    });
  }
}

function assertRuntimeDescription(description) {
  if (!plainObject(description) || description.package?.name !== 'cuda-js' || description.package?.version !== CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version || description.package?.publicApiSchema !== 1 || description.state !== 'open') {
    fail('TENSOR_CUDA_JS_RUNTIME_INCOMPATIBLE', 'unsupported', 'CUDA-JS runtime description does not match the accepted open-runtime contract.');
  }
}

function hashIdentity(record) {
  return createHash('sha256').update(JSON.stringify(record)).digest('hex');
}

function sessionData(session, operation) {
  const data = SESSION_DATA.get(session);
  if (!data) fail('TENSOR_SESSION_INVALID', 'validation', `${operation} requires a TensorSession capability.`);
  return data;
}

function assertSessionOpen(data, operation) {
  if (data.state !== 'open') fail('TENSOR_SESSION_CLOSED', 'closed-session', `${operation} requires an open TensorSession.`, { state: data.state });
}

function tensorData(tensor, operation) {
  const data = TENSOR_DATA.get(tensor);
  if (!data) fail('TENSOR_INVALID', 'validation', `${operation} requires a Tensor capability.`);
  return data;
}

function assertTensorOpen(data, operation) {
  if (data.state !== 'open') fail('TENSOR_CLOSED', 'closed-tensor', `${operation} requires an open Tensor capability.`, { state: data.state });
}

function allocationSpec(defaults, first, second) {
  if (first instanceof TensorSpec) {
    if (second !== undefined) fail('TENSOR_ALLOCATION_OPTIONS_INVALID', 'validation', 'A TensorSpec cannot be combined with another allocation argument.');
    return first;
  }
  return createTensorSpec(first, second, defaults);
}

function physicalAllocationBytes(spec) {
  return Math.max(spec.requiredByteEnd, spec.alignment);
}

function reserveCreation(data, allocationBytes) {
  assertSessionOpen(data, 'TensorSession.allocate');
  if (data.liveTensorCount + 1 > data.limits.maxLiveTensors) fail('TENSOR_SESSION_LIVE_LIMIT', 'pressure', 'Tensor creation exceeds maxLiveTensors.', { maximum: data.limits.maxLiveTensors });
  if (allocationBytes > data.limits.maxTensorBytes) fail('TENSOR_ALLOCATION_LIMIT', 'pressure', 'Tensor allocation exceeds maxTensorBytes.', { byteLength: allocationBytes, maximum: data.limits.maxTensorBytes });
  if (allocationBytes > data.limits.maxSessionBytes - data.reservedBytes) fail('TENSOR_SESSION_BYTE_LIMIT', 'pressure', 'Tensor allocation exceeds remaining session byte capacity.', { byteLength: allocationBytes, reservedBytes: data.reservedBytes, maximum: data.limits.maxSessionBytes });
  data.pendingCreates += 1;
  data.liveTensorCount += 1;
  data.reservedBytes += allocationBytes;
}

function releaseCreationReservation(data, allocationBytes, { retainBytes = false, retainLive = false } = {}) {
  data.pendingCreates -= 1;
  if (!retainLive) data.liveTensorCount -= 1;
  if (!retainBytes) data.reservedBytes -= allocationBytes;
}

function accessAllows(parent, child) {
  if (parent === 'read-write') return true;
  return parent === child;
}

function isRetriableClose(error) {
  return error?.code === 'RESOURCE_BUSY' || error?.code === 'RESOURCE_HAS_CHILDREN' || error?.code === 'CUDA_JS_RESOURCE_BUSY';
}

function terminalTensorReport(data, disposition) {
  return deepFreeze({
    schemaVersion: 1,
    kind: 'tensor',
    state: data.state,
    disposition,
    ownsAllocation: data.ownsAllocation,
    allocationBytes: data.allocationByteLength,
    storageIdentity: data.storageIdentity,
  });
}

async function closeTensor(tensor) {
  const data = tensorData(tensor, 'Tensor.close');
  if (data.state === 'closed' || data.state === 'orphaned') return data.terminalReport;
  if (data.pendingChildren > 0) fail('TENSOR_BUSY', 'backpressure', 'Tensor has a child view creation in progress.');
  if ([...data.children].some((child) => !['closed', 'orphaned'].includes(tensorData(child, 'Tensor.close').state))) {
    fail('TENSOR_HAS_CHILDREN', 'backpressure', 'Tensor cannot close while a child tensor view remains live.', { childCount: data.children.size });
  }

  data.state = 'closing';
  try {
    if (!data.viewClosed) {
      const viewReport = await data.deviceView.close();
      if (viewReport?.state !== 'closed') fail('TENSOR_VIEW_CLEANUP_UNPROVED', 'cleanup-unproved', 'CUDA-JS did not prove tensor view closure.');
      data.viewClosed = true;
    }
    if (data.ownsAllocation && !data.memoryClosed) {
      const memoryReport = await data.memory.close();
      if (memoryReport?.state !== 'closed') fail('TENSOR_MEMORY_CLEANUP_UNPROVED', 'cleanup-unproved', 'CUDA-JS did not prove tensor allocation closure.');
      data.memoryClosed = true;
    }
  } catch (error) {
    if (isRetriableClose(error)) {
      data.state = data.viewClosed ? 'closing' : 'open';
      throw error;
    }
    data.state = 'orphaned';
    data.terminalReport = terminalTensorReport(data, 'cleanup-unproved');
    data.sessionData.liveTensorCount -= 1;
    if (!data.ownsAllocation) data.parentData.children.delete(tensor);
    data.sessionData.unproved.push(Object.freeze({ kind: data.ownsAllocation ? 'tensor-allocation' : 'tensor-view', failure: failureSummary(error), allocationBytes: data.allocationByteLength }));
    return data.terminalReport;
  }

  data.state = 'closed';
  data.sessionData.liveTensorCount -= 1;
  if (data.ownsAllocation) data.sessionData.reservedBytes -= data.allocationByteLength;
  else data.parentData.children.delete(tensor);
  data.terminalReport = terminalTensorReport(data, 'closed');
  return data.terminalReport;
}

function buildTensor(session, sessionRecord, { spec, memory, deviceView, ownsAllocation, allocationByteLength, storageIdentity, parent = null }) {
  const tensor = new Tensor(TENSOR_TOKEN);
  const parentData = parent ? tensorData(parent, 'Tensor.view') : null;
  const data = {
    session,
    sessionData: sessionRecord,
    spec,
    memory,
    deviceView,
    ownsAllocation,
    allocationByteLength,
    storageIdentity,
    aliasGroup: storageIdentity,
    parent,
    parentData,
    children: new Set(),
    pendingChildren: 0,
    depth: parentData ? parentData.depth + 1 : 0,
    sequence: ++sessionRecord.tensorSequence,
    state: 'open',
    viewClosed: false,
    memoryClosed: !ownsAllocation,
    terminalReport: null,
  };
  TENSOR_DATA.set(tensor, data);
  sessionRecord.tensors.add(tensor);
  if (parentData) parentData.children.add(tensor);
  return tensor;
}

function viewSpec(parentData, value) {
  if (value instanceof TensorSpec) return value;
  if (value === undefined) return parentData.spec;
  if (Array.isArray(value)) return createTensorSpec({
    dtype: parentData.spec.dtype,
    capacityShape: value,
    byteOffset: parentData.spec.byteOffset,
    access: parentData.spec.access,
  });
  if (!plainObject(value)) fail('TENSOR_VIEW_OPTIONS_INVALID', 'validation', 'Tensor.view requires a TensorSpec, shape array, exact options record, or no argument.');
  const capacityShape = value.capacityShape ?? parentData.spec.capacityShape;
  const preserveStrides = value.capacityShape === undefined || (Array.isArray(capacityShape) && capacityShape.length === parentData.spec.rank && capacityShape.every((entry, index) => entry === parentData.spec.capacityShape[index]));
  return createTensorSpec({
    ...value,
    dtype: value.dtype ?? parentData.spec.dtype,
    capacityShape,
    activeAxis0: Object.hasOwn(value, 'activeAxis0') ? value.activeAxis0 : parentData.spec.activeAxis0,
    strides: Object.hasOwn(value, 'strides') ? value.strides : (preserveStrides ? parentData.spec.strides : undefined),
    byteOffset: value.byteOffset ?? parentData.spec.byteOffset,
    access: value.access ?? parentData.spec.access,
    aliasGroup: value.aliasGroup ?? parentData.spec.aliasGroup,
  });
}

export class TensorSession {
  constructor(token, data) {
    if (token !== SESSION_TOKEN) fail('TENSOR_SESSION_CONSTRUCTION_INVALID', 'validation', 'Use TensorSession.open().');
    SESSION_DATA.set(this, data);
    Object.freeze(this);
  }

  static async open(options) {
    assertStaticCompatibility();
    const normalized = normalizeOpen(options);
    const runtime = normalized.runtime ?? await openCudaRuntime(normalized.device === undefined ? { compiler: true } : { compiler: true, device: normalized.device });
    let description;
    try {
      description = await runtime.describe();
      assertRuntimeDescription(description);
    } catch (error) {
      if (normalized.runtimeOwnership === 'owned') {
        try {
          const rollback = await runtime.close();
          if (rollback?.graceful !== true) fail('TENSOR_SESSION_OPEN_ROLLBACK_UNPROVED', 'cleanup-unproved', 'Owned runtime did not report graceful rollback.');
        } catch (cleanupError) {
          throw new TensorError('TENSOR_SESSION_OPEN_ROLLBACK_UNPROVED', 'cleanup-unproved', 'Runtime validation failed and owned-runtime rollback was unproved.', { primary: failureSummary(error), cleanup: failureSummary(cleanupError) }, { cause: error });
        }
      }
      throw error;
    }
    const sequence = ++sessionSequence;
    const runtimeIdentity = deepFreeze({
      package: { ...description.package },
      profile: description.profile ?? null,
      device: description.device ?? null,
      compilerEnabled: runtime.compilerEnabled === true,
    });
    const compatibilityIdentity = `tensor-session-v1:${hashIdentity({ contract: CUDA_JS_TENSOR_COMPATIBILITY.contract, runtimeIdentity, limits: normalized.limits, defaults: normalized.defaults, sequence })}`;
    const resolvedOpenOptions = deepFreeze({
      runtimeSource: normalized.runtimeSource,
      runtimeOwnership: normalized.runtimeOwnership,
      deviceSelection: normalized.runtimeSource === 'injected' ? 'injected' : (normalized.device === undefined ? 'default' : 'explicit'),
      compiler: normalized.runtimeSource === 'created' ? 'enabled' : (runtime.compilerEnabled === true ? 'injected-enabled' : 'injected-disabled'),
      limits: { ...normalized.limits },
      defaults: { ...normalized.defaults },
    });
    return new TensorSession(SESSION_TOKEN, {
      runtime,
      runtimeIdentity,
      compatibilityIdentity,
      resolvedOpenOptions,
      ownershipMode: normalized.runtimeOwnership,
      limits: normalized.limits,
      defaults: normalized.defaults,
      state: 'open',
      tensors: new Set(),
      tensorSequence: 0,
      storageSequence: 0,
      pendingCreates: 0,
      pendingExecutionCreates: 0,
      executionChildren: new Set(),
      executionSequence: 0,
      liveTensorCount: 0,
      reservedBytes: 0,
      unproved: [],
      terminalReport: null,
    });
  }

  get kind() { return 'tensor-session'; }
  get state() { return sessionData(this, 'TensorSession.state').state; }
  get ownershipMode() { return sessionData(this, 'TensorSession.ownershipMode').ownershipMode; }
  get limits() { return sessionData(this, 'TensorSession.limits').limits; }
  get defaults() { return sessionData(this, 'TensorSession.defaults').defaults; }
  get compatibilityIdentity() { return sessionData(this, 'TensorSession.compatibilityIdentity').compatibilityIdentity; }
  get resolvedOpenOptions() { return sessionData(this, 'TensorSession.resolvedOpenOptions').resolvedOpenOptions; }

  async allocate(first, second) {
    const data = sessionData(this, 'TensorSession.allocate');
    const spec = allocationSpec(data.defaults, first, second);
    const allocationByteLength = physicalAllocationBytes(spec);
    reserveCreation(data, allocationByteLength);
    let memory = null;
    let deviceView = null;
    try {
      memory = await data.runtime.allocateDevice({ byteLength: allocationByteLength });
      deviceView = await memory.view({ dtype: spec.dtype, byteOffset: spec.byteOffset, elementCount: spec.storageElementSpan, access: spec.access });
      const storageIdentity = `tensor-storage-v1:${hashIdentity({ session: data.compatibilityIdentity, sequence: ++data.storageSequence })}`;
      const tensor = buildTensor(this, data, { spec, memory, deviceView, ownsAllocation: true, allocationByteLength, storageIdentity });
      data.pendingCreates -= 1;
      return tensor;
    } catch (error) {
      const cleanupFailures = [];
      if (deviceView) try { await deviceView.close(); } catch (cleanupError) { cleanupFailures.push(cleanupError); }
      if (memory) try { await memory.close(); } catch (cleanupError) { cleanupFailures.push(cleanupError); }
      if (cleanupFailures.length > 0) {
        releaseCreationReservation(data, allocationByteLength, { retainBytes: true });
        data.unproved.push(Object.freeze({ kind: 'tensor-allocation-rollback', allocationBytes: allocationByteLength, failure: failureSummary(cleanupFailures[0]) }));
        throw new TensorError('TENSOR_ALLOCATION_ROLLBACK_UNPROVED', 'cleanup-unproved', 'Tensor allocation failed and CUDA-JS resource rollback was unproved.', { primary: failureSummary(error), cleanup: cleanupFailures.map(failureSummary), allocationBytes: allocationByteLength }, { cause: error });
      }
      releaseCreationReservation(data, allocationByteLength);
      throw error;
    }
  }

  async status() {
    const data = sessionData(this, 'TensorSession.status');
    return deepFreeze({
      schemaVersion: 1,
      kind: 'tensor-session',
      state: data.state,
      ownershipMode: data.ownershipMode,
      compatibilityIdentity: data.compatibilityIdentity,
      limits: { ...data.limits },
      defaults: { ...data.defaults },
      accounting: { liveTensors: data.liveTensorCount, reservedBytes: data.reservedBytes, pendingCreates: data.pendingCreates, resolvedPlans: data.executionChildren.size, pendingResolutions: data.pendingExecutionCreates, unprovedResources: data.unproved.length },
    });
  }

  async close() {
    const data = sessionData(this, 'TensorSession.close');
    if (data.state === 'closed' || data.state === 'orphaned') return data.terminalReport;
    if (data.pendingCreates > 0 || data.pendingExecutionCreates > 0) fail('TENSOR_SESSION_BUSY', 'backpressure', 'TensorSession cannot close while tensor or resolved-plan creation is in progress.', { pendingCreates: data.pendingCreates, pendingResolutions: data.pendingExecutionCreates });
    data.state = 'closing';
    const failures = [];
    for (const child of [...data.executionChildren].reverse()) {
      try {
        const report = await child.close();
        if (report?.graceful !== true) failures.push(Object.freeze({ kind: 'resolved-plan', failure: { code: 'TENSOR_RESOLVED_PLAN_CLEANUP_UNPROVED', category: 'cleanup-unproved', name: 'TensorError' } }));
      } catch (error) {
        if (error?.category === 'backpressure') {
          data.state = 'open';
          fail('TENSOR_SESSION_BUSY', 'backpressure', 'TensorSession cannot close while a resolved plan remains busy.', { causeCode: error.code ?? null }, { cause: error });
        }
        failures.push(Object.freeze({ kind: 'resolved-plan', failure: failureSummary(error) }));
        data.unproved.push(Object.freeze({ kind: 'resolved-plan', failure: failureSummary(error), allocationBytes: 0 }));
      }
    }
    const tensors = [...data.tensors].sort((left, right) => {
      const leftData = tensorData(left, 'TensorSession.close');
      const rightData = tensorData(right, 'TensorSession.close');
      return rightData.depth - leftData.depth || rightData.sequence - leftData.sequence;
    });
    for (const tensor of tensors) {
      const record = tensorData(tensor, 'TensorSession.close');
      if (record.state === 'closed' || record.state === 'orphaned') continue;
      try {
        const report = await closeTensor(tensor);
        if (report.disposition !== 'closed') failures.push(Object.freeze({ kind: 'tensor', failure: { code: 'TENSOR_CLEANUP_UNPROVED', category: 'cleanup-unproved', name: 'TensorError' } }));
      } catch (error) {
        if (error?.category === 'backpressure') {
          data.state = 'open';
          fail('TENSOR_SESSION_BUSY', 'backpressure', 'TensorSession cannot close while a tensor resource remains busy.', { causeCode: error.code ?? null }, { cause: error });
        }
        failures.push(Object.freeze({ kind: 'tensor', failure: failureSummary(error) }));
      }
    }

    let runtimeReport = null;
    if (data.ownershipMode === 'owned') {
      try { runtimeReport = await data.runtime.close(); } catch (error) { failures.push(Object.freeze({ kind: 'runtime', failure: failureSummary(error) })); }
      data.state = runtimeReport?.graceful === true && failures.length === 0 && data.unproved.length === 0 ? 'closed' : 'orphaned';
    } else if (failures.length === 0 && data.unproved.length === 0) {
      data.state = 'closed';
    } else {
      data.state = data.unproved.length > 0 ? 'orphaned' : 'open';
    }

    const graceful = failures.length === 0 && data.unproved.length === 0 && (data.ownershipMode === 'borrowed' || runtimeReport?.graceful === true);
    const report = deepFreeze({
      schemaVersion: 1,
      kind: 'tensor-session',
      state: data.state,
      graceful,
      runtimeOwnership: data.ownershipMode,
      runtimeClosed: data.ownershipMode === 'owned' && runtimeReport !== null,
      runtimeGraceful: data.ownershipMode === 'owned' ? runtimeReport?.graceful === true : null,
      failures,
      unprovedResources: data.unproved.map((entry) => ({ ...entry })),
      accounting: { liveTensors: data.liveTensorCount, reservedBytes: data.reservedBytes, resolvedPlans: data.executionChildren.size, pendingResolutions: data.pendingExecutionCreates },
    });
    if (data.state !== 'open') data.terminalReport = report;
    return report;
  }
}

export class Tensor {
  constructor(token) {
    if (token !== TENSOR_TOKEN) fail('TENSOR_CONSTRUCTION_INVALID', 'validation', 'Tensor capabilities are created by TensorSession.');
    Object.freeze(this);
  }

  get kind() { return 'tensor'; }
  get state() { return tensorData(this, 'Tensor.state').state; }
  get spec() { return tensorData(this, 'Tensor.spec').spec; }
  get dtype() { return tensorData(this, 'Tensor.dtype').spec.dtype; }
  get capacityShape() { return tensorData(this, 'Tensor.capacityShape').spec.capacityShape; }
  get logicalShape() { return tensorData(this, 'Tensor.logicalShape').spec.logicalShape; }
  get strides() { return tensorData(this, 'Tensor.strides').spec.strides; }
  get byteLength() { return tensorData(this, 'Tensor.byteLength').spec.byteLength; }
  get access() { return tensorData(this, 'Tensor.access').spec.access; }
  get aliasGroup() { return tensorData(this, 'Tensor.aliasGroup').aliasGroup; }
  get sessionCompatibilityIdentity() { return tensorData(this, 'Tensor.sessionCompatibilityIdentity').sessionData.compatibilityIdentity; }

  async write(bytes) {
    const data = tensorData(this, 'Tensor.write');
    assertTensorOpen(data, 'Tensor.write');
    assertSessionOpen(data.sessionData, 'Tensor.write');
    if (data.spec.access === 'read') fail('TENSOR_WRITE_ACCESS_DENIED', 'validation', 'Tensor.write requires write authority.');
    if (data.spec.layout !== 'row-major-contiguous' || data.spec.hasBroadcastAliasing) fail('TENSOR_TRANSFER_LAYOUT_UNSUPPORTED', 'unsupported', 'Tensor.write requires a contiguous non-broadcast tensor; materialize strided values first.');
    const byteLength = data.spec.logicalElementCount * data.spec.dtypeWidth;
    if (!(bytes instanceof Uint8Array) || bytes.byteLength !== byteLength) fail('TENSOR_WRITE_BYTES_INVALID', 'validation', 'Tensor.write requires exactly one logical tensor of bytes.', { expected: byteLength, actual: bytes instanceof Uint8Array ? bytes.byteLength : null });
    if (byteLength > 0) await data.memory.write(Uint8Array.from(bytes), { deviceOffset: data.spec.byteOffset });
    return deepFreeze({ schemaVersion: 1, kind: 'tensor-write', byteLength });
  }

  async read() {
    const data = tensorData(this, 'Tensor.read');
    assertTensorOpen(data, 'Tensor.read');
    assertSessionOpen(data.sessionData, 'Tensor.read');
    if (data.spec.access === 'write') fail('TENSOR_READ_ACCESS_DENIED', 'validation', 'Tensor.read requires read authority.');
    if (data.spec.layout !== 'row-major-contiguous' || data.spec.hasBroadcastAliasing) fail('TENSOR_TRANSFER_LAYOUT_UNSUPPORTED', 'unsupported', 'Tensor.read requires a contiguous non-broadcast tensor; materialize strided values first.');
    const byteLength = data.spec.logicalElementCount * data.spec.dtypeWidth;
    const bytes = byteLength === 0
      ? new Uint8Array()
      : Uint8Array.from((await data.memory.read({ deviceOffset: data.spec.byteOffset, byteLength })).bytes);
    return Object.freeze({ schemaVersion: 1, kind: 'tensor-read', byteLength, bytes });
  }

  async view(options) {
    const parentData = tensorData(this, 'Tensor.view');
    assertTensorOpen(parentData, 'Tensor.view');
    assertSessionOpen(parentData.sessionData, 'Tensor.view');
    const spec = viewSpec(parentData, options);
    if (spec.dtype !== parentData.spec.dtype) fail('TENSOR_VIEW_DTYPE_MISMATCH', 'validation', 'A tensor view cannot reinterpret its parent dtype.');
    if (spec.aliasGroup !== parentData.spec.aliasGroup) fail('TENSOR_VIEW_ALIAS_DECLARATION_MISMATCH', 'validation', 'A tensor view cannot replace its parent declared alias group.');
    if (!accessAllows(parentData.spec.access, spec.access)) fail('TENSOR_VIEW_ACCESS_DENIED', 'validation', 'A tensor view cannot widen its parent access authority.', { parentAccess: parentData.spec.access, requestedAccess: spec.access });
    const parentStart = parentData.spec.byteOffset;
    const parentEnd = parentData.spec.requiredByteEnd;
    if (spec.byteOffset < parentStart || spec.requiredByteEnd > parentEnd) {
      fail('TENSOR_VIEW_RANGE_OUT_OF_BOUNDS', 'validation', 'Tensor view byte envelope exceeds its parent tensor authority.', { parentStart, parentEnd, viewStart: spec.byteOffset, viewEnd: spec.requiredByteEnd });
    }
    if (parentData.spec.layout !== 'row-major-contiguous' && spec.compatibilityIdentity !== parentData.spec.compatibilityIdentity) {
      fail('TENSOR_VIEW_STRIDED_PARENT_UNSUPPORTED', 'unsupported', 'V1 cannot prove a changed child reachable-element set is a subset of a strided parent; materialize or view the contiguous owner instead.');
    }
    const sessionRecord = parentData.sessionData;
    if (sessionRecord.liveTensorCount + 1 > sessionRecord.limits.maxLiveTensors) fail('TENSOR_SESSION_LIVE_LIMIT', 'pressure', 'Tensor view creation exceeds maxLiveTensors.', { maximum: sessionRecord.limits.maxLiveTensors });
    sessionRecord.pendingCreates += 1;
    sessionRecord.liveTensorCount += 1;
    parentData.pendingChildren += 1;
    try {
      const deviceView = await parentData.memory.view({ dtype: spec.dtype, byteOffset: spec.byteOffset, elementCount: spec.storageElementSpan, access: spec.access });
      const child = buildTensor(parentData.session, sessionRecord, {
        spec,
        memory: parentData.memory,
        deviceView,
        ownsAllocation: false,
        allocationByteLength: 0,
        storageIdentity: parentData.storageIdentity,
        parent: this,
      });
      sessionRecord.pendingCreates -= 1;
      parentData.pendingChildren -= 1;
      return child;
    } catch (error) {
      sessionRecord.pendingCreates -= 1;
      sessionRecord.liveTensorCount -= 1;
      parentData.pendingChildren -= 1;
      throw error;
    }
  }

  async status() {
    const data = tensorData(this, 'Tensor.status');
    let deviceViewState = data.viewClosed ? 'closed' : null;
    if (!data.viewClosed && data.state !== 'orphaned') {
      try { deviceViewState = (await data.deviceView.status()).state; } catch { deviceViewState = 'unavailable'; }
    }
    return deepFreeze({
      schemaVersion: 1,
      kind: 'tensor',
      state: data.state,
      spec: data.spec.canonical,
      aliasGroup: data.aliasGroup,
      ownsAllocation: data.ownsAllocation,
      allocationBytes: data.allocationByteLength,
      childCount: [...data.children].filter((child) => !['closed', 'orphaned'].includes(tensorData(child, 'Tensor.status').state)).length,
      deviceViewState,
      sessionCompatibilityIdentity: data.sessionData.compatibilityIdentity,
    });
  }

  close() {
    return closeTensor(this);
  }
}

export function inspectTensorForSession(session, tensor) {
  const owner = sessionData(session, 'inspectTensorForSession');
  assertSessionOpen(owner, 'inspectTensorForSession');
  const data = tensorData(tensor, 'inspectTensorForSession');
  if (data.session !== session) fail('TENSOR_CROSS_SESSION', 'validation', 'Tensor belongs to a different session/runtime epoch.');
  assertTensorOpen(data, 'inspectTensorForSession');
  return Object.freeze({
    spec: data.spec,
    deviceView: data.deviceView,
    storageIdentity: data.storageIdentity,
    aliasGroup: data.aliasGroup,
    sessionCompatibilityIdentity: owner.compatibilityIdentity,
  });
}

export function inspectTensorSessionForExecution(session) {
  const data = sessionData(session, 'inspectTensorSessionForExecution');
  assertSessionOpen(data, 'inspectTensorSessionForExecution');
  return Object.freeze({
    runtime: data.runtime,
    compatibilityIdentity: data.compatibilityIdentity,
    runtimeIdentity: data.runtimeIdentity,
    limits: data.limits,
    accounting: Object.freeze({ liveTensors: data.liveTensorCount, reservedBytes: data.reservedBytes, pendingCreates: data.pendingCreates, resolvedPlans: data.executionChildren.size, pendingResolutions: data.pendingExecutionCreates }),
  });
}

export function reserveTensorSessionExecution(session) {
  const data = sessionData(session, 'reserveTensorSessionExecution');
  assertSessionOpen(data, 'reserveTensorSessionExecution');
  data.pendingExecutionCreates += 1;
  let state = 'pending';
  let child = null;
  const cancel = () => {
    if (state !== 'pending') return;
    state = 'cancelled';
    data.pendingExecutionCreates -= 1;
  };
  return Object.freeze({
    commit(close) {
      if (state !== 'pending' || typeof close !== 'function') fail('TENSOR_EXECUTION_REGISTRATION_INVALID', 'internal', 'Resolved-plan session registration is invalid.');
      assertSessionOpen(data, 'reserveTensorSessionExecution.commit');
      child = Object.freeze({ sequence: ++data.executionSequence, close });
      data.pendingExecutionCreates -= 1;
      data.executionChildren.add(child);
      state = 'registered';
      return Object.freeze({
        release(report) {
          if (state !== 'registered') return;
          state = 'released';
          data.executionChildren.delete(child);
          if (report?.graceful !== true) data.unproved.push(Object.freeze({ kind: 'resolved-plan', failure: { code: 'TENSOR_RESOLVED_PLAN_CLEANUP_UNPROVED', category: 'cleanup-unproved', name: 'TensorError' }, allocationBytes: 0 }));
        },
      });
    },
    cancel,
  });
}
