import { createHash } from 'node:crypto';
import { CUDA_JS_COMPATIBILITY, openCudaRuntime } from 'cuda-js';
import { deepFreeze, fail, failureSummary, TensorError } from './error.mjs';
import { createTensorSpec, TensorSpec } from './tensor-spec.mjs';

export const CUDA_JS_TENSOR_COMPATIBILITY = deepFreeze({
  schemaVersion: 1,
  contract: 'SPEC-0001-tensor-session-value-v1',
  package: { name: 'cuda-js-tensor', version: '0.1.0-alpha.6' },
  cudaJs: {
    name: 'cuda-js',
    version: '0.1.0-alpha.17',
    publicApiSchema: 1,
    protectedMainRevision: 'bc2700f2e5c654567c2e17bf8d67b882351b8681',
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
  const candidate = plainObject(value) ? value : null;
  if (!candidate || Object.keys(candidate).some((key) => !DEFAULT_FIELDS.has(key))) fail('TENSOR_SESSION_DEFAULTS_INVALID', 'validation', 'Session defaults contain unknown fields.');
  const spec = createTensorSpec({ dtype: candidate.dtype ?? DEFAULT_DEFAULTS.dtype, capacityShape: [], access: candidate.access ?? DEFAULT_DEFAULTS.access });
  return Object.freeze({ dtype: spec.dtype, access: spec.access });
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

function compatibility(runtimeDescription) {
  if (runtimeDescription?.package?.name !== CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.name
      || runtimeDescription?.package?.version !== CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.version
      || runtimeDescription?.package?.publicApiSchema !== CUDA_JS_TENSOR_COMPATIBILITY.cudaJs.publicApiSchema) {
    fail('TENSOR_SESSION_CUDA_JS_INCOMPATIBLE', 'unsupported', 'The CUDA-JS runtime does not match the exact Tensor compatibility profile.', {
      required: CUDA_JS_TENSOR_COMPATIBILITY.cudaJs,
      actual: runtimeDescription?.package ?? null,
    });
  }
}

function valueState(value, operation) {
  if (value.state === 'open') return value;
  fail('TENSOR_RESOURCE_NOT_OPEN', value.state === 'orphaned' ? 'orphaned' : 'closed', `${operation} requires an open Tensor resource.`, { state: value.state });
}

function tensorData(tensor) {
  const data = TENSOR_DATA.get(tensor);
  if (!data) fail('TENSOR_RESOURCE_INVALID', 'validation', 'The value is not a Tensor capability.');
  return data;
}

function sessionData(session) {
  const data = SESSION_DATA.get(session);
  if (!data) fail('TENSOR_SESSION_INVALID', 'validation', 'The value is not a TensorSession capability.');
  return data;
}

function tensorDescriptor(data) {
  return deepFreeze({
    schemaVersion: 1,
    kind: 'tensor',
    state: data.state,
    identity: data.identity,
    spec: data.spec.toJSON(),
    storage: data.storageIdentity,
    owner: data.session.identity,
  });
}

function sessionDescriptor(data) {
  return deepFreeze({
    schemaVersion: 1,
    kind: 'tensor-session',
    state: data.state,
    identity: data.identity,
    runtimeOwnership: data.runtimeOwnership,
    runtime: data.runtimeDescription,
    limits: data.limits,
    defaults: data.defaults,
    accounting: {
      liveTensors: data.tensors.size,
      reservedBytes: data.reservedBytes,
      pendingCreates: data.pendingCreates,
      resolvedPlans: data.resolvedPlans.size,
      pendingResolutions: data.pendingResolutions,
      unprovedResources: data.unprovedResources,
    },
  });
}

export class Tensor {
  constructor(token, data) {
    if (token !== TENSOR_TOKEN) fail('TENSOR_CONSTRUCTION_FORBIDDEN', 'validation', 'Tensor capabilities are created by TensorSession.');
    TENSOR_DATA.set(this, data);
    Object.freeze(this);
  }

  get kind() { return 'tensor'; }
  get state() { return tensorData(this).state; }
  get compatibilityIdentity() { return tensorData(this).identity; }
  get spec() { return tensorData(this).spec; }
  get dtype() { return this.spec.dtype; }
  get capacityShape() { return this.spec.capacityShape; }
  get shape() { return this.spec.logicalShape; }
  get strides() { return this.spec.strides; }
  get byteOffset() { return this.spec.byteOffset; }
  get byteLength() { return this.spec.byteLength; }
  get access() { return this.spec.access; }
  get activeAxis0() { return this.spec.activeAxis0; }
  describe() { return tensorDescriptor(tensorData(this)); }
  status() { return this.describe(); }

  async write(bytes, options = {}) {
    const data = valueState(tensorData(this), 'Tensor.write');
    if (data.spec.access === 'read') fail('TENSOR_ACCESS_DENIED', 'validation', 'Tensor.write requires write access.');
    if (!(bytes instanceof Uint8Array)) fail('TENSOR_WRITE_BYTES_INVALID', 'validation', 'Tensor.write requires Uint8Array bytes.');
    const byteOffset = options.byteOffset ?? 0;
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || byteOffset + bytes.byteLength > data.spec.byteLength) fail('TENSOR_WRITE_RANGE_INVALID', 'validation', 'Tensor.write range exceeds the logical tensor byte range.');
    return data.memory.write(bytes, { byteOffset: data.spec.byteOffset + byteOffset });
  }

  async read(options = {}) {
    const data = valueState(tensorData(this), 'Tensor.read');
    if (data.spec.access === 'write') fail('TENSOR_ACCESS_DENIED', 'validation', 'Tensor.read requires read access.');
    const byteOffset = options.byteOffset ?? 0;
    const byteLength = options.byteLength ?? data.spec.byteLength - byteOffset;
    if (!Number.isSafeInteger(byteOffset) || byteOffset < 0 || !Number.isSafeInteger(byteLength) || byteLength < 0 || byteOffset + byteLength > data.spec.byteLength) fail('TENSOR_READ_RANGE_INVALID', 'validation', 'Tensor.read range exceeds the logical tensor byte range.');
    return data.memory.read({ byteOffset: data.spec.byteOffset + byteOffset, byteLength });
  }

  async view(options = {}) {
    const data = valueState(tensorData(this), 'Tensor.view');
    const spec = createTensorSpec({
      dtype: data.spec.dtype,
      capacityShape: data.spec.capacityShape,
      ...(options.strides ? { strides: options.strides } : { strides: data.spec.strides }),
      byteOffset: options.byteOffset ?? data.spec.byteOffset,
      access: options.access ?? data.spec.access,
      activeAxis0: options.activeAxis0 ?? data.spec.activeAxis0,
      aliasGroup: data.spec.aliasGroup,
    });
    const view = await data.session.createTensor(spec, data.memory, { storageIdentity: data.storageIdentity, ownership: 'borrowed' });
    return view;
  }

  async close() {
    const data = tensorData(this);
    if (data.state === 'closed' || data.state === 'orphaned') return data.terminal;
    data.state = 'closing';
    const failures = [];
    if (data.deviceView?.state === 'open') {
      try { await data.deviceView.close(); } catch (error) { failures.push(failureSummary(error)); }
    }
    if (failures.length > 0 && failures.every((entry) => entry.category === 'backpressure')) {
      data.state = 'open';
      fail('TENSOR_RESOURCE_BUSY', 'backpressure', 'Tensor retains live lower resources and may be closed again after they become terminal.', { failures });
    }
    if (failures.length === 0 && data.ownership === 'owned' && data.memory?.state === 'open') {
      try { await data.memory.close(); } catch (error) { failures.push(failureSummary(error)); }
    }
    data.state = failures.length === 0 ? 'closed' : 'orphaned';
    data.terminal = deepFreeze({ schemaVersion: 1, kind: 'tensor', state: data.state, graceful: failures.length === 0, failures });
    data.session.tensors.delete(this);
    if (data.ownership === 'owned') data.session.reservedBytes -= data.reservedBytes;
    if (failures.length > 0) data.session.unprovedResources += 1;
    return data.terminal;
  }
}

export class TensorSession {
  constructor(token, data) {
    if (token !== SESSION_TOKEN) fail('TENSOR_SESSION_CONSTRUCTION_FORBIDDEN', 'validation', 'TensorSession capabilities are created by TensorSession.open().');
    SESSION_DATA.set(this, data);
    Object.freeze(this);
  }

  static async open(options) {
    const normalized = normalizeOpen(options);
    let runtime = normalized.runtime;
    let runtimeOwnership = normalized.runtimeOwnership;
    let runtimeDescription = null;
    let runtimeCreated = false;
    try {
      if (!runtime) {
        runtime = await openCudaRuntime({ ...(normalized.device !== undefined ? { device: normalized.device } : {}) });
        runtimeOwnership = 'owned';
        runtimeCreated = true;
      }
      runtimeDescription = await runtime.describe();
      compatibility(runtimeDescription);
    } catch (error) {
      if (runtimeCreated && runtime) {
        try { await runtime.close(); } catch (cleanupError) {
          fail('TENSOR_SESSION_OPEN_ROLLBACK_UNPROVED', 'cleanup-unproved', 'TensorSession.open failed and the created CUDA-JS runtime cleanup was not proved.', { primary: failureSummary(error), cleanup: failureSummary(cleanupError) }, { cause: error });
        }
      }
      throw error;
    }
    const data = {
      state: 'open',
      identity: `tensor-session:${++sessionSequence}:${createHash('sha256').update(JSON.stringify({ runtime: runtimeDescription, limits: normalized.limits, defaults: normalized.defaults })).digest('hex')}`,
      runtime,
      runtimeDescription,
      runtimeOwnership,
      limits: normalized.limits,
      defaults: normalized.defaults,
      tensors: new Set(),
      resolvedPlans: new Set(),
      pendingCreates: 0,
      pendingResolutions: 0,
      reservedBytes: 0,
      unprovedResources: 0,
      terminal: null,
    };
    const session = new TensorSession(SESSION_TOKEN, data);
    data.session = session;
    return session;
  }

  get kind() { return 'tensor-session'; }
  get state() { return sessionData(this).state; }
  get compatibilityIdentity() { return sessionData(this).identity; }
  get runtime() { return sessionData(this).runtime; }
  get limits() { return sessionData(this).limits; }
  get defaults() { return sessionData(this).defaults; }
  describe() { return sessionDescriptor(sessionData(this)); }
  status() { return this.describe(); }

  async allocate(specOrOptions) {
    const data = valueState(sessionData(this), 'TensorSession.allocate');
    const spec = specOrOptions instanceof TensorSpec ? specOrOptions : createTensorSpec({ ...data.defaults, ...specOrOptions });
    return data.session.createTensor(spec, null, { ownership: 'owned' });
  }

  async createTensor(spec, memory = null, options = {}) {
    const data = valueState(sessionData(this), 'TensorSession.createTensor');
    const owned = options.ownership !== 'borrowed';
    const storageIdentity = options.storageIdentity ?? `tensor-storage:${data.identity}:${data.tensors.size + data.pendingCreates + 1}`;
    const reserve = Math.max(spec.requiredByteEnd, spec.alignment);
    if (reserve > data.limits.maxTensorBytes) fail('TENSOR_SESSION_TENSOR_LIMIT', 'pressure', 'Tensor exceeds the session per-tensor byte limit.', { required: reserve, maximum: data.limits.maxTensorBytes });
    if (data.reservedBytes + reserve > data.limits.maxSessionBytes) fail('TENSOR_SESSION_BYTE_LIMIT', 'pressure', 'Tensor exceeds the session byte limit.', { required: data.reservedBytes + reserve, maximum: data.limits.maxSessionBytes });
    if (data.tensors.size + data.pendingCreates >= data.limits.maxLiveTensors) fail('TENSOR_SESSION_TENSOR_COUNT_LIMIT', 'pressure', 'Tensor session live tensor limit reached.', { maximum: data.limits.maxLiveTensors });
    data.pendingCreates += 1;
    let allocation = memory;
    let deviceView = null;
    let allocationOwned = false;
    try {
      if (!allocation) {
        allocation = await data.runtime.allocateDevice({ byteLength: reserve });
        allocationOwned = true;
      }
      deviceView = await allocation.view({ dtype: spec.dtype, byteOffset: spec.byteOffset, elementCount: Math.max(0, spec.logicalElementCount), access: spec.access });
      const tensor = new Tensor(TENSOR_TOKEN, {
        state: 'open', terminal: null, session: data, spec, memory: allocation, deviceView, storageIdentity,
        ownership: owned ? 'owned' : 'borrowed', reservedBytes: owned ? reserve : 0,
      });
      data.tensors.add(tensor);
      if (owned) data.reservedBytes += reserve;
      return tensor;
    } catch (error) {
      const cleanup = [];
      if (deviceView?.state === 'open') {
        try { await deviceView.close(); } catch (cleanupError) { cleanup.push(failureSummary(cleanupError)); }
      }
      if (allocationOwned && allocation?.state === 'open') {
        try { await allocation.close(); } catch (cleanupError) { cleanup.push(failureSummary(cleanupError)); }
      }
      if (cleanup.length > 0) {
        data.unprovedResources += 1;
        fail('TENSOR_ALLOCATION_ROLLBACK_UNPROVED', 'cleanup-unproved', 'Tensor allocation failed and CUDA-JS resource cleanup was not proved.', { primary: failureSummary(error), cleanup }, { cause: error });
      }
      throw error;
    } finally {
      data.pendingCreates -= 1;
    }
  }

  async close() {
    const data = sessionData(this);
    if (data.state === 'closed' || data.state === 'orphaned') return data.terminal;
    if (data.pendingCreates > 0 || data.pendingResolutions > 0) fail('TENSOR_SESSION_BUSY', 'backpressure', 'Tensor session has in-progress child construction.', { pendingCreates: data.pendingCreates, pendingResolutions: data.pendingResolutions });
    data.state = 'closing';
    const failures = [];
    for (const result of [...data.resolvedPlans]) {
      if (result.state === 'closed' || result.state === 'orphaned') continue;
      try {
        const report = await result.close();
        if (!report.graceful) failures.push(...report.failures);
      } catch (error) { failures.push(failureSummary(error)); }
    }
    for (const tensor of [...data.tensors]) {
      if (tensor.state === 'closed' || tensor.state === 'orphaned') continue;
      try {
        const report = await tensor.close();
        if (!report.graceful) failures.push(...report.failures);
      } catch (error) { failures.push(failureSummary(error)); }
    }
    if (data.runtimeOwnership === 'owned' && data.runtime?.state !== 'closed') {
      try {
        const report = await data.runtime.close();
        if (!report.graceful) failures.push(...(report.failures ?? []));
      } catch (error) { failures.push(failureSummary(error)); }
    }
    data.state = failures.length === 0 ? 'closed' : 'orphaned';
    data.terminal = deepFreeze({ schemaVersion: 1, kind: 'tensor-session', state: data.state, graceful: failures.length === 0, failures });
    return data.terminal;
  }
}

export function inspectTensorForSession(session, tensor) {
  const sessionValue = valueState(sessionData(session), 'inspectTensorForSession');
  const tensorValue = valueState(tensorData(tensor), 'inspectTensorForSession');
  if (tensorValue.session !== sessionValue) fail('TENSOR_SESSION_OWNERSHIP', 'validation', 'Tensor belongs to a different TensorSession.');
  return Object.freeze({
    spec: tensorValue.spec,
    deviceView: tensorValue.deviceView,
    storageIdentity: tensorValue.storageIdentity,
  });
}

export function inspectTensorSessionForExecution(session) {
  const data = valueState(sessionData(session), 'inspectTensorSessionForExecution');
  return Object.freeze({ runtime: data.runtime, runtimeDescription: data.runtimeDescription, limits: data.limits, defaults: data.defaults });
}

export function reserveTensorSessionExecution(session) {
  const data = valueState(sessionData(session), 'reserveTensorSessionExecution');
  data.pendingResolutions += 1;
  let released = false;
  return Object.freeze({
    runtime: data.runtime,
    inspection: Object.freeze({ limits: data.limits, defaults: data.defaults, runtimeDescription: data.runtimeDescription }),
    commit(plan) {
      if (released) fail('TENSOR_SESSION_RESERVATION_CLOSED', 'internal', 'Tensor session resolution reservation is already closed.');
      data.resolvedPlans.add(plan);
      data.pendingResolutions -= 1;
      released = true;
    },
    release() {
      if (released) return;
      data.pendingResolutions -= 1;
      released = true;
    },
  });
}
