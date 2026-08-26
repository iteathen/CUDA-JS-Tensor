import { compileDeviceLibrary } from 'cuda-js';

import { TensorPlan, TensorProgram } from '../../tensor-program/index.mjs';
import { inspectTensorSessionForExecution, reserveTensorSessionExecution } from '../../tensor-value/internal.mjs';

import { deepFreeze, exactRecord, fail, identity } from './contract.mjs';
import { createDeviceItemProfile, TENSOR_DEVICE_PROGRAM_CONTRACT } from './device-item-profile.mjs';

const OPTION_FIELDS = new Set(['itemCapacity', 'itemInputs', 'output', 'maxWorkspaceBytes']);
const IDENTIFIER = /^[A-Za-z_$][A-Za-z0-9_$]*$/;
const TOKEN = Symbol('TensorDeviceProgram');
const DATA = new WeakMap();

function staticPlan(value) {
  if (value instanceof TensorPlan) return value;
  if (value instanceof TensorProgram) return TensorPlan.create(value);
  fail('TENSOR_DEVICE_PLAN_INVALID', 'validation', 'compileTensorDeviceProgram requires a TensorPlan or TensorProgram.');
}

function data(value, operation) {
  const result = DATA.get(value);
  if (!result) fail('TENSOR_DEVICE_PROGRAM_INVALID', 'validation', `${operation} requires a TensorDeviceProgram capability.`);
  return result;
}

function copyLibrary(library) {
  const artifact = Object.freeze({
    ...library.artifact,
    bytes: Uint8Array.from(library.artifact.bytes),
    ...(library.artifact.producer ? { producer: Object.freeze({ ...library.artifact.producer }) } : {}),
  });
  return Object.freeze({
    ...library,
    exports: Object.freeze(library.exports.map((entry) => Object.freeze({
      ...entry,
      parameters: Object.freeze(entry.parameters.map((parameter) => Object.freeze({ ...parameter }))),
    }))),
    artifact,
  });
}

export class TensorDeviceProgram {
  constructor(token, value) {
    if (token !== TOKEN) fail('TENSOR_DEVICE_PROGRAM_CONSTRUCTION_INVALID', 'validation', 'Use compileTensorDeviceProgram().');
    DATA.set(this, value);
    Object.freeze(this);
  }

  get kind() { return 'tensor-device-program'; }
  get contract() { return TENSOR_DEVICE_PROGRAM_CONTRACT; }
  get plan() { return data(this, 'TensorDeviceProgram.plan').plan; }
  get itemCapacity() { return data(this, 'TensorDeviceProgram.itemCapacity').profile.itemCapacity; }
  get itemInputs() { return data(this, 'TensorDeviceProgram.itemInputs').profile.itemInputs; }
  get outputFormat() { return data(this, 'TensorDeviceProgram.outputFormat').profile.output; }
  get parameters() { return data(this, 'TensorDeviceProgram.parameters').profile.parameters; }
  get inputs() { return data(this, 'TensorDeviceProgram.inputs').profile.inputs; }
  get outputs() { return data(this, 'TensorDeviceProgram.outputs').profile.outputs; }
  get workspace() { return data(this, 'TensorDeviceProgram.workspace').profile.workspace; }
  get totalWorkspaceBytes() { return data(this, 'TensorDeviceProgram.totalWorkspaceBytes').profile.totalWorkspaceBytes; }
  get function() { return data(this, 'TensorDeviceProgram.function').publicFunction; }
  get library() { return copyLibrary(data(this, 'TensorDeviceProgram.library').library); }
  get compatibilityIdentity() { return data(this, 'TensorDeviceProgram.compatibilityIdentity').compatibilityIdentity; }
  get canonical() { return data(this, 'TensorDeviceProgram.canonical').canonical; }

  importAs(alias) {
    const value = data(this, 'TensorDeviceProgram.importAs');
    if (typeof alias !== 'string' || !IDENTIFIER.test(alias) || alias === 'gpu') fail('TENSOR_DEVICE_IMPORT_ALIAS_INVALID', 'validation', 'Device-JS import alias must be a valid non-gpu identifier.', { alias: typeof alias === 'string' ? alias : null });
    return Object.freeze({ library: copyLibrary(value.library), name: value.exportedFunction.name, as: alias });
  }

  describe() { return data(this, 'TensorDeviceProgram.describe').canonical; }
}

export async function compileTensorDeviceProgram(session, planOrProgram, options) {
  exactRecord(options, OPTION_FIELDS, 'TENSOR_DEVICE_OPTIONS_INVALID', 'Device-callable Tensor options contain unknown fields.');
  const plan = staticPlan(planOrProgram);
  const profile = createDeviceItemProfile(plan, options);
  const inspection = inspectTensorSessionForExecution(session);
  if (inspection.runtime.compilerEnabled !== true) fail('TENSOR_DEVICE_COMPILER_REQUIRED', 'unsupported', 'Device-callable Tensor compilation requires a compiler-enabled CUDA-JS runtime.');
  const reservation = reserveTensorSessionExecution(session);
  try {
    const compiled = await compileDeviceLibrary(inspection.runtime, {
      source: profile.lowering.source,
      functions: [profile.lowering.function],
      exports: [profile.lowering.function.name],
      output: profile.output,
    });
    const exportedFunction = compiled.library.exports.find((entry) => entry.name === profile.lowering.function.name);
    if (!exportedFunction) fail('TENSOR_DEVICE_EXPORT_MISSING', 'internal', 'CUDA-JS omitted the generated Tensor device export.');
    const publicFunction = Object.freeze({
      name: exportedFunction.name,
      parameters: Object.freeze(exportedFunction.parameters.map((entry) => Object.freeze({ ...entry }))),
      returns: exportedFunction.returns,
    });
    const compiledIdentity = deepFreeze({
      profileIdentity: profile.compatibilityIdentity,
      sessionRuntimeIdentity: inspection.runtimeIdentity,
      library: {
        contract: compiled.library.contract,
        sha256: compiled.library.sha256,
        format: compiled.library.format,
        architecture: compiled.library.architecture,
        artifactSha256: compiled.library.artifact.sha256,
      },
      exportedFunction: publicFunction,
    });
    const compatibilityIdentity = identity('tensor-device-program-v1', compiledIdentity);
    const canonical = deepFreeze({
      contract: TENSOR_DEVICE_PROGRAM_CONTRACT,
      compatibilityIdentity,
      profile: profile.canonical,
      sessionCompatibilityIdentity: inspection.compatibilityIdentity,
      compiled: compiledIdentity,
      compiler: {
        format: compiled.compiler.artifact.format,
        architecture: compiled.compiler.artifact.architecture,
        sha256: compiled.compiler.artifact.sha256,
        headerProfile: compiled.compiler.headerProfile ?? null,
      },
    });
    return new TensorDeviceProgram(TOKEN, Object.freeze({ plan, profile, library: compiled.library, exportedFunction, publicFunction, compatibilityIdentity, canonical }));
  } finally {
    reservation.cancel();
  }
}
