import { CUDA_JS_COMPATIBILITY } from 'cuda-js/compatibility';

import { fail } from './contract.mjs';

function positiveSafeInteger(value, field) {
  if (!Number.isSafeInteger(value) || value < 1) {
    fail('TENSOR_CUDA_JS_COMPATIBILITY_INVALID', 'internal', `CUDA-JS compatibility field ${field} is invalid.`, { field, value });
  }
  return value;
}

function stringSet(value, field) {
  if (!Array.isArray(value) || value.length === 0 || value.some((entry) => typeof entry !== 'string' || entry.length === 0)) {
    fail('TENSOR_CUDA_JS_COMPATIBILITY_INVALID', 'internal', `CUDA-JS compatibility field ${field} is invalid.`, { field });
  }
  return Object.freeze([...value]);
}

const capabilities = CUDA_JS_COMPATIBILITY?.capabilities;
if (capabilities === null || typeof capabilities !== 'object') {
  fail('TENSOR_CUDA_JS_COMPATIBILITY_INVALID', 'internal', 'CUDA-JS compatibility capabilities are unavailable.');
}

const prepared = capabilities.preparedOperationDagLimits;
if (prepared === null || typeof prepared !== 'object' || Array.isArray(prepared)) {
  fail('TENSOR_CUDA_JS_COMPATIBILITY_INVALID', 'internal', 'CUDA-JS prepared-operation-DAG compatibility limits are unavailable.');
}
const preparedOperationDagLimits = Object.freeze({
  nodes: positiveSafeInteger(prepared.nodes, 'preparedOperationDagLimits.nodes'),
  edges: positiveSafeInteger(prepared.edges, 'preparedOperationDagLimits.edges'),
  bindings: positiveSafeInteger(prepared.bindings, 'preparedOperationDagLimits.bindings'),
  predecessorsPerNode: positiveSafeInteger(prepared.predecessorsPerNode, 'preparedOperationDagLimits.predecessorsPerNode'),
});

const deviceJs = capabilities.deviceJsLimits;
if (deviceJs === null || typeof deviceJs !== 'object' || Array.isArray(deviceJs)) {
  fail('TENSOR_CUDA_JS_COMPATIBILITY_INVALID', 'internal', 'CUDA-JS Device-JS compatibility limits are unavailable.');
}
const deviceJsLimits = Object.freeze({
  parametersPerFunction: positiveSafeInteger(deviceJs.parametersPerFunction, 'deviceJsLimits.parametersPerFunction'),
});

const compilerOutputFormats = stringSet(capabilities.compilerOutputFormats, 'compilerOutputFormats');

export const CUDA_JS_TENSOR_COMPATIBILITY = Object.freeze({
  package: Object.freeze({
    name: CUDA_JS_COMPATIBILITY.package?.name ?? null,
    version: CUDA_JS_COMPATIBILITY.package?.version ?? null,
    publicApiSchema: CUDA_JS_COMPATIBILITY.publicApi?.schemaVersion ?? null,
  }),
  preparedOperationDagLimits,
  deviceJsLimits,
  compilerOutputFormats,
});

export function requireTensorDeviceLibraryOutput(output) {
  if (!compilerOutputFormats.includes(output)) {
    fail('TENSOR_DEVICE_PROGRAM_OUTPUT_UNSUPPORTED_BY_CUDA_JS', 'unsupported', 'The selected Tensor device-library output is not supported by the current CUDA-JS compatibility profile.', {
      output,
      supported: compilerOutputFormats,
    });
  }
  return output;
}
