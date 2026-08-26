import type { Tensor, TensorSession, TensorSpec } from '../tensor-value/index.mjs';
import type { TensorPlan, TensorProgram } from '../tensor-program/index.mjs';
import type { DeviceJsImport, DeviceJsLibrary } from 'cuda-js';

export interface ResolveTensorPlanOptions {
  backend?: 'simt' | 'prefer-cublaslt' | 'cublaslt';
  blockSize?: 32 | 64 | 128 | 256 | 512 | 1024;
  maxWorkspaceBytes?: number;
  fusion?: 'none' | 'exact-elementwise';
}

export type TensorExecutionBindings = Readonly<Record<string, Tensor>> | readonly Tensor[] | Tensor;

export class TensorExecutionResult {
  private constructor();
  readonly kind: 'tensor-execution-result';
  readonly contract: 'SPEC-0005-tensor-execution-result-v1';
  readonly state: string;
  readonly outputs: Readonly<Record<string, Tensor>>;
  readonly output: Tensor | null;
  readonly execution: Readonly<Record<string, unknown>>;
  readonly compatibilityIdentity: string;
  get(name: string): Tensor;
  close(): Promise<Readonly<Record<string, unknown>>>;
}

export class ResolvedTensorPlan {
  private constructor();
  static create(session: TensorSession, plan: TensorPlan, options?: ResolveTensorPlanOptions): Promise<ResolvedTensorPlan>;
  static create(session: TensorSession, program: TensorProgram, options?: ResolveTensorPlanOptions): Promise<ResolvedTensorPlan>;
  readonly kind: 'resolved-tensor-plan';
  readonly contract: 'SPEC-0006-resolved-dense-plan-v1+SPEC-0007-exact-elementwise-fusion-v1';
  readonly state: string;
  readonly plan: TensorPlan;
  readonly backend: 'simt' | 'cublaslt' | 'mixed';
  readonly backendPolicy: 'simt' | 'prefer-cublaslt' | 'cublaslt';
  readonly fusionPolicy: 'none' | 'exact-elementwise';
  readonly fusionRegionCount: number;
  readonly fusedNodeCount: number;
  readonly compatibilityIdentity: string;
  readonly kernelCount: number;
  readonly cublasLtNodeCount: number;
  readonly bindingCount: number;
  readonly workspaceBytes: number;
  readonly canonical: Readonly<Record<string, unknown>>;
  describe(): Readonly<Record<string, unknown>>;
  run(): Promise<TensorExecutionResult>;
  run(tensor: Tensor): Promise<TensorExecutionResult>;
  run(tensors: readonly Tensor[]): Promise<TensorExecutionResult>;
  run(bindings: Readonly<Record<string, Tensor>>): Promise<TensorExecutionResult>;
  close(): Promise<Readonly<Record<string, unknown>>>;
}

export function resolveTensorPlan(session: TensorSession, plan: TensorPlan, options?: ResolveTensorPlanOptions): Promise<ResolvedTensorPlan>;
export function resolveTensorPlan(session: TensorSession, program: TensorProgram, options?: ResolveTensorPlanOptions): Promise<ResolvedTensorPlan>;

export const RESOLVED_TENSOR_PLAN_CONTRACT: 'SPEC-0006-resolved-dense-plan-v1+SPEC-0007-exact-elementwise-fusion-v1';
export const TENSOR_EXECUTION_RESULT_CONTRACT: 'SPEC-0005-tensor-execution-result-v1';
export const TENSOR_SIMT_LIMITS: Readonly<{ maxKernels: 32; maxBindings: 64; maxLogicalWorkItems: 4294967295; maxWorkspaceBytes: number }>;
export const TENSOR_BACKEND_POLICIES: readonly ['simt', 'prefer-cublaslt', 'cublaslt'];
export const TENSOR_FUSION_POLICIES: readonly ['none', 'exact-elementwise'];

export interface CompileTensorDeviceProgramOptions {
  itemCapacity: number;
  itemInputs: readonly string[];
  output?: 'ptx' | 'lto-ir';
  maxWorkspaceBytes?: number;
}

export interface TensorDeviceProgramParameterBase {
  readonly parameterIndex: number;
  readonly parameterName: string;
  readonly type: string;
  readonly dtype: string;
  readonly access: 'read' | 'write' | 'read-write';
  readonly itemVarying: boolean;
}

export interface TensorDeviceItemIndexParameter extends TensorDeviceProgramParameterBase {
  readonly role: 'item-index';
  readonly type: 'u32';
  readonly dtype: 'u32';
  readonly access: 'read';
  readonly itemVarying: false;
}

export interface TensorDeviceInputParameter extends TensorDeviceProgramParameterBase {
  readonly role: 'input';
  readonly type: `ptr<${string}>`;
  readonly access: 'read';
  readonly name: string;
  readonly valueId: string;
  readonly spec: TensorSpec;
  readonly elementCount: number;
  readonly byteLength: number;
}

export interface TensorDeviceOutputParameter extends TensorDeviceProgramParameterBase {
  readonly role: 'output';
  readonly type: `ptr<${string}>`;
  readonly access: 'write';
  readonly itemVarying: true;
  readonly name: string;
  readonly valueId: string;
  readonly spec: TensorSpec;
  readonly perItemElements: number;
  readonly elementCount: number;
  readonly byteLength: number;
}

export interface TensorDeviceWorkspaceParameter extends TensorDeviceProgramParameterBase {
  readonly role: 'workspace';
  readonly type: `ptr<${string}>`;
  readonly access: 'read-write';
  readonly itemVarying: true;
  readonly perItemElements: number;
  readonly elementCount: number;
  readonly byteLength: number;
  readonly alignmentBytes: number;
  readonly materials: readonly Readonly<Record<string, unknown>>[];
  readonly scratches: readonly Readonly<Record<string, unknown>>[];
}

export type TensorDeviceProgramParameter = TensorDeviceItemIndexParameter | TensorDeviceInputParameter | TensorDeviceOutputParameter | TensorDeviceWorkspaceParameter;

export class TensorDeviceProgram {
  private constructor();
  readonly kind: 'tensor-device-program';
  readonly contract: 'SPEC-0009-item-parallel-device-tensor-program-v1';
  readonly plan: TensorPlan;
  readonly itemCapacity: number;
  readonly itemInputs: readonly string[];
  readonly outputFormat: 'ptx' | 'lto-ir';
  readonly parameters: readonly TensorDeviceProgramParameter[];
  readonly inputs: readonly TensorDeviceInputParameter[];
  readonly outputs: readonly TensorDeviceOutputParameter[];
  readonly workspace: readonly TensorDeviceWorkspaceParameter[];
  readonly totalWorkspaceBytes: number;
  readonly function: Readonly<{ name: 'tensorRunItem'; parameters: readonly Readonly<{ name: string; type: string }>[]; returns: 'u32' }>;
  readonly library: DeviceJsLibrary;
  readonly compatibilityIdentity: string;
  readonly canonical: Readonly<Record<string, unknown>>;
  importAs(alias: string): DeviceJsImport;
  describe(): Readonly<Record<string, unknown>>;
}

export function compileTensorDeviceProgram(session: TensorSession, plan: TensorPlan, options: CompileTensorDeviceProgramOptions): Promise<TensorDeviceProgram>;
export function compileTensorDeviceProgram(session: TensorSession, program: TensorProgram, options: CompileTensorDeviceProgramOptions): Promise<TensorDeviceProgram>;

export const TENSOR_DEVICE_PROGRAM_CONTRACT: 'SPEC-0009-item-parallel-device-tensor-program-v1';
export const TENSOR_DEVICE_PROGRAM_OUTPUTS: readonly ['ptx', 'lto-ir'];
export const TENSOR_DEVICE_PROGRAM_LIMITS: Readonly<{ maxParameters: 64; maxItemCapacity: 4294967295; defaultMaxWorkspaceBytes: number; maxWorkspaceBytes: number }>;
