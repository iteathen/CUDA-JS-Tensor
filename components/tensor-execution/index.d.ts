import type { Tensor, TensorSession } from '../tensor-value/index.mjs';
import type { TensorPlan, TensorProgram } from '../tensor-program/index.mjs';

export interface ResolveTensorPlanOptions {
  backend?: 'simt' | 'prefer-cublaslt' | 'cublaslt';
  blockSize?: 32 | 64 | 128 | 256 | 512 | 1024;
  maxWorkspaceBytes?: number;
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
  readonly contract: 'SPEC-0006-resolved-dense-plan-v1';
  readonly state: string;
  readonly plan: TensorPlan;
  readonly backend: 'simt' | 'cublaslt' | 'mixed';
  readonly backendPolicy: 'simt' | 'prefer-cublaslt' | 'cublaslt';
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

export const RESOLVED_TENSOR_PLAN_CONTRACT: 'SPEC-0006-resolved-dense-plan-v1';
export const TENSOR_EXECUTION_RESULT_CONTRACT: 'SPEC-0005-tensor-execution-result-v1';
export const TENSOR_SIMT_LIMITS: Readonly<{ maxKernels: 32; maxBindings: 64; maxLogicalWorkItems: 4294967295; maxWorkspaceBytes: number }>;
export const TENSOR_BACKEND_POLICIES: readonly ['simt', 'prefer-cublaslt', 'cublaslt'];
