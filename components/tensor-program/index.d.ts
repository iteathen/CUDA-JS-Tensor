import type { TensorDtype, TensorSpec, TensorSpecOptions } from '../tensor-value/index.mjs';

export type UnaryOperator = 'neg' | 'abs' | 'exp' | 'log' | 'sqrt';
export type BinaryOperator = 'add' | 'sub' | 'mul' | 'div' | 'minimum' | 'maximum';
export type ReductionOperator = 'sum' | 'product' | 'minimum' | 'maximum';

export interface TensorProgramInput { name: string; spec: TensorSpec | TensorSpecOptions; }
export interface TensorProgramNode { id?: string; op: string; inputs: readonly string[]; options?: Readonly<Record<string, unknown>>; }
export interface TensorProgramOutput { name: string; value: string; }
export interface TensorProgramRecord { inputs: readonly TensorProgramInput[]; nodes: readonly TensorProgramNode[]; outputs: readonly TensorProgramOutput[]; }

export class TensorValueRef {
  private constructor();
  readonly kind: 'tensor-program-value';
  readonly id: string;
  readonly spec: TensorSpec;
}

export interface TensorProgramBuilder {
  input(name: string, spec: TensorSpec | TensorSpecOptions): TensorValueRef;
  node(op: string, inputs: readonly TensorValueRef[], options?: Readonly<Record<string, unknown>>, id?: string): TensorValueRef;
  fill(spec: TensorSpec | TensorSpecOptions, value: number | bigint): TensorValueRef;
  copy(input: TensorValueRef): TensorValueRef;
  cast(input: TensorValueRef, dtype: TensorDtype): TensorValueRef;
  reshape(input: TensorValueRef, capacityShape: readonly number[]): TensorValueRef;
  permute(input: TensorValueRef, axes: readonly number[]): TensorValueRef;
  slice(input: TensorValueRef, slices: readonly (null | Readonly<{ start: number; length: number; step?: number }>)[]): TensorValueRef;
  broadcast(input: TensorValueRef, capacityShape: readonly number[]): TensorValueRef;
  contiguous(input: TensorValueRef): TensorValueRef;
  unary(operator: UnaryOperator, input: TensorValueRef): TensorValueRef;
  binary(operator: BinaryOperator, left: TensorValueRef, right: TensorValueRef): TensorValueRef;
  reduce(operator: ReductionOperator, input: TensorValueRef, options?: Readonly<{ axes?: readonly number[]; keepDimensions?: boolean; accumulatorDtype?: TensorDtype; order?: 'fixed-tree-v1' | 'backend-defined'; identity?: number | bigint }>): TensorValueRef;
  matmul(left: TensorValueRef, right: TensorValueRef, options?: Readonly<{ transposeA?: boolean; transposeB?: boolean; accumulatorDtype?: TensorDtype }>): TensorValueRef;
  output(name: string, value: TensorValueRef): TensorValueRef;
}

export class TensorProgram {
  private constructor();
  static create(record: TensorProgramRecord): TensorProgram;
  static define(callback: (builder: TensorProgramBuilder) => TensorValueRef | Readonly<Record<string, TensorValueRef>> | void): TensorProgram;
  readonly kind: 'tensor-program';
  readonly contract: 'SPEC-0004-tensor-program-v1';
  readonly compatibilityIdentity: string;
  readonly inputs: readonly Readonly<Record<string, unknown>>[];
  readonly nodes: readonly Readonly<Record<string, unknown>>[];
  readonly outputs: readonly Readonly<Record<string, unknown>>[];
  readonly canonical: Readonly<Record<string, unknown>>;
  valueSpec(id: string): TensorSpec;
  describe(): Readonly<Record<string, unknown>>;
}

export class TensorPlan {
  private constructor();
  static create(program: TensorProgram, options?: Readonly<{ allocationPolicy?: 'distinct-v1' }>): TensorPlan;
  readonly kind: 'tensor-plan';
  readonly contract: 'SPEC-0004-static-tensor-plan-v1';
  readonly program: TensorProgram;
  readonly compatibilityIdentity: string;
  readonly allocationPolicy: 'distinct-v1';
  readonly inputRequirements: readonly Readonly<Record<string, unknown>>[];
  readonly operations: readonly Readonly<Record<string, unknown>>[];
  readonly liveness: readonly Readonly<Record<string, unknown>>[];
  readonly aliases: readonly Readonly<Record<string, unknown>>[];
  readonly allocations: readonly Readonly<Record<string, unknown>>[];
  readonly totalDistinctBytes: number;
  readonly outputs: readonly Readonly<Record<string, unknown>>[];
  readonly unresolved: readonly string[];
  readonly executable: false;
  readonly canonical: Readonly<Record<string, unknown>>;
  describe(): Readonly<Record<string, unknown>>;
}

export const TENSOR_PROGRAM_CONTRACT: 'SPEC-0004-tensor-program-v1';
export const TENSOR_PLAN_CONTRACT: 'SPEC-0004-static-tensor-plan-v1';
export const TENSOR_PROGRAM_LIMITS: Readonly<{ maxInputs: 256; maxNodes: 4096; maxOutputs: 256 }>;
