import type { CudaDeviceSelector, CudaRuntime } from 'cuda-js';

export type TensorDtype = 'u32' | 'u64' | 'i32' | 'f32' | 'f64' | 'f16' | 'bf16';
export type TensorAccess = 'read' | 'write' | 'read-write';
export type TensorLayout = 'row-major-contiguous' | 'strided';

export interface TensorSpecOptions {
  dtype?: TensorDtype;
  capacityShape: readonly number[];
  activeAxis0?: number | Readonly<{ extent: number; maximum: number }> | null;
  strides?: readonly number[];
  byteOffset?: number;
  alignment?: number;
  access?: TensorAccess;
  aliasGroup?: string | null;
}

export class TensorError extends Error {
  readonly code: string;
  readonly category: string;
  readonly details: Readonly<Record<string, unknown>>;
}

export class TensorSpec {
  private constructor();
  static create(options: TensorSpecOptions): TensorSpec;
  static create(shape: readonly number[]): TensorSpec;
  static create(dtype: TensorDtype, shape: readonly number[]): TensorSpec;
  readonly contract: 'SPEC-0001-tensor-spec-v1';
  readonly dtype: TensorDtype;
  readonly dtypeWidth: number;
  readonly rank: number;
  readonly capacityShape: readonly number[];
  readonly activeAxis0: Readonly<{ extent: number; maximum: number }> | null;
  readonly logicalShape: readonly number[];
  readonly strides: readonly number[];
  readonly layout: TensorLayout;
  readonly byteOffset: number;
  readonly alignment: number;
  readonly access: TensorAccess;
  readonly aliasGroup: string | null;
  readonly capacityElementCount: number;
  readonly logicalElementCount: number;
  readonly storageElementSpan: number;
  readonly byteLength: number;
  readonly requiredByteEnd: number;
  readonly hasBroadcastAliasing: boolean;
  readonly compatibilityIdentity: string;
  readonly canonical: Readonly<Record<string, unknown>>;
  toJSON(): Readonly<Record<string, unknown>>;
}

export interface TensorSessionLimits {
  maxTensorBytes?: number;
  maxSessionBytes?: number;
  maxLiveTensors?: number;
}

export interface TensorSessionDefaults {
  dtype?: TensorDtype;
  access?: TensorAccess;
}

export interface TensorSessionOptions {
  runtime?: CudaRuntime;
  runtimeOwnership?: 'owned' | 'borrowed';
  device?: CudaDeviceSelector;
  limits?: TensorSessionLimits;
  defaults?: TensorSessionDefaults;
}

export class TensorSession {
  private constructor();
  static open(): Promise<TensorSession>;
  static open(runtime: CudaRuntime): Promise<TensorSession>;
  static open(options: TensorSessionOptions): Promise<TensorSession>;
  readonly kind: 'tensor-session';
  readonly state: string;
  readonly ownershipMode: 'owned' | 'borrowed';
  readonly limits: Readonly<Required<TensorSessionLimits>>;
  readonly defaults: Readonly<Required<TensorSessionDefaults>>;
  readonly compatibilityIdentity: string;
  readonly resolvedOpenOptions: Readonly<Record<string, unknown>>;
  allocate(spec: TensorSpec): Promise<Tensor>;
  allocate(options: TensorSpecOptions): Promise<Tensor>;
  allocate(shape: readonly number[]): Promise<Tensor>;
  allocate(dtype: TensorDtype, shape: readonly number[]): Promise<Tensor>;
  status(): Promise<Readonly<Record<string, unknown>>>;
  close(): Promise<Readonly<Record<string, unknown>>>;
}

export class Tensor {
  private constructor();
  readonly kind: 'tensor';
  readonly state: string;
  readonly spec: TensorSpec;
  readonly dtype: TensorDtype;
  readonly capacityShape: readonly number[];
  readonly logicalShape: readonly number[];
  readonly strides: readonly number[];
  readonly byteLength: number;
  readonly access: TensorAccess;
  readonly aliasGroup: string;
  readonly sessionCompatibilityIdentity: string;
  view(): Promise<Tensor>;
  view(spec: TensorSpec): Promise<Tensor>;
  view(options: TensorSpecOptions): Promise<Tensor>;
  view(shape: readonly number[]): Promise<Tensor>;
  status(): Promise<Readonly<Record<string, unknown>>>;
  close(): Promise<Readonly<Record<string, unknown>>>;
}

export const CUDA_JS_TENSOR_COMPATIBILITY: Readonly<Record<string, unknown>>;
export const TENSOR_DTYPES: readonly TensorDtype[];
export const TENSOR_ACCESS_ROLES: readonly TensorAccess[];
export const TENSOR_SPEC_CONTRACT: 'SPEC-0001-tensor-spec-v1';
export function createTensorSpec(options: TensorSpecOptions): TensorSpec;
export function createTensorSpec(shape: readonly number[]): TensorSpec;
export function createTensorSpec(dtype: TensorDtype, shape: readonly number[]): TensorSpec;
export function tensorDtypeWidth(dtype: TensorDtype | string): number | null;
