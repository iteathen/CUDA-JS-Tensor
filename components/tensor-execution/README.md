# Resolved tensor plan and dense execution

This component implements the complete `TENSOR-SIMT-012` baseline under SPEC-0005, optional `TENSOR-CUBLASLT-013` under SPEC-0006, optional exact elementwise fusion under SPEC-0007, and item-parallel device-callable Tensor programs under SPEC-0009. Host-planned resolution submits generated kernels and selected public cuBLASLt nodes through one CUDA-JS prepared DAG. Device-callable compilation instead returns one copied typed CUDA-JS leaf library for import into an independently owned Device-JS program.

The easy and expert forms normalize identically:

```js
const program = TensorProgram.define((graph) => {
  const input = graph.input('input', { dtype: 'f32', capacityShape: [1024] });
  return graph.unary('sqrt', input);
});

const easy = await resolveTensorPlan(session, program);
const explicit = await ResolvedTensorPlan.create(
  session,
  TensorPlan.create(program),
  { backend: 'simt', fusion: 'none', blockSize: 256, maxWorkspaceBytes: 64 * 1024 * 1024 },
);

await input.write(inputBytes);
const result = await easy.run(input);
const outputBytes = (await result.output.read()).bytes;
await result.close();
await easy.close();
```

The resolved plan owns compiler/module/function/fixed-library-plan/prepared resources and cascades its live results. Resolved plans share one public cuBLASLt adapter lease per session/runtime and the last user closes it after its plans. A result owns only its run-created output material, derived views and explicit reduction/accelerator workspace; it borrows inputs. The session cascades resolved plans before tensors and its runtime. Result cleanup is retryable while caller-created child views remain live and remains explicit if any terminal cleanup cannot be proved.

The SIMT path is complete for SPEC-0004's first dense catalog. SPEC-0006 may replace only eligible rank-2 contiguous f32 matmul kernels at their existing DAG position. SPEC-0007 may replace deterministic two-or-more-node cast/unary/binary regions with one exact generated kernel and delete only their unobservable internal materials. `simt` and `fusion: 'none'` remain the convenience defaults; `prefer-cublaslt` records exact fallbacks, `cublaslt` rejects any ineligible/non-admitted matmul, and `fusion: 'exact-elementwise'` is an explicit composable selection. There is no maintained CUDA/PTX source, private CUDA-JS import, hidden host callback loop, arena reuse, automatic performance policy or speedup claim here.

`compileTensorDeviceProgram(session, program, { itemCapacity, itemInputs })` statically proves that every material and output belongs to one independent item. The returned `TensorDeviceProgram` exposes exact input/output/workspace descriptors and `importAs(alias)`. It owns no runtime scheduler, launch, queue, allocation, result readiness or close lifecycle. Static shared inputs such as weights may be read by every item; public outputs and dtype-partitioned workspace are disjoint item-major regions. Cross-item reductions and ambiguous item transforms reject before CUDA-JS compiler work.

Executable, tested analysis frames for remaining future work live outside this production component in [`../../experiments/tensor-implementation-frames/`](../../experiments/tensor-implementation-frames/). The superseded fusion and device-callable discovery frames were deleted when their continuation value moved into accepted production profiles and tests. Remaining frames are not imported, exported, consulted by resolution, or evidence of production support.
