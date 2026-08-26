# Resolved tensor plan and generated SIMT execution

This component implements accepted `TENSOR-SIMT-012` under SPEC-0005. It turns a backend-neutral `TensorPlan` or `TensorProgram` into one immutable session-bound generated Device-JS realization, then submits all nonempty kernels through one public CUDA-JS prepared DAG.

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
  { backend: 'simt', blockSize: 256, maxWorkspaceBytes: 64 * 1024 * 1024 },
);

await input.write(inputBytes);
const result = await easy.run(input);
const outputBytes = (await result.output.read()).bytes;
await result.close();
await easy.close();
```

The resolved plan owns compiler/module/function/prepared resources and cascades its live results. A result owns only its run-created output material, derived views and reduction workspace; it borrows inputs. The session cascades resolved plans before tensors and its runtime. Result cleanup is retryable while caller-created child views remain live and remains explicit if any terminal cleanup cannot be proved.

The SIMT path is complete for SPEC-0004's first dense catalog. View nodes are zero-kernel indexing facts; material nodes generate deterministic restricted Device-JS. Fixed-tree reductions use explicit finite workspace and staged adjacent-pair kernels. There is no maintained CUDA/PTX source, private CUDA-JS import, hidden host callback loop, fusion, arena reuse, accelerator selection or performance claim here.
