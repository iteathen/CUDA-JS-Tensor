# Immutable tensor program and static plan

This component implements the pure portion of `TENSOR-PROGRAM-011` under SPEC-0002/SPEC-0004.

`TensorProgram.create(record)` is the canonical finite DAG form. `TensorProgram.define(builder => outputs)` is the convenience overload; the temporary builder performs identical validation and returns the same immutable program identity. Operations are functional: view nodes inherit alias classes and material nodes receive distinct planned storage.

`TensorPlan.create(program)` calculates exact dependency order, output specifications, definition/last-use liveness, declared alias classes, physical allocation bounds and observable outputs. Its first policy intentionally performs no reuse or fusion. `plan.executable` is `false` and `plan.unresolved` names the session/backend facts still owned by `TENSOR-SIMT-012`; a static plan is not presented as a resolved or executable plan.

```js
const program = TensorProgram.define((graph) => {
  const a = graph.input('a', { dtype: 'f32', capacityShape: [32, 64] });
  const b = graph.input('b', { dtype: 'f32', capacityShape: [64, 16] });
  return { product: graph.matmul(a, b) };
});

const plan = TensorPlan.create(program);
console.log(plan.outputs, plan.totalDistinctBytes, plan.unresolved);
```

No code in this component imports CUDA-JS, opens a runtime, allocates device memory, compiles a kernel, selects cuBLASLt, or owns consumer policy.
