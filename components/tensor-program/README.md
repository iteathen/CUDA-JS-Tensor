# Tensor programs and static plans

This component creates immutable tensor operation graphs and calculates finite static plans without opening a CUDA runtime.

`TensorProgram.create(record)` and the builder form use the same validation. `TensorPlan.create(program)` determines dependency order, output specifications, liveness, and storage bounds. A static plan remains non-executable until the separate [execution component](../tensor-execution/README.md) resolves runtime/backend requirements.

## Example

After installing the development package, this ES module constructs a matrix-multiplication plan without requiring a GPU:

```js
import { TensorProgram, TensorPlan } from 'cuda-js-tensor';

const program = TensorProgram.define((graph) => {
  const a = graph.input('a', { dtype: 'f32', capacityShape: [32, 64] });
  const b = graph.input('b', { dtype: 'f32', capacityShape: [64, 16] });
  return { product: graph.matmul(a, b) };
});

const plan = TensorPlan.create(program);
console.log(plan.outputs, plan.totalDistinctBytes, plan.unresolved);
```

See the [public interface](index.mjs) and [specifications](../../docs/specs/README.md) for the operation catalog and planning contracts.
