# Repository context: CUDA-JS-Tensor

Universal engineering and design guidance comes from the account-global `AGENTS.md`. This file contains only CUDA-JS-Tensor-specific context.

## Mission and ownership

CUDA-JS-Tensor owns consumer-neutral Tensor mathematics and planning: shapes, dtypes, layouts/strides/views, generic Tensor operations, TensorProgram/TensorPlan semantics, item-axis Tensor semantics, callable ABI/workspace, and Tensor-specific conformance.

CUDA-JS owns generic CUDA runtime/compiler/memory/execution/provider mechanisms. CUDA-JS-Tensor does not own NN/training, MCGS/search, chess, model, or downstream product semantics.

## Local routing

- `STATUS.md` and `next_step.yaml` — current repository state and next seam.
- `docs/decisions/` and `docs/specs/` — accepted local authority.
- Package/compatibility files — current public package projection.

## Local constraints

Host code is ordinary JavaScript and device work uses accepted Device-JS through public CUDA-JS contracts. Direct native CUDA/FFI, private CUDA-JS imports, and consumer-specific semantics do not belong here.

## Local validation

```bash
npm run verify
```
