# CUDA-JS-Tensor

CUDA-JS-Tensor is a tensor planning and GPU execution library for JavaScript developers, built on public [CUDA-JS](https://github.com/iteathen/CUDA-JS) contracts.

**Public pre-release; the package is private and has not been published to npm.** It is not production-supported. Native evidence applies only to the exact recorded library and hardware profiles.

## What exists

The repository implements tensor values and views, immutable tensor programs and static plans, generated Device-JS execution, optional bounded cuBLASLt matmul, exact elementwise fusion, and device-callable programs for independent items.

A session owns or borrows one CUDA-JS runtime/device. Tensor shapes, types, workspace bounds, and cleanup are explicit. It provides generic mathematics; it does not provide model layers, training, graph search, or application scheduling.

## Intended development

The aim is reusable tensor computation for independent consumers. Further work includes qualifying complete consumer programs and their resource requirements, with additional operations or acceleration justified by those needs. Current limitations and priorities are recorded in [STATUS.md](STATUS.md) and the [plans](docs/plans/README.md).

There is no general Linux, Tensor Core, multi-GPU, or speedup claim. Prior native evidence does not automatically qualify a newer Tensor/CUDA-JS pair.

## Getting started

Source development requires Node.js 26.1.0 or later and Git:

```bash
git clone https://github.com/iteathen/CUDA-JS-Tensor.git
cd CUDA-JS-Tensor
npm ci
npm run verify
```

This checks repository structure and runs the Node test suite. CUDA-JS is fetched at the exact revision pinned by the package. GPU execution additionally requires that dependency's supported native environment.

Start with [tensor programs](components/tensor-program/README.md), then [sessions and values](components/tensor-value/README.md) and [execution](components/tensor-execution/README.md). [Native conformance](conformance/native/README.md) explains the separate GPU evidence path.

## Further information

- [Documentation](docs/README.md) and [project charter](docs/PROJECT_CHARTER.md).
- [Contributing](CONTRIBUTING.md) and [developer instructions](AGENTS.md).
- [Private security reporting](SECURITY.md).
- [AGPL-3.0-or-later license](LICENSE) and [commercial licensing information](LICENSING.md).
