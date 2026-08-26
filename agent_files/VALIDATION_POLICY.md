# Validation Policy

## Foundation gate

Run `npm run verify` for every change. It validates repository structure, links/registries, package identity, accepted status vocabulary, forbidden native source, and JavaScript tests.

## Implementation gate

Each component adds ownership-focused unit tests, mutation/failure tests, first-consumer deletion, lifecycle cleanup, cross-session rejection, and package-consumer evidence.

## Native gate

Native promotion additionally requires:

- an exact qualified CUDA-JS public package/revision;
- exact Node/OS/ABI/Driver/toolkit/GPU/provider identity;
- independent mathematical outputs, not package self-comparison;
- semantic equivalence for dtype/shape/layout/aliasing/precision;
- terminal resource and workspace cleanup;
- explicit unsupported shapes and claim limits.

## Performance gate

Acceleration is recommended only after reproducible representative evidence includes dispatch, planning, packing, padding, conversion, workspace, synchronization, warmup, small/tail batches, latency distribution, memory high-water, and the best credible SIMT baseline. Isolated Tensor Core utilization or GEMM throughput is insufficient.

