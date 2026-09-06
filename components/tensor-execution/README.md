# Tensor execution

This component resolves immutable Tensor plans into GPU execution through public CUDA-JS contracts.

It implements the accepted dense SIMT path, optional bounded cuBLASLt matmul, exact elementwise fusion, and device-callable programs for independent items. The default is SIMT without fusion; acceleration is explicit and has no general speedup claim.

Resolved plans own execution resources and their results. Results borrow inputs and own run-created output/workspace. Cleanup is explicit; device-callable programs instead return a library for use by a caller-owned device program and do not create a scheduler.

Start with [programs and plans](../tensor-program/README.md) and [sessions and values](../tensor-value/README.md). The [public interface](index.mjs) and [specifications](../../docs/specs/README.md) define resolution, running, results, and cleanup.

[Native conformance](../../conformance/native/README.md) exercises the public path. [Analysis frames](../../experiments/tensor-implementation-frames/README.md) explore future work and are not runtime capabilities.
