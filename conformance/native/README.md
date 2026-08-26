# Native resolved-SIMT conformance

Run `npm run smoke:native:tensor-execution` with exact Node 26.7.0 on a CUDA-JS-qualified host. The runner packs and installs the current CUDA-JS-Tensor tarball into a temporary unrelated consumer, executes the public fixture, then removes the temporary installation.

The fixture covers view indexing plus materialization, fill, copy, cast including saturating special values, unary/binary half and bfloat behavior, broadcast, fixed-tree reduction, rank-2 matmul, prepared replay and terminal CUDA-JS cleanup. A pass qualifies only the recorded invocation profile. It does not establish performance, tensor-core use, accelerator selection, Linux, another GPU, or production stability.
