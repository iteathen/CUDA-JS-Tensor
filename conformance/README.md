# Conformance

Conformance fixtures exercise the installed public package independently of component tests. They are evidence consumers, not production implementation and not additional API authority.

The native lane is exact-profile gated under SPEC-0005. Its JavaScript fixture uses only public CUDA-JS and CUDA-JS-Tensor exports, independent expected mathematics and terminal resource accounting. It contains no maintained CUDA C++, PTX or private import.
