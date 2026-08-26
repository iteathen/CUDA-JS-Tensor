# Public package facade

The `cuda-js-tensor` package root re-exports accepted public component contracts here. This facade adds no tensor semantics, CUDA dependency, state or lifecycle. Package-internal ports such as `tensor-value/internal.mjs` are intentionally absent and blocked by package exports.
