# Components

Production code is organized by one visible owner per tensor contract.

- [`tensor-value/`](tensor-value/README.md) owns `TensorSession`, `TensorSpec`, tensor allocation/view capabilities, session limits, runtime ownership, and terminal cleanup.

Program planning, dense operation semantics, generated SIMT execution, and accelerated adapters remain separate planned components. They consume `tensor-value` only through its declared public or package-internal ports.
