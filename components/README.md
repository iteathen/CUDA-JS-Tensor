# Components

Production code is organized by one visible owner per tensor contract.

- [`tensor-value/`](tensor-value/README.md) owns `TensorSession`, `TensorSpec`, tensor allocation/view capabilities, session limits, runtime ownership, and terminal cleanup.
- [`tensor-program/`](tensor-program/README.md) owns immutable backend-neutral dense programs and finite static plans without runtime/backend resolution.
- [`tensor-execution/`](tensor-execution/README.md) owns session-bound resolved plans, complete generated SIMT lowering, prepared execution, item-parallel device-callable compilation and per-run result cleanup.
- [`public-api/`](public-api/README.md) is the installed-package facade over accepted public component ports.

Optional acceleration remains removable behind the execution owner. Every component consumes `tensor-value` only through its declared public or package-internal ports.
