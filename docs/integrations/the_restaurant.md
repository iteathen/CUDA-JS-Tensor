# Deferred `the_restaurant` integration plan

**Status:** Deferred; plan retained; no repository changes authorized  
**Date:** 2026-08-26

`the_restaurant` is an existing neural-network training system with its own CUDA API. It is not a dependency or first implementation target for CUDA-JS-Tensor.

When explicitly resumed, use separate assessed steps:

1. qualify the training system on the Node version required by the selected CUDA-JS/CUDA-JS-Tensor pair;
2. inventory its current CUDA and LibTorch ownership without replacing working training backends by assumption;
3. add a default-off adapter at a named operation boundary and prove parity against the existing backend;
4. migrate individual operations only when lifecycle, numerical, performance and maintenance evidence favors it;
5. keep model, training, optimizer, checkpoint and data-pipeline semantics in `the_restaurant`;
6. preserve rollback and deletion: removing the adapter leaves both repositories complete.

No clone, branch, issue, code, dependency, Node upgrade, or setting change in `the_restaurant` is part of the current plan.

