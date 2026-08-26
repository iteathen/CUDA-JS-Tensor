# Project Charter

## Purpose

Make finite GPU tensor work easy without making simple use shallow or expert use impossible. CUDA-JS-Tensor provides complete explicit contracts and a convenience layer of smart documented defaults that normalizes to them.

## Intended consumers

Neural inference, search/evaluation systems, DSP, simulation, image processing, optimization, scientific computing, and future materially different consumers may use the library. None owns the core vocabulary.

## Product boundary

CUDA-JS-Tensor owns tensor dtype/shape/layout/view semantics, immutable operation programs, finite plan resolution, mathematical operation meaning, backend equivalence, and user-facing tensor/session convenience. CUDA-JS owns native CUDA providers, selected devices/contexts, memory/typed views, compilation/linking, operations/graphs, library loading, device-callable library composition, and cleanup.

Training loops, neural layers, autodiff, losses, optimizers, checkpoints, model formats, MCGS policy, domain encoders, data pipelines, distributed training, and NCCL policy are outside this repository.

## Success criteria

- A caller can open a safe default session and perform useful dense tensor work with minimal configuration.
- An expert can specify exact dtype, shape, stride, active extent, precision, aliasing, workspace, backend and lifecycle policies.
- Both forms resolve to one inspectable canonical plan.
- The complete SIMT path works without optional accelerators.
- Qualified accelerated variants are strongly recommended only where end-to-end evidence justifies them.
- Removing CUDA-MCGS, neural inference, or any first consumer leaves the tensor library coherent.
- No CUDA-JS internal or native implementation leaks into the public contract.

## Phase policy

This is a public pre-release project. No stability, publication, native support, performance, or production-readiness claim exists until separately qualified and explicitly released.
