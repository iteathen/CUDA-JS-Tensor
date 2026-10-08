# Public CUDA-JS cohort refresh implementation plan

> **For agentic workers:** Use superpowers:executing-plans with TDD; the owner already authorized autonomous producer fixes.

**Goal:** Allow the consumer-selected exact CUDA-JS alpha.21 package to compose through Tensor's public API without changing Tensor math, ABI, or workspace.

**Architecture:** Refresh the publication-guarded Tensor package to alpha.7, exact dependency revision, and compatibility projection. Preserve strict package/runtime version validation and all historical evidence. Qualify the same public math/callable contracts through portable tests and the existing native fixture.

**Tech stack:** JavaScript/Node.js, exact public CUDA-JS package, existing Tensor test/native runners.

**Spec:** ADR-0001 exact cross-repository compatibility; SPEC-0001 session/value; SPEC-0005 physical boundary; accepted SPEC-0009 callable ABI. Concrete consumer request supplies the required generic interoperability gap.

## Global constraints

- CUDA-JS owns native mechanisms; maintain no native/Python source or private imports.
- Current cohort is Tensor `0.1.0-alpha.7` with CUDA-JS `0.1.0-alpha.21` at `2bff226b752d3c0af8b9185274d411e5990008d4`.
- Strict identity checks stay intact; historical exact-pair records remain historical.
- Package metadata gains no native or performance claim before actual qualification.

## Review focus

- Selected public runtime must open successfully; an older runtime remains rejected.
- Generic callable compilation must preserve ABI and workspace bytes.
- Package lock, production compatibility, active test fixtures, and current status must agree.
- Legacy native fixture must assert the current pair rather than stale alpha.16.
- A successful portable compiler test must never be described as physical execution.

## Task: Refresh exact package cohort

- [x] Baseline `npm run verify`: 107 tests passed. Install exact CUDA-JS 21 without changing tracked files, reproduce `TENSOR_CUDA_JS_INCOMPATIBLE` (expected alpha.18, actual alpha.21), and prove borrowed runtime cleanup.
- [x] Add public-session/strict-rejection/callable regression tests; observe failures before production edits.
- [x] Refresh only package/lock, compatibility projection, active fixtures/current component metadata, and minimal status routing. Run `npm run verify`.
- [x] Run existing installed-package native conformance on exact Node 26.7.0, record exact pair and honest cleanup; retain historical evidence.
- [x] Pack the producer candidate outside source, commit bounded changes, and hand off package path for physical product comparison.

## Execution record

Worktree supplied by root: `E:/uci-arena-task-builds/tensor-vector-cohort-20261008`, branch `codex/vector-cuda-js-cohort-20261008`, protected base `061dce586d1a0a2df83d8350850956421bff6b65`. Product numerical reference stays downstream and is not a Tensor readiness gate.

Current native receipt records 109 portable/package tests and the existing unrelated installed-package fixture on Node 26.7.0/Windows/cc-7.5. The test's initial per-item-byte property assumption was corrected to the documented public workspace entry (`perItemElements: 3`, total bytes 24); production math and ABI required no change. No proactive optimization, version-range relaxation, private import, or consumer semantics were introduced.

Package retained at `E:/uci-arena-task-builds/tensor-vector-cohort-evidence-20261008/cuda-js-tensor-0.1.0-alpha.7.tgz`, SHA256 `7b9cdc84f2b6cb330d304a6052f56875d5372d63cfe9260f4b9a4e246360465d`. The source worktree and package are retained for integration review. Native runner removed its temporary consumer and reported graceful session/runtime cleanup; no background process remains.
