import assert from 'node:assert/strict';
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs';
import path from 'node:path';

const root = path.resolve(import.meta.dirname, '..');
const required = [
  'AGENTS.md', 'README.md', 'STATUS.md', 'LICENSE', 'LICENSING.md', 'CONTRIBUTING.md', 'SECURITY.md',
  'agent_files/AGENTS.md', 'agent_files/AI_RULES.md', 'agent_files/DESIGN_ALIGNMENT_CARD.md',
  'agent_files/SYSTEM_REGISTRY.md', 'agent_files/VALIDATION_POLICY.md', 'agent_files/general_foundation/PRINCIPLES.md',
  'docs/PROJECT_CHARTER.md', 'docs/architecture/TARGET_ARCHITECTURE.md',
  'docs/decisions/ADR-0001-separate-tensor-package-and-dependency-direction.md',
  'docs/specs/SPEC-0000-tensor-contract-map.md', 'docs/specs/SPEC-0001-tensor-session-spec-and-value-model.md',
  'docs/specs/SPEC-0002-tensor-program-plan-and-dense-operations.md',
  'docs/specs/SPEC-0003-accelerated-dense-backend-profiles.md',
  'docs/specs/SPEC-0004-first-dense-program-semantics.md',
  'docs/specs/SPEC-0005-resolved-simt-execution.md',
  'docs/specs/SPEC-0006-host-planned-cublaslt-matmul.md',
  'docs/specs/SPEC-0007-exact-elementwise-fusion.md',
  'docs/specs/SPEC-0009-item-parallel-device-callable-tensor-program.md',
  'docs/plans/2026-08-26-cuda-mcgs-readiness-assessment-and-plan.md',
  'docs/plans/2026-08-26-foundation-plan.md', 'docs/integrations/the_restaurant.md', 'next_step.yaml',
  '.github/dependabot.yml', '.github/ISSUE_TEMPLATE/config.yml', '.github/workflows/verify.yml',
  'conformance/README.md', 'conformance/native/README.md', 'conformance/native/fixtures/resolved-simt-consumer.mjs',
  'components/README.md', 'components/tensor-value/README.md', 'components/tensor-value/component.yaml',
  'components/tensor-value/index.mjs', 'components/tensor-value/index.d.ts', 'components/tensor-value/internal.mjs',
  'components/tensor-program/README.md', 'components/tensor-program/component.yaml', 'components/tensor-program/index.mjs', 'components/tensor-program/index.d.ts',
  'components/tensor-execution/README.md', 'components/tensor-execution/component.yaml', 'components/tensor-execution/index.mjs', 'components/tensor-execution/index.d.ts', 'components/tensor-execution/testing.mjs',
  'components/public-api/README.md', 'components/public-api/component.yaml', 'components/public-api/index.mjs', 'components/public-api/index.d.ts',
  'scripts/smoke-tensor-value-native.mjs',
  'scripts/run-tensor-execution-native.mjs',
];

for (const file of required) assert(existsSync(path.join(root, file)), `Missing required artifact: ${file}`);

const packageJson = JSON.parse(readFileSync(path.join(root, 'package.json'), 'utf8'));
assert.equal(packageJson.name, 'cuda-js-tensor');
assert.equal(packageJson.version, '0.1.0-alpha.6');
assert.equal(packageJson.private, true, 'Package must remain publication-guarded during foundation work.');
assert.equal(packageJson.license, 'AGPL-3.0-or-later');
assert.equal(packageJson.dependencies?.['cuda-js'], 'https://codeload.github.com/iteathen/CUDA-JS/tar.gz/4971302cfb48431c0843126a59d5884d84a81641');
assert.equal(packageJson.exports?.['.']?.import, './components/public-api/index.mjs');
JSON.parse(readFileSync(path.join(root, 'next_step.yaml'), 'utf8'));

const security = readFileSync(path.join(root, 'SECURITY.md'), 'utf8');
assert(security.includes('https://github.com/iteathen/CUDA-JS-Tensor/security/advisories/new'),
  'Security policy must route vulnerabilities to the enabled private reporting endpoint.');
const issueConfig = readFileSync(path.join(root, '.github/ISSUE_TEMPLATE/config.yml'), 'utf8');
assert(issueConfig.includes('https://github.com/iteathen/CUDA-JS-Tensor/security/advisories/new'),
  'Issue chooser must route vulnerabilities to the enabled private reporting endpoint.');
const workflow = readFileSync(path.join(root, '.github/workflows/verify.yml'), 'utf8');
for (const match of workflow.matchAll(/^\s*-\s+uses:\s*([^\s#]+)/gm)) {
  assert(/@[0-9a-f]{40}$/.test(match[1]), `Remote Action must use an immutable full commit: ${match[1]}`);
}
const dependabot = readFileSync(path.join(root, '.github/dependabot.yml'), 'utf8');
assert(dependabot.includes('package-ecosystem: github-actions'),
  'Dependabot must monitor GitHub Actions references.');

const component = JSON.parse(readFileSync(path.join(root, 'components/tensor-value/component.yaml'), 'utf8'));
assert.equal(component.component.id, 'tensor.value');
assert.equal(component.component.owner, 'tensor.session/tensor.spec/tensor.value');
assert.equal(JSON.parse(readFileSync(path.join(root, 'components/tensor-program/component.yaml'), 'utf8')).component.id, 'tensor.program-plan');
assert.equal(JSON.parse(readFileSync(path.join(root, 'components/tensor-execution/component.yaml'), 'utf8')).component.id, 'tensor.execution');
assert.equal(JSON.parse(readFileSync(path.join(root, 'components/public-api/component.yaml'), 'utf8')).component.id, 'tensor.public-api');

const ignored = new Set(['.git', 'build', 'node_modules']);
const files = [];
const visit = (directory) => {
  for (const name of readdirSync(directory)) {
    if (ignored.has(name)) continue;
    const target = path.join(directory, name);
    if (statSync(target).isDirectory()) visit(target);
    else files.push(path.relative(root, target).replaceAll('\\', '/'));
  }
};
visit(root);

const forbiddenNative = files.filter((file) => /\.(?:c|cc|cpp|cxx|cu|h|hh|hpp|ptx)$/i.test(file));
assert.deepEqual(forbiddenNative, [], `Forbidden maintained native source: ${forbiddenNative.join(', ')}`);

for (const file of files.filter((entry) => /\.(?:md|mjs|json|yaml|yml)$/i.test(entry))) {
  const text = readFileSync(path.join(root, file), 'utf8');
  assert(!/^(?:<<<<<<<|=======|>>>>>>>)/m.test(text), `Merge marker in ${file}`);
  assert(!/[ \t]+$/m.test(text), `Trailing whitespace in ${file}`);
  assert(text.endsWith('\n') && !text.endsWith('\n\n'), `Expected exactly one final newline in ${file}`);
}

console.log(`CUDA-JS-Tensor repository verification passed: ${files.length} maintained files, no native production source, package and authority identities aligned.`);
