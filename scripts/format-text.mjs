import { execFileSync } from 'node:child_process';
import { extname } from 'node:path';
import { readFileSync, writeFileSync } from 'node:fs';

const textExtensions = new Set(['.json', '.md', '.mjs', '.yaml', '.yml']);
const textNames = new Set(['.gitattributes', '.gitignore', 'LICENSE']);
const files = execFileSync('git', ['ls-files', '--cached', '--others', '--exclude-standard', '-z'], { encoding: 'utf8' })
  .split('\0')
  .filter(Boolean)
  .filter((file) => textExtensions.has(extname(file)) || textNames.has(file));

for (const file of files) {
  const source = readFileSync(file, 'utf8');
  const formatted = `${source
    .replaceAll('\r\n', '\n')
    .replace(/[ \t]+$/gm, '')
    .replace(/\n+$/u, '')}\n`;

  if (formatted !== source) writeFileSync(file, formatted, 'utf8');
}

console.log(`Normalized ${files.length} tracked text files.`);
