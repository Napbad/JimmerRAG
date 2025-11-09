#!/usr/bin/env node

/**
 * Lightweight MDX -> text extractor for the ingestion pipeline.
 * It removes YAML frontmatter and import/export statements, returning JSON
 * with { metadata, content }.
 */

const fs = require("fs");
const path = require("path");
const yaml = require("yaml");

function readFile(filePath) {
  try {
    return fs.readFileSync(filePath, "utf8");
  } catch (error) {
    console.error(`无法读取文件：${filePath}\n${error.message}`);
    process.exit(1);
  }
}

function extractFrontMatter(text) {
  const frontMatterMatch = text.match(/^---\n([\s\S]*?)\n---\n?/);
  if (!frontMatterMatch) {
    return { metadata: {}, content: text };
  }

  let metadata = {};
  try {
    metadata = yaml.parse(frontMatterMatch[1]) || {};
  } catch (error) {
    console.warn("YAML frontmatter 解析失败，已忽略：", error.message);
  }

  const content = text.slice(frontMatterMatch[0].length);
  return { metadata, content };
}

function normaliseContent(text) {
  const lines = text.split(/\r?\n/);
  const cleaned = lines.filter((line) => {
    const trimmed = line.trim();
    if (
      trimmed.startsWith("import ") ||
      trimmed.startsWith("export ") ||
      trimmed.startsWith("const ") ||
      trimmed.startsWith("let ") ||
      trimmed.startsWith("var ")
    ) {
      return false;
    }
    return true;
  });

  return cleaned.join("\n").replace(/\n{3,}/g, "\n\n").trim();
}

function main() {
  const filePath = process.argv[2];
  if (!filePath) {
    console.error("用法：node parse_mdx.js <path-to-mdx>");
    process.exit(1);
  }

  const absolutePath = path.resolve(process.cwd(), filePath);
  const raw = readFile(absolutePath);
  const { metadata, content } = extractFrontMatter(raw);
  const normalised = normaliseContent(content);

  const output = {
    metadata: {
      ...metadata,
      source: absolutePath,
    },
    content: normalised,
  };

  process.stdout.write(JSON.stringify(output, null, 2));
}

main();

