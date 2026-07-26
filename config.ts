import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { configFile, getRagDir } from "./store.ts";
import { DEFAULT_TEXT_EXTS } from "./constants.ts";

/**
 * Gitignore-style patterns excluded from indexing by default. These match
 * files that are technically in the extension allowlist (mostly `.json`) but
 * are pure machine-generated noise — indexing them pollutes semantic search
 * and auto-injection with lock-file / minified boilerplate. Users can override
 * via `/rag exclude` (add) or by editing config.
 */
export const DEFAULT_EXCLUDE_PATTERNS = [
  "**/package-lock.json",
  "**/pnpm-lock.yaml",
  "**/yarn.lock",
  "**/bun.lockb",
  "**/Cargo.lock",
  "**/poetry.lock",
  "**/composer.lock",
  "**/Gemfile.lock",
  "**/*.min.js",
  "**/*.min.css",
  "**/*.map",
];

export interface RagConfig {
  ragEnabled: boolean;
  ragTopK: number;
  ragScoreThreshold: number;
  ragAlpha: number; // 0 = pure vector, 1 = pure BM25
  extraExtensions: string[];   // user-added file extensions (e.g. [".cs", ".tex"])
  excludeExtensions: string[]; // extensions to drop from the default set
  trackedPaths: string[];      // absolute paths previously passed to /rag index
  excludePatterns: string[];   // gitignore-style path patterns
}

export function defaultConfig(): RagConfig {
  return {
    // ragScoreThreshold gates auto-injection. Hybrid scores are min-max
    // normalized per query, so a low floor (e.g. 0.1) lets the top-K of
    // WHATEVER is indexed through even when nothing is truly relevant — which
    // is how lock-file / vendored noise ends up injected. 0.35 keeps only
    // reasonably strong matches; tune with `/rag` config if too strict.
    ragEnabled: true, ragTopK: 5, ragScoreThreshold: 0.35, ragAlpha: 0.4,
    extraExtensions: [], excludeExtensions: [],
    trackedPaths: [], excludePatterns: [...DEFAULT_EXCLUDE_PATTERNS],
  };
}

export function loadConfig(): RagConfig {
  const cfgFile = configFile(getRagDir());
  if (!existsSync(cfgFile)) return defaultConfig();
  try {
    return { ...defaultConfig(), ...JSON.parse(readFileSync(cfgFile, "utf-8")) };
  } catch { return defaultConfig(); }
}

export function saveConfig(config: RagConfig) {
  writeFileSync(configFile(getRagDir()), JSON.stringify(config, null, 2));
}

/** Normalize a user-supplied extension to lowercase ".ext" form. */
export function normalizeExt(ext: string): string {
  const trimmed = ext.trim().toLowerCase();
  if (!trimmed) return "";
  return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
}

/** Build the effective extension allowlist from defaults + user config. */
export function resolveExtensions(config: Pick<RagConfig, "extraExtensions" | "excludeExtensions">): Set<string> {
  const set = new Set(DEFAULT_TEXT_EXTS);
  for (const e of config.extraExtensions) {
    const n = normalizeExt(e);
    if (n) set.add(n);
  }
  for (const e of config.excludeExtensions) {
    const n = normalizeExt(e);
    if (n) set.delete(n);
  }
  return set;
}
