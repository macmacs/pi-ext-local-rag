import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, mkdtempSync, rmSync, writeFileSync, promises as fsPromises } from "node:fs";
import { extname, basename, dirname, join, relative } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import { spawnSync } from "node:child_process";
import ignore from "ignore";
import { BINARY_DOC_EXTS, TEXT_MAX_BYTES, BINARY_DOC_MAX_BYTES, SKIP_DIRS } from "./constants.ts";
import { loadConfig, resolveExtensions, type RagConfig } from "./config.ts";
import { getRagDir, ocrCacheDir } from "./store.ts";

const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

/** The slice of RagConfig the directory walkers actually consult. */
type TrackedPathsConfig = Pick<RagConfig, "trackedPaths" | "excludePatterns">;

function stderrProgress(msg: string) { process.stderr.write(`\r\x1b[2K${msg}`); }

export function sha256(data: string): string {
  return createHash("sha256").update(data).digest("hex").slice(0, 12);
}

export function chunkText(text: string, maxLines = 50): { content: string; lineStart: number; lineEnd: number }[] {
  const lines = text.split("\n");
  const chunks: { content: string; lineStart: number; lineEnd: number }[] = [];
  let i = 0;
  while (i < lines.length) {
    let end = Math.min(i + maxLines, lines.length);
    for (let j = end - 1; j > i + 10 && j > end - 15; j--) {
      if (lines[j]?.trim() === "") { end = j + 1; break; }
    }
    const chunk = lines.slice(i, end).join("\n");
    if (chunk.trim().length > 20) {
      chunks.push({ content: chunk, lineStart: i + 1, lineEnd: end });
    }
    i = end;
  }
  return chunks;
}

export function collectFiles(
  dirPath: string,
  exts?: Set<string>,
  excludePatterns: string[] = [],
): string[] {
  const allowed = exts ?? resolveExtensions(loadConfig());
  const ig = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const files: string[] = [];
  const root = dirPath;

  function acceptable(fp: string, size: number): boolean {
    const ext = extname(fp).toLowerCase();
    if (allowed.has(ext)) return size < TEXT_MAX_BYTES;
    if (BINARY_DOC_EXTS.has(ext)) return size < BINARY_DOC_MAX_BYTES;
    return false;
  }

  function isExcluded(absPath: string): boolean {
    if (!ig) return false;
    const rel = relative(root, absPath);
    if (!rel || rel.startsWith("..")) return false;
    return ig.ignores(rel);
  }

  try {
    const stat = statSync(dirPath);
    if (stat.isFile()) {
      if (!acceptable(dirPath, stat.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch { return []; }

  function walk(dir: string) {
    try {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        const fp = join(dir, entry.name);
        if (entry.isDirectory()) {
          if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
          if (isExcluded(fp)) continue;
          walk(fp);
        } else {
          const ext = extname(entry.name).toLowerCase();
          if (!allowed.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
          if (isExcluded(fp)) continue;
          try {
            if (acceptable(fp, statSync(fp).size)) files.push(fp);
          } catch {}
        }
      }
    } catch {}
  }
  walk(root);
  return files;
}

/** Takes only the fields it walks, so callers (and tests) need not build a whole RagConfig. */
export function collectFromTracked(cfg: TrackedPathsConfig): string[] {
  const out = new Set<string>();
  for (const p of cfg.trackedPaths) {
    if (!existsSync(p)) continue;
    for (const f of collectFiles(p, undefined, cfg.excludePatterns)) out.add(f);
  }
  return [...out];
}

/**
 * Async variant of collectFiles that uses fs.promises and yields to the event
 * loop between directories. Required for /rag rebuild on large trackedPaths
 * (45k+ files) — the synchronous walk pegs the event loop long enough that
 * the TUI freezes before reaching the embed phase. Adapted from
 * theli-ua/pi-local-rag@8432a15.
 */
export async function collectFilesAsync(
  dirPath: string,
  exts?: Set<string>,
  excludePatterns: string[] = [],
): Promise<string[]> {
  const allowed = exts ?? resolveExtensions(loadConfig());
  const ig = excludePatterns.length ? ignore().add(excludePatterns) : null;
  const files: string[] = [];
  const root = dirPath;

  function acceptable(fp: string, size: number): boolean {
    const ext = extname(fp).toLowerCase();
    if (allowed.has(ext)) return size < TEXT_MAX_BYTES;
    if (BINARY_DOC_EXTS.has(ext)) return size < BINARY_DOC_MAX_BYTES;
    return false;
  }

  function isExcluded(absPath: string): boolean {
    if (!ig) return false;
    const rel = relative(root, absPath);
    if (!rel || rel.startsWith("..")) return false;
    return ig.ignores(rel);
  }

  try {
    const st = await fsPromises.stat(dirPath);
    if (st.isFile()) {
      if (!acceptable(dirPath, st.size)) return [];
      if (ig && ig.ignores(basename(dirPath))) return [];
      return [dirPath];
    }
  } catch { return []; }

  async function walk(dir: string): Promise<void> {
    let entries: import("node:fs").Dirent[];
    try {
      entries = await fsPromises.readdir(dir, { withFileTypes: true });
    } catch { return; }
    for (const entry of entries) {
      const fp = join(dir, entry.name);
      if (entry.isDirectory()) {
        if (SKIP_DIRS.has(entry.name) || entry.name.startsWith(".")) continue;
        if (isExcluded(fp)) continue;
        await walk(fp);
      } else {
        const ext = extname(entry.name).toLowerCase();
        if (!allowed.has(ext) && !BINARY_DOC_EXTS.has(ext)) continue;
        if (isExcluded(fp)) continue;
        try {
          const st = await fsPromises.stat(fp);
          if (acceptable(fp, st.size)) files.push(fp);
        } catch {}
      }
    }
    // Yield between directories so the event loop can process UI updates.
    await yield_();
  }

  await walk(root);
  return files;
}

export async function collectFromTrackedAsync(cfg: TrackedPathsConfig): Promise<string[]> {
  const out = new Set<string>();
  for (const p of cfg.trackedPaths) {
    if (!existsSync(p)) continue;
    for (const f of await collectFilesAsync(p, undefined, cfg.excludePatterns)) out.add(f);
  }
  return [...out];
}

/** Returns true if `file` is matched by `excludePatterns` relative to any of `roots`. */
export function isExcludedByConfig(file: string, roots: string[], excludePatterns: string[]): boolean {
  if (!excludePatterns.length) return false;
  const ig = ignore().add(excludePatterns);
  for (const root of roots) {
    const rel = relative(root, file);
    if (!rel || rel.startsWith("..")) continue;
    if (ig.ignores(rel)) return true;
  }
  return false;
}

// pdfjs (bundled inside pdf-parse) routes warnings through console.log with a
// "Warning: " prefix. On real-world PDFs this fires thousands of times per
// document ("Ran out of space in font private use area", missing glyphs, …).
// The font warnings come from pdf.worker.js, which is a separate webpack
// bundle whose verbosity is not externally configurable (its setVerbosityLevel
// export exists only as a placeholder at the outer module level). Filtering
// console.log for the known pdfjs prefixes is the only reliable approach.
const PDFJS_LOG_PREFIX = /^(Warning|Info|Deprecated API usage):/;

// Malformed PDFs also make pdfjs reject *detached* promises — per-page and
// per-XObject parse tasks that pdf-parse starts but never awaits. Those
// rejections have no owner, so a try/catch around pdf() cannot see them and
// Node's default `--unhandled-rejections=throw` kills the whole agent process
// with e.g. `FormatError: bad XRef entry` while pdf() itself resolves happily.
// The only place to intercept them is the process-level event, so while a PDF
// parse is in flight we install a listener that drops rejections originating
// in the bundled pdfjs and rethrows everything else to preserve default
// crash-on-unhandled semantics for the host.
const PDFJS_STACK = /[\\/]pdf-parse[\\/]lib[\\/]pdf\.js[\\/]/;
const PDFJS_ERROR_NAMES = new Set([
  "FormatError", "InvalidPDFException", "MissingPDFException", "MissingDataException",
  "UnexpectedResponseException", "UnknownErrorException", "XRefParseException",
  "XRefEntryException", "PasswordException", "AbortException",
]);
// Detached rejections surface a tick or more after pdf() settles, so the
// listener has to outlive the call that provoked them.
const PDFJS_GUARD_GRACE_MS = 10_000;

let _pdfjsGuardDepth = 0;
let _pdfjsGuardTimer: ReturnType<typeof setTimeout> | undefined;
let _pdfjsGuardListener: ((reason: unknown) => void) | undefined;
let _pdfjsSwallowedLogged = false;
/** Count of pdfjs rejections dropped by the guard. Exported for tests. */
export let pdfjsSwallowedRejections = 0;
/** Count of PDFs that failed to parse and fell through to OCR. Exported for tests. */
export let pdfParseFailures = 0;

function isPdfjsRejection(reason: unknown): boolean {
  if (!(reason instanceof Error)) return false;
  if (typeof reason.stack === "string" && PDFJS_STACK.test(reason.stack)) return true;
  return PDFJS_ERROR_NAMES.has(reason.name);
}

function acquirePdfjsGuard(): void {
  _pdfjsGuardDepth++;
  if (_pdfjsGuardTimer) { clearTimeout(_pdfjsGuardTimer); _pdfjsGuardTimer = undefined; }
  if (_pdfjsGuardListener) return;
  _pdfjsGuardListener = (reason: unknown) => {
    if (isPdfjsRejection(reason)) {
      pdfjsSwallowedRejections++;
      if (!_pdfjsSwallowedLogged) {
        _pdfjsSwallowedLogged = true;
        process.stderr.write(
          `\r\x1b[2K[rag] malformed PDF internals detected; ignoring parser errors and indexing whatever text was recovered\n`
        );
      }
      return;
    }
    // Not ours. Adding any listener disables Node's default crash, so if we are
    // the only listener we have to reproduce it.
    if (process.listenerCount("unhandledRejection") === 1) throw reason;
  };
  process.on("unhandledRejection", _pdfjsGuardListener);
}

function releasePdfjsGuard(): void {
  if (--_pdfjsGuardDepth > 0) return;
  _pdfjsGuardDepth = 0;
  _pdfjsGuardTimer = setTimeout(() => {
    _pdfjsGuardTimer = undefined;
    if (_pdfjsGuardDepth > 0 || !_pdfjsGuardListener) return;
    process.off("unhandledRejection", _pdfjsGuardListener);
    _pdfjsGuardListener = undefined;
  }, PDFJS_GUARD_GRACE_MS);
  _pdfjsGuardTimer.unref?.();
}

async function withPdfjsSilenced<T>(fn: () => Promise<T>): Promise<T> {
  const origLog = console.log;
  console.log = (...args: unknown[]) => {
    const first = args[0];
    if (typeof first === "string" && PDFJS_LOG_PREFIX.test(first)) return;
    origLog(...args);
  };
  acquirePdfjsGuard();
  try {
    return await fn();
  } finally {
    console.log = origLog;
    releasePdfjsGuard();
  }
}

// ─── OCR fallback for image-based PDFs ───────────────────────────────────────

type OcrTooling = { available: false } | { available: true; langs: string };
let _ocrTooling: OcrTooling | undefined;
let _ocrUnavailableLogged = false;

/**
 * One-shot probe for system pdftoppm + tesseract. Caches the result.
 *
 * The languages come from `ocrLanguages` in the config, in preference order -
 * tesseract weights the first entry of a `deu+eng` chain most heavily, and
 * running German scans under `-l eng` mangles umlauts badly enough to poison
 * the index. Languages with no installed traineddata are dropped rather than
 * passed through, since tesseract exits non-zero on an unknown `-l`.
 */
export function getOcrTooling(): OcrTooling {
  if (_ocrTooling) return _ocrTooling;
  const pdftoppm = spawnSync("pdftoppm", ["-v"]);
  const tess = spawnSync("tesseract", ["--list-langs"], { encoding: "utf-8" });
  if (pdftoppm.error || tess.error) return (_ocrTooling = { available: false });
  // tesseract prints langs on stderr in some builds, stdout in others.
  const out = `${tess.stdout || ""}\n${tess.stderr || ""}`;
  const have = new Set(out.split(/\r?\n/).map(s => s.trim()).filter(Boolean));

  let configured: string[];
  try { configured = loadConfig().ocrLanguages; } catch { configured = ["eng", "deu"]; }
  const wanted = configured.filter(l => have.has(l));
  if (!wanted.length) {
    if (configured.length && have.size) {
      process.stderr.write(
        `\r\x1b[2K[rag] OCR: no traineddata for ${configured.join(", ")} - install it (e.g. apt install tesseract-ocr-deu) or fix ocrLanguages\n`
      );
    }
    return (_ocrTooling = { available: false });
  }
  return (_ocrTooling = { available: true, langs: wanted.join("+") });
}

/**
 * OCR is by far the most expensive thing the indexer does - minutes per scanned
 * document - and its input is immutable: the same bytes under the same
 * languages always produce the same text. Results are therefore cached on disk
 * under the rag dir, keyed by content hash + language chain, so a forced
 * rebuild, an embedding-model change or a fresh index never re-runs tesseract
 * on a document it has already read. Changing `ocrLanguages` misses the cache
 * and re-OCRs, which is the intended behaviour.
 */
function ocrCachePath(hash: string, langs: string): string {
  return join(ocrCacheDir(getRagDir()), `${hash}-${langs.replace(/\+/g, "_")}.txt`);
}

function readOcrCache(hash: string, langs: string): string | undefined {
  try {
    const p = ocrCachePath(hash, langs);
    return existsSync(p) ? readFileSync(p, "utf-8") : undefined;
  } catch { return undefined; }
}

function writeOcrCache(hash: string, langs: string, text: string): void {
  try {
    const p = ocrCachePath(hash, langs);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, text, "utf-8");
  } catch { /* cache is an optimization; a failure to write is not fatal */ }
}

/** Render `buf` to PNGs via pdftoppm, OCR each page via tesseract, return concatenated text. */
async function ocrPdf(buf: Buffer, langs: string, label: string): Promise<string> {
  const MAX_PAGES = 200;
  const PER_PAGE_TIMEOUT_MS = 60_000;
  const dir = mkdtempSync(join(tmpdir(), "rag-ocr-"));
  try {
    const pdfPath = join(dir, "in.pdf");
    writeFileSync(pdfPath, buf);
    const render = spawnSync("pdftoppm", ["-png", "-r", "200", pdfPath, join(dir, "p")], { encoding: "utf-8" });
    if (render.status !== 0) return "";
    const pages = readdirSync(dir).filter(f => f.startsWith("p-") && f.endsWith(".png")).sort();
    const total = Math.min(pages.length, MAX_PAGES);
    if (pages.length > MAX_PAGES) {
      process.stderr.write(`\r\x1b[2K[rag] OCR ${label}: ${pages.length} pages, capping at ${MAX_PAGES}\n`);
    }
    const out: string[] = [];
    for (let i = 0; i < total; i++) {
      stderrProgress(`[OCR ${i + 1}/${total}] ${label}`);
      await yield_();
      const r = spawnSync("tesseract", [join(dir, pages[i]), "-", "-l", langs], {
        encoding: "utf-8",
        timeout: PER_PAGE_TIMEOUT_MS,
        maxBuffer: 16 * 1024 * 1024,
      });
      out.push(r.status === 0 ? (r.stdout ?? "") : "");
    }
    process.stderr.write(`\r\x1b[2K`);
    return out.join("\n\n");
  } finally {
    try { rmSync(dir, { recursive: true, force: true }); } catch {}
  }
}

/** True if `text` looks too sparse for `numpages` to be the real content of the document. */
export function isSparsePdfText(text: string, numpages: number): boolean {
  return text.trim().length < 50 * Math.max(1, numpages);
}

// Identity of a file as stored in the `files` table. Kept in one place so
// `extractText` and `fileIdentity` can never drift apart - the indexer compares
// one against the other to decide whether a file needs re-reading at all.
const binaryIdentity = (buf: Buffer) => ({ hash: sha256(buf.toString("binary")), size: buf.length });
const textIdentity = (raw: string) => ({ hash: sha256(raw), size: raw.length });

/**
 * Hash and size of a file without decoding it - the same values `extractText`
 * returns, for a fraction of the cost. The indexer calls this first so an
 * unchanged PDF is skipped before anything pays for pdfjs or OCR.
 */
export function fileIdentity(fp: string): { hash: string; size: number } {
  const ext = extname(fp).toLowerCase();
  if (BINARY_DOC_EXTS.has(ext)) return binaryIdentity(readFileSync(fp));
  return textIdentity(readFileSync(fp, "utf-8"));
}

/**
 * Read and decode a file into UTF-8 text. PDF and DOCX are routed through
 * extraction libraries; everything else is read as plain UTF-8. Hash is
 * computed over the raw bytes for binaries (so the source file's identity
 * drives skip-on-rebuild) and over the decoded text for plain text files.
 */
export async function extractText(fp: string): Promise<{ text: string; hash: string; size: number }> {
  const ext = extname(fp).toLowerCase();
  if (ext === ".pdf") {
    const buf = readFileSync(fp);
    const { default: pdf } = await import("pdf-parse/lib/pdf-parse.js");
    // Hand pdfjs a plain Uint8Array, never the Node Buffer. pdfjs derives its
    // sub-streams from `bytes.buffer` and ignores `byteOffset`, so a Buffer
    // that sits inside Node's shared allocation pool (anything under ~64 KB,
    // which is most single-page documents) makes it read object data from a
    // neighbouring buffer and fail a perfectly valid file with "bad XRef entry".
    const bytes = new Uint8Array(buf);
    // A PDF broken badly enough to fail outright (bad xref, truncated file) is
    // still worth an OCR attempt — pdftoppm is far more forgiving than pdfjs —
    // so treat the failure as "no text, one page" and fall through to that path.
    let data: { text: string; numpages?: number };
    try {
      data = await withPdfjsSilenced(() => pdf(bytes));
    } catch (e) {
      pdfParseFailures++;
      process.stderr.write(`\r\x1b[2K[rag] ${basename(fp)}: PDF parse failed (${(e as Error).message}), trying OCR\n`);
      data = { text: "", numpages: 1 };
    }
    let text = data.text;
    const identity = binaryIdentity(buf);
    if (isSparsePdfText(text, data.numpages ?? 1)) {
      const tools = getOcrTooling();
      if (tools.available) {
        const cached = readOcrCache(identity.hash, tools.langs);
        const ocr = cached ?? await ocrPdf(buf, tools.langs, basename(fp));
        // Only a non-empty result is worth keeping: an empty one means pdftoppm
        // or tesseract failed, and that is worth retrying on the next run.
        if (cached === undefined && ocr.trim()) writeOcrCache(identity.hash, tools.langs, ocr);
        if (ocr.trim().length > text.trim().length) text = ocr;
      } else if (!_ocrUnavailableLogged) {
        _ocrUnavailableLogged = true;
        process.stderr.write(
          `\r\x1b[2K[rag] OCR unavailable: install pdftoppm + tesseract (with traineddata for your ocrLanguages) to index image PDFs\n`
        );
      }
    }
    return { text, ...identity };
  }
  if (ext === ".docx") {
    const buf = readFileSync(fp);
    const { default: mammoth } = await import("mammoth");
    const { value } = await mammoth.extractRawText({ buffer: buf });
    return { text: value, ...binaryIdentity(buf) };
  }
  if (ext === ".html" || ext === ".htm") {
    const { default: TurndownService } = await import("turndown");
    const raw = readFileSync(fp, "utf-8");
    const td = new TurndownService({
      headingStyle: "atx",
      codeBlockStyle: "fenced",
      blankReplacement: (_content, node) => node.tagName === "BR" ? "\n" : "",
    });
    td.remove(["script", "style"]);
    td.remove(["nav", "footer"]);
    const text = td.turndown(raw);
    return { text, ...textIdentity(raw) };
  }
  const text = readFileSync(fp, "utf-8");
  return { text, ...textIdentity(text) };
}
