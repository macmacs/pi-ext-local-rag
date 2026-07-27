import { EMBEDDING_MODEL } from "./constants.ts";
import tls from "node:tls";
import { existsSync, readFileSync } from "node:fs";

let _pipeline: any = null;
let _caTrustEnsured = false;

/**
 * Some environments (corporate TLS interception, custom proxies) present a
 * certificate chain that Node's bundled root store doesn't trust, so Node's
 * global `fetch` (undici) fails the model download with
 * `unable to get local issuer certificate` — which surfaces as "fetch failed"
 * and leaves the index with 0% vector coverage. `curl` still works because it
 * reads the OS trust store.
 *
 * If the user hasn't already pointed Node at a CA bundle via
 * NODE_EXTRA_CA_CERTS, additively merge the system CA bundle into Node's
 * default trust store before Transformers.js runs its download. This is
 * additive (system roots are kept), best-effort, and a no-op on Node versions
 * without `tls.setDefaultCACertificates` (added in Node 22+).
 */
function ensureCaTrust(): void {
  if (_caTrustEnsured) return;
  _caTrustEnsured = true;

  // Respect an explicit operator override.
  if (process.env.NODE_EXTRA_CA_CERTS) return;

  try {
    if (typeof (tls as any).setDefaultCACertificates !== "function") return;

    const candidates = [
      process.env.SSL_CERT_FILE,
      "/etc/ssl/certs/ca-certificates.crt",   // Debian/Ubuntu/Alpine
      "/etc/pki/tls/certs/ca-bundle.crt",     // RHEL/CentOS/Fedora
      "/etc/ssl/ca-bundle.pem",               // openSUSE
      "/etc/ssl/cert.pem",                    // macOS/BSD
    ].filter((p): p is string => !!p);

    let bundle: string | null = null;
    for (const p of candidates) {
      try {
        if (existsSync(p)) { bundle = readFileSync(p, "utf-8"); break; }
      } catch { /* unreadable — try next */ }
    }
    if (!bundle) return;

    const extra = bundle
      .split(/(?=-----BEGIN CERTIFICATE-----)/)
      .filter(s => s.includes("BEGIN CERTIFICATE"));
    if (!extra.length) return;

    // Keep Node's built-in roots and append the system bundle.
    const merged = [...tls.rootCertificates, ...extra];
    (tls as any).setDefaultCACertificates(merged);
  } catch (err) {
    process.stderr.write(`[rag] CA trust setup skipped: ${(err as Error).message}\n`);
  }
}

async function getEmbedder() {
  if (_pipeline) return _pipeline;
  ensureCaTrust();
  const { pipeline } = await import("@xenova/transformers");
  _pipeline = await pipeline("feature-extraction", EMBEDDING_MODEL);
  return _pipeline;
}

export async function embed(text: string): Promise<number[]> {
  const embedder = await getEmbedder();
  const output = await embedder(text, { pooling: "mean", normalize: true });
  return Array.from(output.data as Float32Array);
}

/**
 * Yield to the event loop so the TUI can render progress updates.
 * ONNX inference is synchronous from the event loop's perspective;
 * without this, the UI freezes during embedding.
 */
const yield_ = () => new Promise<void>(r => setTimeout(r, 0));

/** Upper bound on texts per forward pass; the memory budget below usually bites first. */
export const BATCH_SIZE = 64;

/** Fallback sequence length when the tokenizer can't be consulted (all-MiniLM-L6-v2 truncates at 512). */
const MAX_SEQ_LEN = 512;

/**
 * Peak RSS of one forward pass is dominated by the attention score matrices,
 * which scale with `batchSize x seqLen^2` - seqLen being the *longest* text in
 * the batch, since transformers.js pads the whole batch to it.
 *
 * A flat batch of 64 full-length chunks costs ~4.3 GB on top of the ~290 MB
 * baseline, which OOM-kills the host agent on a small machine - and buys
 * nothing: on a CPU-bound box throughput is flat at ~158 ms/chunk from batch 1
 * to batch 64. So cap the batch by a memory budget instead of a count, which
 * still lets short texts (code, prose) batch wide.
 */
const BYTES_PER_ITEM_TOKEN_SQ = 250;

/** Memory a single forward pass may use, above baseline. */
const EMBED_BUDGET_BYTES = Math.max(32, Number(process.env.PI_RAG_EMBED_BUDGET_MB) || 384) * 1024 * 1024;

/** How many texts of `tokens` length fit one forward pass within the budget. Exported for tests. */
export function batchLimitFor(tokens: number): number {
  const fit = Math.floor(EMBED_BUDGET_BYTES / (BYTES_PER_ITEM_TOKEN_SQ * tokens * tokens));
  return Math.min(BATCH_SIZE, Math.max(1, fit));
}

/** Token count after the pipeline's own truncation, so the budget sees real sequence lengths. */
function tokenLength(embedder: any, text: string): number {
  const cap = embedder?.tokenizer?.model_max_length || MAX_SEQ_LEN;
  try {
    return Math.min(cap, embedder.tokenizer.encode(text).length) || 1;
  } catch {
    return cap; // Unknown length - assume worst case, i.e. the smallest batch.
  }
}

/**
 * Embed `texts` using true batched ONNX inference.
 *
 * Batches are assembled greedily up to `EMBED_BUDGET_BYTES` (and never more
 * than `BATCH_SIZE` texts), so one long chunk shrinks its batch instead of
 * blowing up the process. The output Tensor has dims [batchSize, VECTOR_DIM];
 * we slice it into per-text arrays.
 *
 * `onProgress` is fired after each batch with the cumulative count so the TUI
 * can render a smooth progress bar (same contract as before).
 */
export async function embedBatch(
  texts: string[],
  onProgress?: (i: number, total: number) => void,
): Promise<number[][]> {
  if (texts.length === 0) return [];
  const embedder = await getEmbedder();
  const results: number[][] = new Array(texts.length);
  const lengths = texts.map(t => tokenLength(embedder, t));

  let start = 0;
  while (start < texts.length) {
    // Grow the batch while the longest member still leaves room for one more.
    let end = start + 1;
    let seqLen = lengths[start];
    while (end < texts.length) {
      const grown = Math.max(seqLen, lengths[end]);
      if (end + 1 - start > batchLimitFor(grown)) break;
      seqLen = grown;
      end++;
    }

    const batch = texts.slice(start, end);
    const output = await embedder(batch, { pooling: "mean", normalize: true });
    const flat = output.data as Float32Array;
    const dim = flat.length / batch.length; // should equal VECTOR_DIM (384)

    for (let j = 0; j < batch.length; j++) {
      results[start + j] = Array.from(flat.subarray(j * dim, (j + 1) * dim));
    }

    start = end;
    onProgress?.(start, texts.length);
    // Yield after each batch so the TUI can re-render before the next pass.
    await yield_();
  }

  return results;
}
