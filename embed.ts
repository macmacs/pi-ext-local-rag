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

/** Default batch size for a single ONNX forward pass. */
export const BATCH_SIZE = 64;

/**
 * Embed `texts` using true batched ONNX inference.
 *
 * The model is called once per batch of up to `BATCH_SIZE` texts rather than
 * once per text, giving a ~BATCH_SIZE× speedup on CPU.  The output Tensor has
 * dims [batchSize, VECTOR_DIM]; we slice it into per-text arrays.
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

  for (let start = 0; start < texts.length; start += BATCH_SIZE) {
    const batch = texts.slice(start, start + BATCH_SIZE);
    // Pass the whole batch in a single forward pass — the model returns a
    // Tensor with dims [batchSize, VECTOR_DIM].
    const output = await embedder(batch, { pooling: "mean", normalize: true });
    const flat = output.data as Float32Array;
    const dim = flat.length / batch.length; // should equal VECTOR_DIM (384)

    for (let j = 0; j < batch.length; j++) {
      results[start + j] = Array.from(flat.subarray(j * dim, (j + 1) * dim));
    }

    onProgress?.(Math.min(start + batch.length, texts.length), texts.length);
    // Yield after each batch so the TUI can re-render before the next pass.
    await yield_();
  }

  return results;
}
