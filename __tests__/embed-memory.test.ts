/**
 * Guards the embedding step's memory ceiling.
 *
 * A flat batch of 64 texts padded to the model's 512-token limit allocates
 * ~4.6 GB of attention scratch, which OOM-kills the host agent on a small
 * machine. embedBatch sizes batches by a memory budget instead; these tests
 * pin both the arithmetic and the real peak RSS.
 *
 * Set SKIP_EMBEDDING_TESTS=1 to skip the runtime tests (they load the ONNX model).
 */
import { describe, it, expect } from "vitest";
import { spawnSync } from "node:child_process";
import { resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { batchLimitFor, embedBatch, embed, BATCH_SIZE } from "../embed.ts";

const skip = process.env.SKIP_EMBEDDING_TESTS === "1";
const embedPath = resolve(fileURLToPath(import.meta.url), "../../embed.ts");

const LINE = "Versicherungsschein Nummer 09/65/2688761 Beitrag monatlich Versicherungssumme ";
/** ~1500 tokens of German prose, i.e. truncated to the model's 512-token limit. */
const LONG = Array.from({ length: 50 }, (_, i) => LINE + i).join("\n");
const SHORT = "Hausratversicherung Beitrag 12,50 EUR monatlich";

describe("batchLimitFor", () => {
  it("shrinks the batch quadratically as texts get longer", () => {
    const full = batchLimitFor(512);
    expect(full).toBeGreaterThanOrEqual(1);
    // Halving the sequence length quarters the cost, so ~4x the texts fit.
    expect(batchLimitFor(256)).toBeGreaterThan(full * 3);
    expect(batchLimitFor(512)).toBeLessThan(batchLimitFor(128));
  });

  it("never exceeds BATCH_SIZE and never drops below 1", () => {
    expect(batchLimitFor(1)).toBe(BATCH_SIZE);
    // Absurdly long input: still has to make progress rather than stall.
    expect(batchLimitFor(100_000)).toBe(1);
  });
});

describe("embedBatch", () => {
  it.skipIf(skip)("returns one vector per text for a mix of long and short inputs", async () => {
    const texts = Array.from({ length: 24 }, (_, i) => (i % 4 === 0 ? LONG + i : SHORT + " " + i));
    const vecs = await embedBatch(texts);
    expect(vecs).toHaveLength(texts.length);
    for (const v of vecs) expect(v).toHaveLength(384);

    // Splitting a batch must not move a vector: batched and solo agree.
    // (They are not bit-identical - padding shifts the quantized model
    // slightly - which was equally true of the old fixed-64 batching.)
    for (const i of [0, 1, texts.length - 1]) {
      const solo = await embed(texts[i]);
      const cos = solo.reduce((s, x, k) => s + x * vecs[i][k], 0);
      expect(cos).toBeGreaterThan(0.95);
    }
  }, 180_000);

  it.skipIf(skip)("embeds a full batch of long chunks without a memory blow-up", () => {
    // Peak RSS is a whole-process property, so measure it in a child.
    const dir = mkdtempSync(join(tmpdir(), "rag-embed-mem-"));
    try {
      const script = join(dir, "child.mjs");
      writeFileSync(script, `
        import { embedBatch } from ${JSON.stringify(pathToFileURL(embedPath).href)};
        const line = ${JSON.stringify(LINE)};
        const long = Array.from({ length: 50 }, (_, i) => line + i).join("\\n");
        let peak = 0;
        setInterval(() => { peak = Math.max(peak, process.memoryUsage().rss); }, 50).unref();
        await embedBatch(Array.from({ length: 128 }, (_, i) => long + i));
        process.stdout.write("PEAK_MB=" + Math.round(peak / 1e6));
      `, "utf-8");

      const r = spawnSync(process.execPath, [script], { encoding: "utf-8", timeout: 300_000 });
      if (/ERR_UNKNOWN_FILE_EXTENSION|Unknown file extension/.test(r.stderr)) {
        return; // Node too old to import TypeScript directly; nothing to assert.
      }
      const peak = Number(/PEAK_MB=(\d+)/.exec(r.stdout ?? "")?.[1]);
      expect(peak).toBeGreaterThan(0);
      // ~700 MB with budgeted batches, ~4650 MB with flat batches of 64.
      expect(peak).toBeLessThan(1500);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  }, 300_000);
});
