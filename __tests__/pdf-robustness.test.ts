import { describe, it, expect, beforeAll, afterAll } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { pathToFileURL, fileURLToPath } from "node:url";
import { tmpdir } from "node:os";
import { deflateSync } from "node:zlib";
import { spawnSync } from "node:child_process";
import { extractText, pdfjsSwallowedRejections, pdfParseFailures } from "../chunking.ts";

const chunkingPath = resolve(fileURLToPath(import.meta.url), "../../chunking.ts");

const PAGE_TEXT =
  "Vertragsspiegel Testdokument Hausratversicherung Beitrag Versicherungssumme " +
  "Selbstbeteiligung Laufzeit Kuendigungsfrist Versicherungsnehmer";

function assemble(objs: (string | Buffer)[], pad: number): Buffer {
  let header = "%PDF-1.4\n";
  if (pad > 0) header += `%${"P".repeat(pad)}\n`;
  const parts: Buffer[] = [Buffer.from(header, "binary")];
  const offsets: number[] = [];
  let len = header.length;
  for (const o of objs) {
    const b = Buffer.isBuffer(o) ? o : Buffer.from(o, "binary");
    offsets.push(len);
    parts.push(b);
    len += b.length;
  }
  let xref = `xref\n0 ${objs.length + 1}\n0000000000 65535 f \n`;
  for (const off of offsets) xref += `${String(off).padStart(10, "0")} 00000 n \n`;
  parts.push(Buffer.from(
    `${xref}trailer\n<< /Size ${objs.length + 1} /Root 1 0 R >>\nstartxref\n${len}\n%%EOF\n`, "binary"));
  return Buffer.concat(parts);
}

function streamObj(num: number, dict: string, data: Buffer): Buffer {
  return Buffer.concat([
    Buffer.from(`${num} 0 obj\n<< ${dict} /Length ${data.length} >>\nstream\n`, "binary"),
    data,
    Buffer.from("\nendstream\nendobj\n", "binary"),
  ]);
}

/** Flips bits through a deflate payload, leaving the PDF object structure intact. */
function mangle(data: Buffer): Buffer {
  const c = Buffer.from(data);
  for (let i = 4; i < c.length; i += 3) c[i] ^= 0xff;
  return c;
}

/**
 * Builds a real PDF. `pad` inflates it past Node's shared buffer pool: pdfjs
 * derives sub-streams from `bytes.buffer` and ignores `byteOffset`, so a small
 * pool-backed document is misparsed unless the caller hands over a standalone
 * Uint8Array. `corruptStream` leaves the structure valid but destroys the page
 * content's flate data, which is what makes pdfjs reject a detached promise.
 */
function makePdf({ pad = 70_000, corruptStream = false } = {}): Buffer {
  const content = deflateSync(Buffer.from(`BT /F1 12 Tf 72 700 Td (${PAGE_TEXT}) Tj ET\n`, "binary"));
  return assemble([
    "1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
    "2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
    "3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R " +
      "/Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n",
    streamObj(4, "/Filter /FlateDecode", corruptStream ? mangle(content) : content),
    "5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
  ], pad);
}

let dir: string;
beforeAll(() => { dir = mkdtempSync(join(tmpdir(), "rag-pdf-test-")); });
afterAll(() => { rmSync(dir, { recursive: true, force: true }); });

function write(name: string, buf: Buffer): string {
  const p = join(dir, name);
  writeFileSync(p, buf);
  return p;
}

describe("extractText on PDFs", () => {
  it("extracts text from a well-formed PDF", async () => {
    const { text, size } = await extractText(write("clean.pdf", makePdf()));
    expect(text).toContain("Vertragsspiegel Testdokument");
    expect(size).toBeGreaterThan(0);
  });

  it("extracts text from a PDF small enough to sit in Node's buffer pool", async () => {
    // Regression: handing pdfjs the Node Buffer failed this valid document with
    // "bad XRef entry", because pdfjs read objects out of a neighbouring buffer.
    // OCR would paper over that, so assert the parse itself succeeded.
    const before = pdfParseFailures;
    const { text } = await extractText(write("small.pdf", makePdf({ pad: 0 })));
    expect(pdfParseFailures).toBe(before);
    expect(text).toContain("Vertragsspiegel Testdokument");
  });

  it("resolves instead of throwing for structurally invalid PDFs", async () => {
    for (const [name, buf] of [
      ["empty.pdf", Buffer.alloc(0)],
      ["garbage.pdf", Buffer.from("this is definitely not a pdf")],
      ["truncated.pdf", makePdf().subarray(0, 4096)],
    ] as const) {
      const r = await extractText(write(name, buf));
      expect(typeof r.text).toBe("string");
      expect(r.hash).toHaveLength(12);
    }
  });

  it("traps the detached rejection a corrupt content stream provokes", async () => {
    const before = pdfjsSwallowedRejections;
    const r = await extractText(write("bad-stream.pdf", makePdf({ corruptStream: true })));
    expect(typeof r.text).toBe("string");
    // The rejection surfaces after pdf() has already resolved; give it a turn.
    await new Promise(res => setTimeout(res, 500));
    // If this stops incrementing, either the fixture no longer provokes pdfjs
    // or pdfjs stopped leaking - check which before touching the guard.
    expect(pdfjsSwallowedRejections).toBeGreaterThan(before);
  });

  // The guard cannot stop other unhandledRejection listeners, and its whole
  // effect is on a process where it is the only one - so the crash it prevents
  // is only observable from outside. Before the guard, `bad-stream.pdf` took
  // the host down with `FormatError: Bad encoding in flate stream`.
  it("does not take the host process down, and still lets host bugs crash it", () => {
    const pdfPath = write("child-bad-stream.pdf", makePdf({ corruptStream: true }));
    const script = write("child.mjs", Buffer.from(`
      import { extractText } from ${JSON.stringify(pathToFileURL(chunkingPath).href)};
      await extractText(${JSON.stringify(pdfPath)});
      await new Promise(r => setTimeout(r, 800));
      if (process.argv[2] === "with-host-bug") Promise.reject(new Error("unrelated app bug"));
      await new Promise(r => setTimeout(r, 800));
      process.stdout.write("SURVIVED");
    `, "utf-8"));

    const survives = spawnSync(process.execPath, [script], { encoding: "utf-8" });
    if (/ERR_UNKNOWN_FILE_EXTENSION|Unknown file extension/.test(survives.stderr)) {
      return; // Node too old to import TypeScript directly; nothing to assert.
    }
    expect(survives.stdout).toContain("SURVIVED");
    expect(survives.status).toBe(0);

    // Same run plus a genuine host bug: the guard must let that one through.
    const crashes = spawnSync(process.execPath, [script, "with-host-bug"], { encoding: "utf-8" });
    expect(crashes.status).not.toBe(0);
    expect(crashes.stderr).toContain("unrelated app bug");
  }, 30_000);
});
