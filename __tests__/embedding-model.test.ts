/**
 * Covers making the embedding model configurable.
 *
 * The two things that silently corrupt retrieval if they regress: E5 models
 * need asymmetric query/passage prefixes, and vectors from a different model
 * are not comparable to the ones already stored.
 *
 * Set SKIP_EMBEDDING_TESTS=1 to skip the test that loads the real ONNX model.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { prefixFor, activeEmbeddingModel } from "../embed.ts";
import { EMBEDDING_MODEL } from "../constants.ts";

const skip = process.env.SKIP_EMBEDDING_TESTS === "1";

let dir: string;
let prevRagDir: string | undefined;

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), "rag-model-"));
  prevRagDir = process.env.PI_RAG_DIR;
  process.env.PI_RAG_DIR = dir;
});

afterEach(async () => {
  const { closeDbConn } = await import("../db.ts");
  closeDbConn();
  if (prevRagDir === undefined) delete process.env.PI_RAG_DIR;
  else process.env.PI_RAG_DIR = prevRagDir;
  rmSync(dir, { recursive: true, force: true });
});

function writeConfig(cfg: Record<string, unknown>) {
  writeFileSync(join(dir, "config.json"), JSON.stringify(cfg));
}

describe("prefixFor", () => {
  it("tags both sides for E5 models", () => {
    expect(prefixFor("Xenova/multilingual-e5-small", "query")).toBe("query: ");
    expect(prefixFor("Xenova/multilingual-e5-small", "passage")).toBe("passage: ");
  });

  it("leaves non-E5 models untouched", () => {
    for (const kind of ["query", "passage"] as const) {
      expect(prefixFor("Xenova/all-MiniLM-L6-v2", kind)).toBe("");
      expect(prefixFor("Xenova/paraphrase-multilingual-MiniLM-L12-v2", kind)).toBe("");
    }
  });
});

describe("activeEmbeddingModel", () => {
  it("falls back to the shipped default when unconfigured", () => {
    expect(activeEmbeddingModel()).toBe(EMBEDDING_MODEL);
  });

  it("honours the configured model", () => {
    writeConfig({ embeddingModel: "Xenova/paraphrase-multilingual-MiniLM-L12-v2" });
    expect(activeEmbeddingModel()).toBe("Xenova/paraphrase-multilingual-MiniLM-L12-v2");
  });

  it("falls back when the config sets an empty model", () => {
    writeConfig({ embeddingModel: "" });
    expect(activeEmbeddingModel()).toBe(EMBEDDING_MODEL);
  });
});

describe("indexFiles model-change handling", () => {
  it.skipIf(skip)("re-embeds everything when the stored model differs", async () => {
    const { indexFiles } = await import("../indexing.ts");
    const { getDbConn } = await import("../db.ts");
    const repo = await import("../repository.ts");

    const file = join(dir, "note.md");
    writeFileSync(file, "Hausratversicherung Beitrag und Versicherungssumme laut Vertrag.\n".repeat(4));

    const db = getDbConn();
    const first = await indexFiles([file], undefined, db);
    expect(first.chunks).toBeGreaterThan(0);
    expect(repo.getMetadata(db, repo.MetadataKey.EmbeddingModel)).toBe(EMBEDDING_MODEL);

    // Unchanged file + unchanged model: the hash check should skip it.
    const second = await indexFiles([file], undefined, db);
    expect(second.skipped).toBe(1);
    expect(second.chunks).toBe(0);

    // Pretend the index was built by another model. The stored vectors are now
    // meaningless, so the skip must not apply.
    repo.setMetadata(db, repo.MetadataKey.EmbeddingModel, "some/other-model");
    const third = await indexFiles([file], undefined, db);
    expect(third.skipped).toBe(0);
    expect(third.chunks).toBeGreaterThan(0);
    expect(repo.getMetadata(db, repo.MetadataKey.EmbeddingModel)).toBe(EMBEDDING_MODEL);
  }, 180_000);
});
