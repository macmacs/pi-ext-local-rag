# pi-local-rag

Local hybrid RAG pipeline for the [Pi coding agent](https://github.com/badlogic/pi-mono). Index your local files and search them with BM25 + vector similarity — **zero cloud dependency, works fully offline**.

## Features

- **Hybrid BM25 + vector search** — SQLite FTS5 for keyword scoring, [`sqlite-vec`](https://github.com/asg017/sqlite-vec) for 384-dim cosine NN, blended at retrieval time
- **Local ONNX embeddings** — `Xenova/multilingual-e5-small` via Transformers.js (~113 MB model, 100+ languages, runs fully offline after first download)
- **Many file formats** — text, source code, Markdown, JSON, YAML, plus PDF (with optional OCR fallback for scanned docs), DOCX, HTML (auto-converted to Markdown)
- **Per-project storage** — walks up from cwd looking for `.pi/rag/`; falls back to `~/.pi/rag/` global store
- **Tracked paths + exclude patterns** — `/rag index <path>` remembers what to keep current; gitignore-style `/rag exclude` for `dist/`, `*.log`, etc.
- **Auto-refresh** — stale index (>24 h) silently refreshed before the next agent turn; manual `/rag refresh` for on-demand incremental updates
- **Auto-injection** — relevant chunks appended after the user prompt before every agent turn (KV-cache friendly)
- **3 AI tools** — `rag_index`, `rag_query`, `rag_status` for the agent to call directly

## Install

```bash
pi install npm:pi-local-rag
```

Or via git:

```bash
pi install git:github.com/vahidkowsari/pi-local-rag
```

Optional: install `pdftoppm` (poppler) + `tesseract` with traineddata for the languages your documents are in (see `ocrLanguages`) to enable OCR fallback for image-only PDFs.

```bash
# macOS
brew install poppler tesseract tesseract-lang

# Debian/Ubuntu
apt install poppler-utils tesseract-ocr tesseract-ocr-eng   # add tesseract-ocr-deu etc. as needed
```

The OCR fallback is silent when these tools aren't installed (logs one stderr hint on the first image-only PDF encountered).

## Commands

| Command | Description |
|---|---|
| `/rag index <path>` | Index a file or directory (chunks → embeds → stores); adds the path to tracked paths |
| `/rag search <query>` | Hybrid BM25 + vector search over the index |
| `/rag find <glob>` | List indexed files matching a glob (e.g. `*.ts`, `src/*`) |
| `/rag status` | Show index stats, active config, tracked paths, exclude patterns, storage scope |
| `/rag rebuild [--force]` | Re-walk tracked paths and re-embed all files. `--force` wipes the DB and bypasses the hash-cache check |
| `/rag refresh` | Incremental refresh — only new/changed files (same code path as the 24 h auto-refresh) |
| `/rag clear` | Wipe the entire index (tracked paths are preserved) |
| `/rag exclude <pattern>` | Add a gitignore-style exclude pattern; `/rag exclude -<pattern>` to remove; no arg to list |
| `/rag ext list \| add <.ext> \| remove <.ext> \| reset` | Manage the indexable file-extension allowlist |
| `/rag on` \| `off` | Toggle auto-injection |
| `/rag help` | Show all subcommands |

Tab-completion is available for every subcommand.

## Example session

```text
$ /rag index ~/code/my-app
Found 412 files to index
Indexing  ████████████████████████  100%
file:    src/server/handlers/payments.ts
done:    412 embedded · 0 unchanged
✅ Indexed 412 files (1,847 chunks) · 0 unchanged · 38.4s · tracking 1 path(s) · project store

$ /rag status
🔍 pi-local-rag

  Files indexed:    412
  Chunks:           1847
  Vectors:          1847  (100% coverage)
  Total tokens:     438,219
  Embedding model:  Xenova/multilingual-e5-small
  Last build:       2026-05-26T20:14:03.221Z
  Storage:          /Users/you/code/my-app/.pi/rag (project)

  RAG injection:    enabled  topK=5  threshold=0.1  alpha=0.4

  File types:
    .ts    231
    .tsx   118
    .md     34
    .json   18
    .yaml    7

  Tracked paths:
    /Users/you/code/my-app

  Exclude patterns:
    (none — add with /rag exclude <pattern>)

$ /rag search "stripe webhook signature verification"
🔍 4 results for "stripe webhook signature verification"  hybrid BM25+vector

payments.ts:142-187  score=0.92
  export async function verifyStripeWebhook(req: Request) {
    const sig = req.headers.get("stripe-signature");
    if (!sig) throw new Error("missing signature header");

webhooks.md:1-23  score=0.71
  # Webhook signing
  All inbound webhooks are verified against the shared secret stored in
  STRIPE_WEBHOOK_SECRET. Stripe signs each request with a t= timestamp...

$ /rag exclude dist/
✅ Added exclude: dist/ · 1 pattern(s) total. Run /rag rebuild to re-apply.

$ /rag find *.html
🔍 12 indexed files matching "*.html"
src/docs/install.html
src/docs/quickstart.html
...

$ /rag rebuild
Scanning tracked paths...
Discovered 3 new files
Rebuilding 415 files...
Rebuilding  ████████████████████████  100%
Embedding   ████████████████████████  100%  1847/1847 chunks
✅ Rebuilt: 3 re-indexed · 412 unchanged · 0 deleted · 1850 chunks · 14.2s
```

> Output above is approximate — actual colors, spacing, and widget layout depend on your terminal theme and the Pi agent's UI.

## AI Tools

The extension registers three tools the agent can call directly:

- **`rag_index`** — Index a path into the pipeline (also adds it to tracked paths)
- **`rag_query`** — Hybrid BM25 + vector search; returns file paths + line numbers + previews + scores
- **`rag_status`** — Index stats, RAG config, storage path + scope

## How It Works

1. **Index** — files are chunked (~50 lines each, broken at blank lines where possible), embedded with `Xenova/multilingual-e5-small` (384-dim), and stored in SQLite. PDF/DOCX go through `pdf-parse`/`mammoth`; HTML is converted to Markdown via `turndown`; scanned PDFs fall back to OCR (`pdftoppm` + `tesseract`) when the system tools are installed.
2. **Search** — FTS5 `bm25()` + `sqlite-vec` cosine NN, normalized and blended: `alpha × BM25 + (1-alpha) × cosine` (default `alpha=0.4`). Filename matches on the first query term get a 1.5× boost.
3. **Auto-inject** — before every agent turn, the user's prompt is searched against the index and relevant chunks are appended after the prompt as a hidden `customType: "rag"` message (KV-cache friendly — the system prompt is unchanged across turns).
4. **Auto-refresh** — if the index is older than 24 h, the `before_agent_start` hook re-walks tracked paths and re-indexes new/changed files in the background. Throttled to one stale check per hour.

## Storage

Index data lives in `rag.db` (SQLite, WAL mode, with FTS5 + sqlite-vec extensions loaded). Three resolution rules:

1. **`$PI_RAG_DIR`** — explicit override, wins over everything
2. **Walk-up** from `process.cwd()` looking for an existing `.pi/rag/` directory (stopping before `$HOME`)
3. **Global** fallback at `~/.pi/rag/`

`/rag index <path>` creates a project store at the current cwd if no parent store is in scope. `/rag status` shows the resolved path and whether it's project-local or global.

Legacy `~/.pi/lens/` directories are renamed to `~/.pi/rag/` on first run; legacy `index.json` files are migrated into `rag.db` and removed.

## Configuration

Auto-injection is on by default. Config lives in `<ragDir>/config.json`:

| Setting | Default | Description |
|---|---|---|
| `ragEnabled` | `true` | Auto-inject context before each turn |
| `ragTopK` | `5` | Max chunks to inject |
| `ragScoreThreshold` | `0.35` | Min hybrid score to auto-inject a chunk. Higher = stricter (fewer, more relevant hits); lower lets weaker/noisier matches through. |
| `ragAlpha` | `0.4` | BM25/vector blend (0 = pure vector, 1 = pure BM25). Raise it when the vector half is weak on your corpus - with the old English-only default model, a German archive needed `0.7` to reach the quality `0.5` gives on the current model. |
| `extraExtensions` | `[]` | Extra file extensions to index beyond the defaults |
| `excludeExtensions` | `[]` | Default extensions to skip |
| `trackedPaths` | `[]` | Absolute paths that `/rag rebuild`/`refresh` re-walk |
| `excludePatterns` | lock files, `*.min.js/css`, `*.map` | Gitignore-style patterns skipped when walking. Seeded with machine-generated noise (e.g. `package-lock.json`) so it never pollutes search. Add more with `/rag exclude <pattern>`. |
| `ocrLanguages` | `["eng"]` | Tesseract languages for scanned PDFs, best match first (e.g. `["deu", "eng"]`). Getting this wrong is expensive: German scans OCR'd as English lose every umlaut. Needs the matching traineddata installed (`apt install tesseract-ocr-deu`); entries without it are dropped. |
| `embeddingModel` | `Xenova/multilingual-e5-small` | Transformers.js feature-extraction model. Changing it invalidates every stored vector, so the next index run wipes and re-embeds automatically. See below. |

### Choosing an embedding model

The default was `Xenova/all-MiniLM-L6-v2` up to 0.4.1. That model is English-only and its tokenizer strips accents, so `Kündigungsfrist` became `kun|##di|##gun|##gs|##fr|##ist` - six fragments with the umlaut gone. On a non-English corpus the vector half of hybrid search contributed little and `ragAlpha` had to be pushed up to compensate.

Measured on a 1719-chunk German document archive, 10 labelled lookups, pure-vector retrieval:

| model | dim | quantized | top-1 | top-3 | MRR |
|---|---|---|---|---|---|
| `Xenova/all-MiniLM-L6-v2` (old default) | 384 | 22 MB | 3/10 | 4/10 | 0.333 |
| `Xenova/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 113 MB | 2/10 | 3/10 | 0.267 |
| `Xenova/multilingual-e5-small` (default) | 384 | 113 MB | 5/10 | 9/10 | 0.683 |

Note that the multilingual *paraphrase* model scores worse than the English-only default despite understanding the language. Paraphrase models are trained for symmetric sentence-to-sentence similarity, whereas RAG is asymmetric - a short query against a long chunk. Pick a model trained for retrieval.

E5 models expect `query: ` / `passage: ` prefixes and lose most of their advantage without them; `embed()` and `embedBatch()` apply these automatically based on the model name, which is why they take a `query`/`passage` argument.

Only 384-dim models are drop-ins. `VECTOR_DIM` in `constants.ts` is baked into the `chunks_vec` table definition, so a wider model (e.g. `multilingual-e5-base`, 768) needs that constant changed *and* the existing `rag.db` deleted - the automatic re-embed cannot widen an existing table.

Environment overrides:

| Variable | Default | Description |
|---|---|---|
| `PI_RAG_DIR` / `PI_RAG_LEGACY_DIR` | auto-resolved | Where the index and config live |
| `PI_RAG_EMBED_BUDGET_MB` | `384` | Memory one embedding forward pass may use. Batch size is derived from it, so lower this on a memory-tight host and raise it on a large one. |

## Testing

```bash
npm test                          # full suite (downloads ~113 MB model on first run)
SKIP_EMBEDDING_TESTS=1 npm test   # skip the real-ONNX semantic tests
```

OCR end-to-end test is skipped when `tesseract` isn't installed.
