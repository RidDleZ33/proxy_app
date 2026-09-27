# Context Compaction Proxy

Dynamic multi-pass context compaction proxy for long coding-agent sessions.

It sits in front of an OpenAI-compatible backend (llama-swap, llama.cpp server, vLLM, …) and/or native Ollama. On each chat request it measures the effective payload, and when that payload crosses a configurable fraction of the model context window it folds older turns into a structured persistent summary. Recent turns stay verbatim.

**Current version:** `v2.4.0-universal` (`__VERSION__` in `app.py`)

---

## What problem this solves

Coding agents replay the full tool-call history every turn. A few dozen `read_file` / `bash` results will fill a 128k window. This proxy:

1. Detects when the live transcript plus any stored summary would exceed `TRIGGER_RATIO` of the context window.
2. Summarizes only the *new overflow* since the last compact (delta, not the whole thread).
3. Forwards a rebuilt payload: synthetic system prompt (original system + summary + optional policy directive) + the recent untouched window.
4. Remembers the summary and an anchor hash in SQLite so the next turn does not re-summarize the same messages.

It is a transparent HTTP proxy. Clients keep talking OpenAI `/v1/chat/completions` or Ollama `/api/chat`. Compaction happens on the request body before the call is forwarded.

---

## Architecture

```
Client (Continue, Open WebUI, custom agent, …)
        |
        |  /v1/*   →  llama-swap / OpenAI-compatible backend
        |  /api/*  →  native Ollama
        v
   this proxy  (:11435 by default)
        |  compact calls (optional separate model)
        +-- SQLite chat_cache (summary + last_anchor per thread)
```

Routing is by **the URL the client used**, not by model name:

| Client hits | Backend |
|---|---|
| `/v1/chat/completions`, `/v1/transform`, any other `/v1/*` | `LLAMASWAP_URL` |
| `/api/chat`, `/api/generate`, any other `/api/*` | `OLLAMA_URL` |

`backend_from_path()` encodes that rule. `/v1/transform` is treated as llama-swap and returns compacted messages *without* forwarding (Open WebUI filter inlet).

---

## Request flow

Implemented in `process_and_tier_messages()`.

1. **Thread key.** Prefer `chat_id` / `conversation_id` / `session_id` on the body or `body.metadata`. Otherwise hash the first non-empty user message. That key is the SQLite primary key.
2. **Load state.** `summary` + `last_anchor` for that key, or empty.
3. **Thresholds.** From `num_ctx` on the body (`num_ctx`, `options.num_ctx`, or Open WebUI `params` / `model_info` scrape). Fallback: `DEFAULT_MODEL_CTX_TOKENS`. All budgets are `num_ctx * CHARS_PER_TOKEN * ratio`.
4. **Stable hashes.** Each message gets `_proxy_hash` from original `role` + `content` + `tool_call_id`/`name`/`tool_calls` *before* any rewrite. Anchors stay valid across truncate/banner edits. The field is stripped before the payload is forwarded.
5. **Tool mutations (mode-dependent).** See [Tool output mutations](#tool-output-mutations).
6. **Delta slice.** If `last_anchor` matches a hash in the current conversation, everything up to and including that message is treated as already summarized. Only the tail is “uncompacted.” If the anchor is missing, the whole conversation is uncompacted and a warning is logged; the previous summary is still used.
7. **Trigger math.** `effective = len(summary) + chars(uncompacted) + chars(system)`. Compared against `TRIGGER_RATIO * num_ctx * CHARS_PER_TOKEN`. This is a character heuristic, not a real tokenizer, and is **not** sent to the backend as tokenizer config.
8. **On trigger.** Walk backward through the uncompacted tail until `ACTIVE_RATIO` of the window is reserved. `snap_active_index()` then moves the cut to a user turn (never between an assistant `tool_calls` message and its `role=tool` results). The prefix of that tail is overflow; the suffix stays verbatim.
9. **Multi-pass compact.** Overflow is split into bites of `CHUNK_RATIO` and each bite is merged into the running summary by `SUMMARY_MODEL` (see [Summarizer](#summarizer)). On success the new summary and the hash of the last *original* overflow message are written in one `BEGIN IMMEDIATE` transaction. On failure the previous anchor is left in place and the full uncompacted tail is forwarded so history is not dropped.
10. **Rebuild.**
    - System message = original system (or a generic default) + `HEADER_SUMMARY` + summary.
    - Optional `SYSTEM_DIRECTIVE` appended to that same system message when `INJECT_SYSTEM_DIRECTIVE=true`.
    - First active message gets `HEADER_ACTIVE` prepended if a summary exists and the banner is not already there.
    - If the active window has no non-empty user message, the last user message in the original conversation is prepended (Ollama/llama.cpp reject a prompt with no user turn).
11. **Forward.** `/v1/chat/completions` and `/api/*` stream the upstream response. `/v1/transform` returns `{"messages": [...]}` only.

Passthrough routes (`/v1/<path>`, `/api/<path>`) do not compact. Use them for `/v1/models`, `/api/tags`, embeddings, etc.

---

## Tool output mutations

Two rewrites can change `role=tool` messages:

- **Truncate** (`sanitize_tool_outputs`). If a tool body exceeds `TOOL_TRUNCATE_RATIO * num_ctx * CHARS_PER_TOKEN`, keep the first and last half of that budget and insert a middle marker with the removed character count.
- **Dedupe** (`deduplicate_tool_outputs`). Walk the list from the end; the newest copy of a given tool-body MD5 is kept, older copies become a one-line notice.

Those rewrites save tokens. They also change bytes that already sat in the llama.cpp prefix cache. Dedupe is the worse offender: because it walks from the tail, a *new* duplicate on this turn rewrites an *earlier* tool message, so the prefix is no longer a byte-for-byte match and the cache rebuilds.

`TOOL_OUTPUT_MUTATIONS` chooses when those two rewrites run:

| Value | Forwarded chat payload | Compact / summarizer input |
|---|---|---|
| `compact_only` **(default)** | Original tool text on ordinary turns. After a successful compact, truncate+dedupe the **active window only**. | Truncate+dedupe a copy of the overflow before it is sent to the summarizer. |
| `always` | Truncate+dedupe every request, before the trigger check. | Same (already rewritten). |
| `never` | Never rewrite tool results. | Overflow is sent as-is. Individual tool bodies are still clipped to 2000 characters *inside* the summarizer prompt builder so a compact pass cannot blow its own context. |

Accepted aliases: `true` / `on` / `1` / `yes` / `all` → `always`; `false` / `off` / `0` / `no` / `none` → `never`; anything else (including empty) → `compact_only`.

**When to use which**

- `compact_only` — default for llama.cpp / llama-swap prefix cache. Long tool-call storms no longer invalidate KV on the next user message. You pay the full tool-token cost until a compact fires; after it fires the prefix was going to rebuild anyway.
- `always` — maximum token savings on every turn, and a safety cap on a single huge tool result even when the thread is still under the trigger. Expect cache rebuilds after duplicate tools or when a result newly crosses `TOOL_TRUNCATE_RATIO`.
- `never` — raw tool text always. Only the summarizer’s internal 2000-char clip still applies.

Hashes used as anchors are computed on the **original** message, so switching this flag does not orphan existing `last_anchor` rows.

---

## Summarizer

`resolve_compact_target(chat_backend, chat_model)`:

- `SUMMARY_MODEL` set to a real name (anything other than empty / `auto`) → use that model on `SUMMARY_BACKEND` (`ollama` or `llamaswap`). `SUMMARY_URL`, if set, overrides the backend base URL for compact calls only.
- `SUMMARY_MODEL=auto` or empty → compact with the **live chat model on the same backend the client hit**. `/v1` + auto → llama-swap + that request’s model. `/api` + auto → Ollama + that request’s model.

Warning: `auto` on a 70B coder blocks the user’s turn for the entire compact. A small dedicated model (`qwen2.5:3b` on Ollama is the example default) keeps the big coder loaded.

Each compact pass asks for this structure:

1. Core user intent and goals
2. Key technical facts and discoveries
3. Completed actions and diagnostics
4. Abandoned approaches and failures
5. Current pending task state

`<think>…</think>` tags are stripped from summarizer output. If a pass returns empty or HTTP-errors, compaction is aborted and the previous summary/anchor are kept.

After the last successful pass (or on failure, when keep-alive is `0`), `unload_summary_model()` runs if `UNLOAD_SUMMARY_AFTER_COMPACT` is true or `SUMMARY_KEEP_ALIVE` is `0`/`0s`/`0m`:

- Ollama: `POST /api/generate` with `keep_alive: 0`.
- llama-swap: tries `POST {root}/unload`, `/v1/unload`, and `/unload/{model}`.

`COMPACT_TIMEOUT` is the HTTP socket timeout per compact call (seconds). `0` means no timeout. It is not model residency.

---

## Persistence

SQLite file: `CACHE_DB` if set, otherwise `proxy_cache.db` in the same directory as `app.py`. WAL mode, `busy_timeout=5000`.

```sql
CREATE TABLE chat_cache (
    root_hash   TEXT PRIMARY KEY,
    summary     TEXT,
    last_anchor TEXT,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

Writes use `BEGIN IMMEDIATE` so two in-flight turns cannot split summary vs anchor. After each write, rows older than the `MAX_CHATS` most-recently-updated keys are deleted.

Put the DB on the same mount as `app.py` so it survives container recreation.

---

## HTTP API

The process binds `PROXY_HOST`:`PROXY_PORT` (default `0.0.0.0:11435`).

| Method | Path | Behavior |
|---|---|---|
| `GET` | `/version` | JSON: version, summary model/backend, keep-alive, cache path, directive flag, `chars_per_token`, `tool_output_mutations`, `tool_truncate_ratio`. |
| `POST` | `/v1/chat/completions` | Compact `messages`, stream to llama-swap. |
| `POST` | `/v1/transform` | Open WebUI filter: compact and return `{"messages": …}` only. Model/ctx scraped from `base_model_id` / `model_info` / `params`. |
| `*` | `/v1/<path>` | Transparent proxy to llama-swap `/v1/<path>`. |
| `POST` | `/api/chat` | Compact Ollama chat (or promote `prompt`+`system` to messages), stream to Ollama. Writes compacted text back onto `messages` and `system`. |
| `POST` | `/api/generate` | Same promotion/compaction, then Ollama generate. |
| `*` | `/api/<path>` | Transparent proxy to Ollama `/api/<path>`. |

Client base URLs:

- OpenAI-compatible (Continue OpenAI provider, Open WebUI OpenAI mode): `http://<proxy-host>:11435/v1`
- Native Ollama (Continue Ollama provider, Open WebUI Ollama mode): `http://<proxy-host>:11435`

---

## Configuration

Copy `example.env` (or `.env.example` — they are kept identical) to `.env` in the **same directory as `app.py`**. The process loads that file at import time. Existing OS / Compose environment variables win; `.env` only fills keys that are not already set. No `python-dotenv` dependency.

### Listen

| Variable | Default | Meaning |
|---|---|---|
| `PROXY_HOST` | `0.0.0.0` | Bind address. |
| `PROXY_PORT` | `11435` | Bind port. |

### Backends

| Variable | Default | Meaning |
|---|---|---|
| `LLAMASWAP_URL` | `http://arceus:9931/v1` | OpenAI-compatible upstream. With or without trailing `/v1` is fine. |
| `OLLAMA_URL` | `http://ollama:11434` | Native Ollama upstream. |

### Context math

Used only to decide *when* to compact.

| Variable | Default | Meaning |
|---|---|---|
| `DEFAULT_MODEL_CTX_TOKENS` | `131072` | Fallback when the client does not send `num_ctx`. |
| `CHARS_PER_TOKEN` | `2.55` | Heuristic calibrated to llama-swap infill (~1% error on code/tool dumps). Do not set this to `4` unless you measured that tokenizer; `4` fires compact far too late. |
| `TRIGGER_RATIO` | `0.80` | Compact when effective payload ≥ this fraction of the window. |
| `ACTIVE_RATIO` | `0.45` | Keep this fraction of the window as verbatim recent history. |
| `SUMMARY_RATIO` | `0.12` | Target size of the persistent summary. Floor 1000 characters. |
| `CHUNK_RATIO` | `0.25` | Max size of one multi-pass compact bite. |
| `TOOL_TRUNCATE_RATIO` | `0.05` | Max size of one tool body before head/tail trim (when mutations run). |

### Tool mutations

| Variable | Default | Meaning |
|---|---|---|
| `TOOL_OUTPUT_MUTATIONS` | `compact_only` | `compact_only` \| `always` \| `never`. See [Tool output mutations](#tool-output-mutations). |

### Summarizer

| Variable | Default | Meaning |
|---|---|---|
| `SUMMARY_MODEL` | `qwen2.5:3b` | Named model, or `auto` / empty to follow the live chat model. |
| `SUMMARY_BACKEND` | `ollama` | `ollama` or `llamaswap`. Ignored when model is `auto`. |
| `SUMMARY_URL` | empty | Optional override base URL for compact calls only. |
| `COMPACT_TIMEOUT` | `900` | HTTP timeout per compact call, seconds. `0` = no socket timeout. |
| `SUMMARY_KEEP_ALIVE` | `0` | Forwarded as Ollama `keep_alive` / llama-swap `keep_alive`. `0` unloads as soon as the call finishes. Empty = backend default. |
| `UNLOAD_SUMMARY_AFTER_COMPACT` | `true` | After the last pass, also hit an explicit unload endpoint so the primary model can reclaim VRAM before the user completion is forwarded. |

### Cache

| Variable | Default | Meaning |
|---|---|---|
| `CACHE_DB` | empty → `./proxy_cache.db` next to `app.py` | SQLite path. |
| `MAX_CHATS` | `50` | Keep this many most-recently-updated threads. |

### Optional policy injection

Off by default. Keep it off for Continue / generic Open WebUI unless you actually want the Engraphis (or custom) block in the system prompt.

| Variable | Default | Meaning |
|---|---|---|
| `INJECT_SYSTEM_DIRECTIVE` | `false` | Append the directive to the synthetic system prompt. Same prefix every non-compact turn so llama.cpp KV can extend. User messages are not touched. |
| `SYSTEM_DIRECTIVE` | empty | Custom block. If injection is on and this is empty, the built-in Engraphis mandate in `DEFAULT_SYSTEM_DIRECTIVE` is used. |

---

## Running

Needs Python 3 with `flask` and `requests`.

```bash
cp example.env .env          # then edit backends / models
python app.py                # binds PROXY_HOST:PROXY_PORT
```

Typical Compose layout: mount the directory that contains `app.py` + `.env` + `proxy_cache.db` into the container so config and cache survive restarts. Point Continue / Open WebUI at the proxy port, not at llama-swap or Ollama directly.

Health check:

```bash
curl -s http://127.0.0.1:11435/version
```

---

## Operational notes

- **Prefix cache.** Any compact rebuilds the prompt prefix (new system summary). That is expected and rare. Tool truncate/dedupe on ordinary turns is what used to bust cache every time an agent dumped a large or repeated tool result — that is why `TOOL_OUTPUT_MUTATIONS` defaults to `compact_only`.
- **`INJECT_SYSTEM_DIRECTIVE`.** Injects into the system prompt, not the last user message. Last-user injection moved every turn and invalidated KV.
- **Anchor miss.** If the client resends a rewritten history whose hashes no longer include `last_anchor`, the proxy logs `[Delta Guard Warning]` and treats the whole conversation as uncompacted. The stored summary is still prepended. Usually this means the client is not replaying the original messages.
- **Failed compact.** Previous summary and anchor stay. The current turn is forwarded uncompacted. The next turn can try again.
- **No user turn after snap.** A fallback user message is inserted so Ollama / llama.cpp do not reject the request.
- **Image / multimodal.** `estimate_length()` adds 2048 chars per image block or `images[]` entry so vision turns count toward the trigger.
- **`.env` vs OS env.** OS wins. Safe to bake secrets in Compose and keep `.env` as documentation of the non-secret knobs.

---

## Files

| File | Role |
|---|---|
| `app.py` | The proxy. |
| `example.env` / `.env.example` | Documented defaults. Copy to `.env`. |
| `proxy_cache.db` | Created at runtime. Not committed. |
| `.gitignore` | Ignores `.env`, the SQLite files, `__pycache__`. |

---

## Changelog (behavior that matters)

- **v2.4.0** — `TOOL_OUTPUT_MUTATIONS` (`compact_only` / `always` / `never`). Default `compact_only`: live tool results are forwarded verbatim; truncate+dedupe run on the compact overflow copy and on the post-compact active window. `/version` reports the mode.
- **v2.3.2** — Chat backend follows the client path. `SUMMARY_MODEL=auto` uses that same backend. Directive injects into the system prompt. Default DB is `proxy_cache.db` beside `app.py`. `CHARS_PER_TOKEN=2.55`.
