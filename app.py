"""
app.py - Dynamic Multi-Pass Context Compaction Proxy
Version: v2.4.0-universal
"""

import hashlib
import logging
import os
import re
import sys
import sqlite3
import uuid
from flask import Flask, jsonify, request, Response
import requests

__VERSION__ = "v2.4.0-universal"

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path):
    """Minimal .env loader so Docker images do not need python-dotenv."""
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv(os.path.join(APP_DIR, ".env"))


def _env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _env_tool_mutation_mode(name, default="compact_only"):
    """
    How aggressively the proxy rewrites tool-result messages before forwarding.

    always        — truncate + dedupe on every request (saves tokens, busts
                    llama.cpp prefix cache whenever a new duplicate appears
                    or a previously-short tool grows past the limit)
    compact_only  — leave live tool results untouched until a compact fires;
                    then rewrite overflow (summarizer input) and the active
                    window. Cache already rebuilds on compact.
    never         — never rewrite tool results in the forwarded chat payload.
                    Summarizer input still clips individual tool bodies at
                    2000 chars so a compact pass cannot blow its own context.
    """
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        text = default
    else:
        text = raw.strip().lower()
    if text in ("always", "on", "true", "1", "yes", "all"):
        return "always"
    if text in ("never", "off", "false", "0", "no", "none"):
        return "never"
    return "compact_only"


def _strip_slash(url):
    return (url or "").rstrip("/")


logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [%(levelname)s] [{__VERSION__}] [%(process)d] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("proxy_compactor")

PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = _env_int("PROXY_PORT", 11435)

DEFAULT_MODEL_CTX_TOKENS = _env_int("DEFAULT_MODEL_CTX_TOKENS", 131072)
CHARS_PER_TOKEN = _env_float("CHARS_PER_TOKEN", 2.55)

OLLAMA_URL = _strip_slash(os.getenv("OLLAMA_URL", "http://ollama:11434"))
LLAMASWAP_URL = _strip_slash(os.getenv("LLAMASWAP_URL", "http://arceus:9931/v1"))

SUMMARY_MODEL = (os.getenv("SUMMARY_MODEL", "qwen2.5:3b") or "").strip()
SUMMARY_BACKEND = (os.getenv("SUMMARY_BACKEND", "ollama") or "ollama").strip().lower()
SUMMARY_URL = _strip_slash(os.getenv("SUMMARY_URL", ""))

COMPACT_TIMEOUT = _env_int("COMPACT_TIMEOUT", 900)
SUMMARY_KEEP_ALIVE = os.getenv("SUMMARY_KEEP_ALIVE", "0").strip()
UNLOAD_SUMMARY_AFTER_COMPACT = _env_bool("UNLOAD_SUMMARY_AFTER_COMPACT", True)

CACHE_DB = os.getenv("CACHE_DB", "").strip() or os.path.join(APP_DIR, "proxy_cache.db")
MAX_CHATS = _env_int("MAX_CHATS", 50)

TRIGGER_RATIO = _env_float("TRIGGER_RATIO", 0.80)
ACTIVE_RATIO = _env_float("ACTIVE_RATIO", 0.45)
SUMMARY_RATIO = _env_float("SUMMARY_RATIO", 0.12)
CHUNK_RATIO = _env_float("CHUNK_RATIO", 0.25)
TOOL_TRUNCATE_RATIO = _env_float("TOOL_TRUNCATE_RATIO", 0.05)
TOOL_OUTPUT_MUTATIONS = _env_tool_mutation_mode("TOOL_OUTPUT_MUTATIONS", "compact_only")

INJECT_SYSTEM_DIRECTIVE = _env_bool("INJECT_SYSTEM_DIRECTIVE", False)
SYSTEM_DIRECTIVE = os.getenv("SYSTEM_DIRECTIVE", "").strip()

DEFAULT_SYSTEM_DIRECTIVE = (
    "\n\n---\n"
    "[SYSTEM DIRECTIVE: ACTIVE EXECUTION RULES & CONSTRAINTS]\n"
    "1. ENGRAFIS MANDATE: You MUST query Engraphis (`engraphis_recall` / `engraphis_remember`) to retrieve repository architecture, "
    "historical code decisions, and previous diagnostic conclusions before attempting code changes. Save new key findings or architectural decisions back into Engraphis.\n"
    "2. EXACTNESS & COMPLETENESS: Always return complete code blocks for edited files. Do NOT use placeholder comments like '// ... rest of code remains the same'.\n"
    "3. TOOL VERIFICATION: Verify environment parameters and file paths directly via tools before making assumptions.\n"
    "---"
)

logger.info(f"=== STARTING CONTEXT PROXY SERVER {__VERSION__} ON PORT {PROXY_PORT} ===")
logger.info(
    f"[Config]: llamaswap={LLAMASWAP_URL} ollama={OLLAMA_URL} "
    f"summary_model={SUMMARY_MODEL or 'auto'} summary_backend={SUMMARY_BACKEND} "
    f"keep_alive={SUMMARY_KEEP_ALIVE} compact_timeout={COMPACT_TIMEOUT} "
    f"cache_db={CACHE_DB} inject_directive={INJECT_SYSTEM_DIRECTIVE} "
    f"tool_mutations={TOOL_OUTPUT_MUTATIONS}"
)

app = Flask(__name__)


def _get_db_connection():
    conn = sqlite3.connect(CACHE_DB, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _ensure_cache_table():
    conn = _get_db_connection()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS chat_cache (
                root_hash TEXT PRIMARY KEY,
                summary TEXT,
                last_anchor TEXT,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
    finally:
        conn.close()


_ensure_cache_table()


def get_or_create_chat_state(root_hash):
    conn = _get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT summary, last_anchor FROM chat_cache WHERE root_hash = ?",
            (root_hash,),
        )
        row = cursor.fetchone()
        if row:
            return {"summary": row[0] or "", "last_anchor": row[1]}
        return {"summary": "", "last_anchor": None}
    finally:
        conn.close()


def update_chat_state(root_hash, summary, last_anchor):
    """Single IMMEDIATE transaction so two in-flight turns cannot split summary vs anchor."""
    conn = _get_db_connection()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT OR REPLACE INTO chat_cache (root_hash, summary, last_anchor, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (root_hash, summary, last_anchor),
        )
        conn.execute(
            """
            DELETE FROM chat_cache WHERE root_hash NOT IN (
                SELECT root_hash FROM (
                    SELECT root_hash FROM chat_cache ORDER BY updated_at DESC LIMIT ?
                )
            )
            """,
            (MAX_CHATS,),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


HEADER_SUMMARY = (
    "\n\n--- [SYSTEM NOTICE: PERSISTENT EXECUTION STATE SUMMARY] ---\n"
    "The following is an anchored, structured summary of historical events prior to the active context window:\n"
)

HEADER_ACTIVE = (
    "\n\n--- [SYSTEM NOTICE: ACTIVE WORKING CONTEXT] ---\n"
    "The following recent context is 100% untouched and contains all active reasoning logs, tool calls, and environment outputs:\n"
)


def backend_from_path(path):
    """Chat backend is the door the client used, not the model name."""
    path = "/" + (path or "").lstrip("/")
    if path == "/v1/transform":
        return "llamaswap"
    if path.startswith("/api/"):
        return "ollama"
    return "llamaswap"


def llamaswap_chat_url():
    base = LLAMASWAP_URL
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/v1/chat/completions"


def llamaswap_root():
    base = LLAMASWAP_URL
    if base.endswith("/v1"):
        return base[: -len("/v1")]
    return base


def resolve_compact_target(chat_backend, chat_model):
    """
    Named SUMMARY_MODEL -> SUMMARY_BACKEND (or SUMMARY_URL).
    auto / empty -> live chat model on the same backend the client hit.
    """
    named = SUMMARY_MODEL.lower() not in ("", "auto")
    if named:
        backend = SUMMARY_BACKEND if SUMMARY_BACKEND in ("ollama", "llamaswap") else "ollama"
        model = SUMMARY_MODEL
        url_override = SUMMARY_URL or None
        return backend, model, url_override

    model = chat_model or SUMMARY_MODEL or "qwen2.5:3b"
    return chat_backend, model, None


def compact_http_timeout():
    if COMPACT_TIMEOUT <= 0:
        return None
    return COMPACT_TIMEOUT


def normalize_keep_alive(value):
    """Accept 0, 5, 5m, -1. Empty -> None (backend default)."""
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def calculate_dynamic_thresholds(body):
    options = body.get("options", {}) if isinstance(body.get("options"), dict) else {}
    raw_ctx = body.get("num_ctx") or options.get("num_ctx") or DEFAULT_MODEL_CTX_TOKENS
    try:
        num_ctx = int(raw_ctx)
    except (TypeError, ValueError):
        num_ctx = DEFAULT_MODEL_CTX_TOKENS

    total_ctx_chars = num_ctx * CHARS_PER_TOKEN
    trigger_chars = int(total_ctx_chars * TRIGGER_RATIO)
    target_active_chars = int(total_ctx_chars * ACTIVE_RATIO)
    summary_budget_chars = max(1000, int(total_ctx_chars * SUMMARY_RATIO))
    chunk_limit_chars = int(total_ctx_chars * CHUNK_RATIO)
    tool_truncate_limit = int(total_ctx_chars * TOOL_TRUNCATE_RATIO)
    return num_ctx, trigger_chars, target_active_chars, summary_budget_chars, chunk_limit_chars, tool_truncate_limit


def estimate_length(msg):
    content = msg.get("content")
    content_len = 0
    if isinstance(content, str):
        content_len += len(content)
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                content_len += len(str(block.get("text", "")))
                if block.get("image_url") or block.get("type") in ("image_url", "image", "input_image"):
                    content_len += 2048
            else:
                content_len += len(str(block))
    elif content is not None:
        content_len += len(str(content))
    if "images" in msg and msg.get("images"):
        content_len += 2048 * len(msg.get("images") or [])
    if "tool_calls" in msg:
        content_len += len(str(msg["tool_calls"]))
    return content_len


def strip_think_tags(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
    return text.strip()


def sanitize_tool_outputs(messages, max_tool_limit):
    sanitized = []
    truncated_count = 0
    for m in messages:
        msg = dict(m)
        role = msg.get("role")
        if role == "tool" or "tool_call_id" in msg:
            content = str(msg.get("content", ""))
            if len(content) > max_tool_limit:
                half_limit = max_tool_limit // 2
                head = content[:half_limit]
                tail = content[-half_limit:]
                removed = len(content) - max_tool_limit
                msg["content"] = (
                    f"{head}\n\n"
                    f"[SYSTEM WARNING: TOOL OUTPUT EXCEEDED SAFETY LIMIT. "
                    f"TRUNCATED {removed} CHARACTERS FROM THE MIDDLE.]\n\n"
                    f"{tail}"
                )
                truncated_count += 1
        sanitized.append(msg)
    if truncated_count:
        logger.info(f"[Tool Guard]: Truncated {truncated_count} massive tool response(s).")
    return sanitized


def deduplicate_tool_outputs(messages):
    seen_signatures = set()
    sanitized = []
    pruned_count = 0
    for msg in reversed(messages):
        m = dict(msg)
        if m.get("role") == "tool":
            content = str(m.get("content", "")).strip()
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
            if content_hash in seen_signatures:
                m["content"] = f"[SYSTEM NOTICE: Duplicate tool execution response pruned (Hash: {content_hash[:8]}...)]"
                pruned_count += 1
            else:
                seen_signatures.add(content_hash)
        sanitized.append(m)
    if pruned_count:
        logger.info(f"[Deduplication Guard]: Pruned {pruned_count} duplicate tool output(s).")
    return list(reversed(sanitized))


def compute_stable_hash(msg):
    """Identity hash from original fields only (before truncation / banners)."""
    if msg.get("_proxy_hash"):
        return msg["_proxy_hash"]
    content = str(msg.get("content", ""))
    role = str(msg.get("role", ""))
    extra = str(msg.get("tool_call_id", "") or msg.get("name", "") or "")
    if "tool_calls" in msg:
        extra += str(msg.get("tool_calls"))
    return hashlib.md5(f"{role}:{content}:{extra}".encode("utf-8")).hexdigest()


def stamp_stable_hashes(messages):
    stamped = []
    for m in messages:
        msg = dict(m)
        msg["_proxy_hash"] = compute_stable_hash(msg)
        stamped.append(msg)
    return stamped


def strip_internal_fields(messages):
    cleaned = []
    for m in messages:
        msg = dict(m)
        msg.pop("_proxy_hash", None)
        cleaned.append(msg)
    return cleaned


def get_message_hash(msg):
    return compute_stable_hash(msg)


def first_user_message(messages):
    for m in messages or []:
        if (m.get("role") == "user") and str(m.get("content", "")).strip():
            return m
    return None


def get_chat_root_key(messages, body=None):
    """Stable thread id: client chat id if present, else first user message."""
    body = body or {}
    for key in ("chat_id", "conversation_id", "session_id"):
        value = body.get(key)
        if value:
            return f"id:{value}"
    nested = body.get("metadata") if isinstance(body.get("metadata"), dict) else {}
    for key in ("chat_id", "conversation_id", "session_id"):
        value = nested.get(key)
        if value:
            return f"id:{value}"

    user = first_user_message(messages)
    if not user:
        if not messages:
            return "default_chat"
        first = messages[0]
        return hashlib.md5(
            f"{first.get('role', '')}:{first.get('content', '')}".encode("utf-8")
        ).hexdigest()
    return hashlib.md5(
        f"user:{user.get('content', '')}".encode("utf-8")
    ).hexdigest()


def _keep_alive_for_pass(is_last_pass):
    ka = normalize_keep_alive(SUMMARY_KEEP_ALIVE)
    if ka is None:
        return None
    if is_last_pass:
        return ka
    if ka in ("0", "0s", "0m"):
        return ka
    return ka


def run_anchored_compaction(
    existing_summary,
    new_messages_to_compact,
    compact_model,
    compact_backend,
    summary_budget_chars,
    is_last_pass=True,
    url_override=None,
):
    target_tokens = max(500, int(abs(summary_budget_chars) // CHARS_PER_TOKEN))
    logger.info(
        f"[Compactor Pass]: Compacting {len(new_messages_to_compact)} msgs via "
        f"'{compact_model}' on {compact_backend} (Budget: ~{target_tokens} tokens)..."
    )

    system_instruction = (
        "You are an expert Context Compactor for an AI coding agent.\n"
        "Analyze the existing summary state and the new execution history delta provided below.\n"
        f"Update the overall persistent execution state. Target a comprehensive summary size up to {target_tokens} tokens.\n"
        "Output strictly in this structured format:\n\n"
        "### 1. CORE USER INTENT & GOALS\n"
        "- High-level goals, requested features, or target architecture.\n\n"
        "### 2. KEY TECHNICAL FACTS & DISCOVERIES\n"
        "- Verified environment parameters, file paths, dependencies, hardware/model specs, or configs.\n\n"
        "### 3. COMPLETED ACTIONS & DIAGNOSTICS\n"
        "- Summary of code modifications, commands executed, file edits, and diagnostic checks that SUCCEEDED.\n\n"
        "### 4. ABANDONED APPROACHES & FAILURES\n"
        "- Errors encountered, failed build steps, or attempted fixes that FAILED.\n\n"
        "### 5. CURRENT PENDING TASK STATE\n"
        "- What was the exact task in progress immediately prior to the recent context window?\n\n"
    )

    user_payload_text = ""
    if existing_summary:
        user_payload_text += f"=== EXISTING SUMMARY STATE ===\n{existing_summary}\n\n"
    user_payload_text += "=== NEW CONVERSATION DELTA TO MERGE ===\n"
    for m in new_messages_to_compact:
        role = m.get("role", "user")
        content = strip_think_tags(str(m.get("content", "")))
        if role == "tool" and len(content) > 2000:
            content = content[:2000] + "\n...[tool output truncated for compaction pass]..."
        user_payload_text += f"[{role.upper()}]: {content}\n"

    keep_alive = _keep_alive_for_pass(is_last_pass)
    timeout = compact_http_timeout()

    try:
        if compact_backend == "llamaswap":
            if url_override:
                target_endpoint = url_override
                if not target_endpoint.endswith("/chat/completions"):
                    if target_endpoint.endswith("/v1"):
                        target_endpoint = f"{target_endpoint}/chat/completions"
                    else:
                        target_endpoint = f"{target_endpoint}/v1/chat/completions"
            else:
                target_endpoint = llamaswap_chat_url()
            payload = {
                "model": compact_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_payload_text},
                ],
                "temperature": 0.2,
                "stream": False,
            }
            if keep_alive is not None:
                payload["keep_alive"] = keep_alive
                payload.setdefault("options", {})["keep_alive"] = keep_alive
            res = requests.post(target_endpoint, json=payload, timeout=timeout)
            if res.status_code == 200:
                updated_summary = strip_think_tags(res.json()["choices"][0]["message"]["content"].strip())
                if not updated_summary:
                    logger.error("[Compactor Error via llama-swap]: Empty summary body.")
                    return existing_summary, False
                logger.info("[Compactor Pass via llama-swap]: State updated successfully.")
                return updated_summary, True
            logger.error(f"[Compactor Error via llama-swap]: HTTP {res.status_code} - {res.text[:500]}")
            return existing_summary, False

        base = url_override or OLLAMA_URL
        target_endpoint = f"{base}/api/chat"
        payload = {
            "model": compact_model,
            "messages": [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_payload_text},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        }
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
        res = requests.post(target_endpoint, json=payload, timeout=timeout)
        if res.status_code == 200:
            data = res.json()
            content = ""
            if isinstance(data.get("message"), dict):
                content = data["message"].get("content", "") or ""
            content = content or data.get("response", "") or ""
            updated_summary = strip_think_tags(content.strip())
            if not updated_summary:
                logger.error("[Compactor Error via Ollama]: Empty summary body.")
                return existing_summary, False
            logger.info("[Compactor Pass via Ollama]: State updated successfully.")
            return updated_summary, True
        logger.error(f"[Compactor Error via Ollama]: HTTP {res.status_code} - {res.text[:500]}")
        return existing_summary, False

    except Exception as e:
        logger.error(f"[Compactor Exception]: Request failed: {e}")
        return existing_summary, False


def unload_summary_model(compact_backend, compact_model, url_override=None):
    if not UNLOAD_SUMMARY_AFTER_COMPACT:
        return
    logger.info(f"[Unload]: Releasing summary model '{compact_model}' on {compact_backend}")
    try:
        if compact_backend == "ollama":
            base = url_override or OLLAMA_URL
            requests.post(
                f"{base}/api/generate",
                json={"model": compact_model, "keep_alive": 0, "prompt": "", "stream": False},
                timeout=30,
            )
            return
        roots = []
        if url_override:
            roots.append(url_override)
        roots.append(llamaswap_root())
        roots.append(LLAMASWAP_URL)
        seen = set()
        for root in roots:
            root = _strip_slash(root)
            if not root or root in seen:
                continue
            seen.add(root)
            for url, kwargs in (
                (f"{root}/unload", {"json": {"model": compact_model}}),
                (f"{root}/v1/unload", {"json": {"model": compact_model}}),
                (f"{root}/unload/{compact_model}", {}),
            ):
                try:
                    resp = requests.post(url, timeout=15, **kwargs)
                    if resp.status_code < 500:
                        logger.info(f"[Unload]: llama-swap accepted {url} ({resp.status_code})")
                        return
                except Exception:
                    continue
        logger.warning("[Unload]: No llama-swap unload endpoint accepted the request.")
    except Exception as e:
        logger.warning(f"[Unload]: Failed to unload '{compact_model}': {e}")


def run_chunked_compaction(
    existing_summary,
    overflow_messages,
    compact_model,
    compact_backend,
    summary_budget_chars,
    chunk_limit_chars,
    url_override=None,
):
    logger.info(
        f"[Multi-Pass Compactor]: Digesting {len(overflow_messages)} overflow msgs in "
        f"~{int(chunk_limit_chars // CHARS_PER_TOKEN)} token bites on {compact_backend}/{compact_model}..."
    )

    chunks = []
    current_chunk = []
    current_chars = 0
    for msg in overflow_messages:
        msg_len = estimate_length(msg)
        if current_chars + msg_len > chunk_limit_chars and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_chars = 0
        current_chunk.append(msg)
        current_chars += msg_len
    if current_chunk:
        chunks.append(current_chunk)

    summary = existing_summary
    for index, chunk in enumerate(chunks, start=1):
        is_last = index == len(chunks)
        logger.info(f"[Multi-Pass Compactor]: Pass {index}/{len(chunks)} ({len(chunk)} msgs)...")
        summary, ok = run_anchored_compaction(
            summary,
            chunk,
            compact_model,
            compact_backend,
            summary_budget_chars,
            is_last_pass=is_last,
            url_override=url_override,
        )
        if not ok:
            if UNLOAD_SUMMARY_AFTER_COMPACT or normalize_keep_alive(SUMMARY_KEEP_ALIVE) in ("0", "0s", "0m"):
                unload_summary_model(compact_backend, compact_model, url_override=url_override)
            return summary, False

    if UNLOAD_SUMMARY_AFTER_COMPACT or normalize_keep_alive(SUMMARY_KEEP_ALIVE) in ("0", "0s", "0m"):
        unload_summary_model(compact_backend, compact_model, url_override=url_override)

    if summary_budget_chars and summary and len(summary) > summary_budget_chars * 2:
        summary = summary[:summary_budget_chars] + "\n...[summary truncated to budget]..."

    return summary, True


def snap_active_index(messages, active_idx):
    """Cut on a user turn or a complete tool round — never between a call and its results."""
    if active_idx <= 0 or active_idx >= len(messages):
        return active_idx
    i = active_idx
    if messages[i].get("role") == "tool":
        while i > 0 and messages[i].get("role") == "tool":
            i -= 1
    if messages[i].get("role") != "user":
        j = i
        while j > 0 and messages[j].get("role") != "user":
            j -= 1
        if messages[j].get("role") == "user":
            i = j
    return max(i, 0)


def process_and_tier_messages(messages, body, chat_model=None, chat_backend="llamaswap"):
    req_id = uuid.uuid4().hex[:8]
    try:
        hdr = request.headers.get("x-request-id")
        if hdr:
            req_id = hdr
    except RuntimeError:
        pass

    root_key = get_chat_root_key(messages, body)
    chat_state = get_or_create_chat_state(root_key)
    latest_summary = chat_state["summary"]
    last_compacted_hash = chat_state["last_anchor"]

    logger.info(
        f"[Chat Thread]: req={req_id} key={root_key[:16]}... backend={chat_backend} "
        f"model={chat_model} (PID {os.getpid()})"
    )

    num_ctx, TRIGGER_CHARS, TARGET_ACTIVE_CHARS, SUMMARY_BUDGET_CHARS, CHUNK_LIMIT_CHARS, TOOL_TRUNCATE_LIMIT = calculate_dynamic_thresholds(body)

    messages = stamp_stable_hashes(messages)
    if TOOL_OUTPUT_MUTATIONS == "always":
        messages = sanitize_tool_outputs(messages, TOOL_TRUNCATE_LIMIT)
        messages = deduplicate_tool_outputs(messages)

    system_msgs = [m for m in messages if m.get("role") == "system"]
    conversation = [m for m in messages if m.get("role") != "system"]

    if not conversation and not system_msgs:
        return messages

    uncompacted_conversation = conversation
    if last_compacted_hash:
        for idx in range(len(conversation) - 1, -1, -1):
            if get_message_hash(conversation[idx]) == last_compacted_hash:
                uncompacted_conversation = conversation[idx + 1 :]
                logger.info(
                    f"[Delta Guard]: Stripping {idx + 1} historical messages up to anchor idx {idx} "
                    f"({last_compacted_hash[:8]}...)"
                )
                break
        else:
            logger.warning(
                f"[Delta Guard Warning]: Anchor hash {last_compacted_hash[:8]}... not found in conversation stream!"
            )

    summary_len = len(latest_summary)
    uncompacted_chars = sum(estimate_length(m) for m in uncompacted_conversation)
    system_chars = sum(estimate_length(m) for m in system_msgs)
    effective_total_chars = summary_len + uncompacted_chars + system_chars
    effective_tokens = int(effective_total_chars / CHARS_PER_TOKEN)

    logger.info(
        f"[Context Audit]: Model Ctx: {num_ctx} tokens | Effective Payload: {effective_total_chars} chars "
        f"(~{effective_tokens} tokens) | Float Trigger ({int(TRIGGER_RATIO * 100)}%): {TRIGGER_CHARS} chars | "
        f"Active Target ({int(ACTIVE_RATIO * 100)}%): {TARGET_ACTIVE_CHARS} chars"
    )

    recent_untouched = [dict(m) for m in uncompacted_conversation]
    if effective_total_chars >= TRIGGER_CHARS:
        logger.info(
            f"[Float Trigger Hit]: Effective footprint ({effective_total_chars} chars) EXCEEDED {int(TRIGGER_RATIO * 100)}% threshold. Compacting..."
        )
        active_budget = TARGET_ACTIVE_CHARS
        active_idx = len(uncompacted_conversation)
        for i in range(len(uncompacted_conversation) - 1, -1, -1):
            msg_len = estimate_length(uncompacted_conversation[i])
            if active_budget - msg_len >= 0:
                active_budget -= msg_len
                active_idx = i
            else:
                break
        active_idx = snap_active_index(uncompacted_conversation, active_idx)
        overflow_delta = uncompacted_conversation[:active_idx]
        recent_untouched = [dict(m) for m in uncompacted_conversation[active_idx:]]

        if overflow_delta:
            compact_backend, compact_model, url_override = resolve_compact_target(chat_backend, chat_model)
            logger.info(
                f"[Delta Compactor]: Digesting {len(overflow_delta)} NEW overflow message(s) "
                f"via {compact_backend}/{compact_model}"
            )
            compact_src = [dict(m) for m in overflow_delta]
            if TOOL_OUTPUT_MUTATIONS != "never":
                compact_src = sanitize_tool_outputs(compact_src, TOOL_TRUNCATE_LIMIT)
                compact_src = deduplicate_tool_outputs(compact_src)
            updated_summary, ok = run_chunked_compaction(
                latest_summary,
                compact_src,
                compact_model,
                compact_backend,
                SUMMARY_BUDGET_CHARS,
                CHUNK_LIMIT_CHARS,
                url_override=url_override,
            )
            if ok:
                new_anchor_hash = get_message_hash(overflow_delta[-1])
                logger.info(f"[Anchor Update]: Set anchor to last overflow msg ({new_anchor_hash[:8]}...)")
                update_chat_state(root_key, updated_summary, new_anchor_hash)
                latest_summary = updated_summary
                if TOOL_OUTPUT_MUTATIONS == "compact_only":
                    recent_untouched = sanitize_tool_outputs(recent_untouched, TOOL_TRUNCATE_LIMIT)
                    recent_untouched = deduplicate_tool_outputs(recent_untouched)
                    logger.info(
                        "[Tool Guard]: compact_only — truncated/deduped tool results in the "
                        "post-compact active window only."
                    )
            else:
                logger.error(
                    "[Compactor]: Compact failed; leaving previous anchor in place so history is not stripped."
                )
                recent_untouched = [dict(m) for m in uncompacted_conversation]
    else:
        logger.info("[Context Audit]: Effective footprint within safe boundaries. Bypassing compaction.")

    sys_content = system_msgs[0].get("content", "") if system_msgs else "You are a helpful AI assistant."
    structured_system_prompt = sys_content
    if latest_summary:
        structured_system_prompt += f"{HEADER_SUMMARY}{latest_summary}\n"

    if INJECT_SYSTEM_DIRECTIVE:
        directive = SYSTEM_DIRECTIVE or DEFAULT_SYSTEM_DIRECTIVE
        marker = "[SYSTEM DIRECTIVE: ACTIVE EXECUTION RULES"
        if marker not in structured_system_prompt and directive not in structured_system_prompt:
            structured_system_prompt += directive
            logger.info("[Anchor Directive]: Appended system directive to system prompt (KV-stable).")

    if recent_untouched and latest_summary:
        first_active_content = str(recent_untouched[0].get("content", ""))
        if HEADER_ACTIVE not in first_active_content:
            recent_untouched[0]["content"] = f"{HEADER_ACTIVE}\n{first_active_content}"

    has_valid_user = any(
        m.get("role") == "user" and str(m.get("content", "")).strip() != ""
        for m in recent_untouched
    )
    if not has_valid_user:
        last_user_msg = None
        for m in reversed(conversation):
            if m.get("role") == "user":
                last_user_msg = m
                break
        if last_user_msg:
            fallback_user = dict(last_user_msg)
            if not str(fallback_user.get("content", "")).strip():
                fallback_user["content"] = "Continue execution based on current context."
            recent_untouched.insert(0, fallback_user)

    final_payload = [{"role": "system", "content": structured_system_prompt}] + recent_untouched
    final_payload = strip_internal_fields(final_payload)
    final_chars = sum(estimate_length(m) for m in final_payload)
    logger.info(
        f"[Proxy Complete]: req={req_id} Returning cleaned payload -> {final_chars} chars "
        f"(~{int(final_chars / CHARS_PER_TOKEN)} tokens) across {len(final_payload)} msgs."
    )
    return final_payload


def extract_payload_specs(body, default_model="qwen2.5:3b", default_ctx=None):
    if default_ctx is None:
        default_ctx = DEFAULT_MODEL_CTX_TOKENS
    target_model = (
        body.get("base_model_id")
        or (body.get("model_info") or {}).get("base_model_id")
        or (body.get("params") or {}).get("model")
        or body.get("model")
        or default_model
    )
    target_ctx = (
        body.get("num_ctx")
        or (body.get("params") or {}).get("num_ctx")
        or ((body.get("model_info") or {}).get("params") or {}).get("num_ctx")
        or default_ctx
    )
    try:
        target_ctx = int(target_ctx)
    except (ValueError, TypeError):
        target_ctx = default_ctx
    return target_model, target_ctx


def ollama_messages_from_body(body):
    if body.get("messages"):
        return body.get("messages")
    prompt = body.get("prompt")
    if prompt:
        msgs = []
        if body.get("system"):
            msgs.append({"role": "system", "content": body.get("system")})
        msgs.append({"role": "user", "content": prompt})
        return msgs
    return []


def apply_compacted_messages_to_ollama_body(body, cleaned):
    if "messages" in body or body.get("messages") is not None:
        body["messages"] = cleaned
        if cleaned and cleaned[0].get("role") == "system":
            body["system"] = cleaned[0].get("content", "")
        return body
    system = ""
    rest = cleaned
    if cleaned and cleaned[0].get("role") == "system":
        system = cleaned[0].get("content", "")
        rest = cleaned[1:]
    body["system"] = system
    body["prompt"] = "\n".join(
        f"{m.get('role', 'user').upper()}: {m.get('content', '')}" for m in rest
    )
    return body


def stream_upstream(method, url, **kwargs):
    res = requests.request(method, url, stream=True, timeout=(15, compact_http_timeout() or 900), **kwargs)

    def generate():
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                yield chunk

    headers = {k: v for k, v in res.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding", "connection")}
    return Response(generate(), status=res.status_code, headers=headers)


@app.route("/version", methods=["GET"])
def get_version():
    return jsonify({
        "version": __VERSION__,
        "status": "active",
        "summary_model": SUMMARY_MODEL or "auto",
        "summary_backend": SUMMARY_BACKEND,
        "keep_alive": SUMMARY_KEEP_ALIVE,
        "cache_db": CACHE_DB,
        "inject_directive": INJECT_SYSTEM_DIRECTIVE,
        "chars_per_token": CHARS_PER_TOKEN,
        "tool_output_mutations": TOOL_OUTPUT_MUTATIONS,
        "tool_truncate_ratio": TOOL_TRUNCATE_RATIO,
    })


@app.route("/v1/transform", methods=["POST"])
def openwebui_transform_filter():
    logger.info("---> [OPEN-WEBUI FILTER DETECTED]: POST /v1/transform ---")
    body = request.get_json() or {}
    messages = body.get("messages", [])
    target_model, dynamic_ctx = extract_payload_specs(body, default_model=SUMMARY_MODEL or "qwen2.5:3b")
    logger.info(
        f"[Payload Scraper]: Workspace '{body.get('id') or body.get('model')}' "
        f"-> Base Model: '{target_model}' | Target Context: {dynamic_ctx} tokens"
    )
    if messages:
        body["num_ctx"] = dynamic_ctx
        cleaned = process_and_tier_messages(
            messages, body, chat_model=target_model, chat_backend="llamaswap"
        )
        return jsonify({"messages": cleaned})
    return jsonify({"messages": messages})


@app.route("/v1/chat/completions", methods=["POST"])
def llama_swap_chat():
    logger.info("--- [Direct Proxy Request]: POST /v1/chat/completions ---")
    body = request.get_json() or {}
    messages = body.get("messages", [])
    model = body.get("model") or request.headers.get("x-model-name") or SUMMARY_MODEL
    if messages:
        body["messages"] = process_and_tier_messages(
            messages, body, chat_model=model, chat_backend="llamaswap"
        )
    return stream_upstream("POST", llamaswap_chat_url(), json=body)


@app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy_llamaswap(path):
    target_url = f"{llamaswap_root()}/v1/{path}"
    if LLAMASWAP_URL.endswith("/v1"):
        target_url = f"{LLAMASWAP_URL}/{path}"
    resp = requests.request(
        method=request.method,
        url=target_url,
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        timeout=(15, compact_http_timeout() or 900),
    )
    headers = {k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding", "connection")}
    return Response(resp.content, resp.status_code, headers=headers)


@app.route("/api/chat", methods=["POST"])
def ollama_chat():
    logger.info("--- [Ollama Proxy Request]: POST /api/chat ---")
    body = request.get_json() or {}
    messages = ollama_messages_from_body(body)
    model = body.get("model") or SUMMARY_MODEL
    if messages:
        cleaned = process_and_tier_messages(messages, body, chat_model=model, chat_backend="ollama")
        body = apply_compacted_messages_to_ollama_body(body, cleaned)
    return stream_upstream("POST", f"{OLLAMA_URL}/api/chat", json=body)


@app.route("/api/generate", methods=["POST"])
def ollama_generate():
    logger.info("--- [Ollama Proxy Request]: POST /api/generate ---")
    body = request.get_json() or {}
    messages = ollama_messages_from_body(body)
    model = body.get("model") or SUMMARY_MODEL
    if messages:
        cleaned = process_and_tier_messages(messages, body, chat_model=model, chat_backend="ollama")
        body = apply_compacted_messages_to_ollama_body(body, cleaned)
    return stream_upstream("POST", f"{OLLAMA_URL}/api/generate", json=body)


@app.route("/api/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
def proxy_ollama(path):
    target_url = f"{OLLAMA_URL}/api/{path}"
    resp = requests.request(
        method=request.method,
        url=target_url,
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
        timeout=(15, compact_http_timeout() or 900),
    )
    headers = {k: v for k, v in resp.headers.items() if k.lower() not in ("content-encoding", "transfer-encoding", "connection")}
    return Response(resp.content, resp.status_code, headers=headers)


if __name__ == "__main__":
    app.run(host=PROXY_HOST, port=PROXY_PORT)
