"""
app.py - Dynamic Multi-Pass Context Compaction Proxy
Version: v2.2.0-webui-filter
"""

import hashlib
import json
import logging
import os
import re
import sys
import sqlite3
from flask import Flask, jsonify, request, Response
import requests

__VERSION__ = "v2.2.0-webui-filter"

# Force root logger to bind directly to stdout for Docker visibility
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s [%(levelname)s] [{__VERSION__}] [%(process)d] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("proxy_compactor")

logger.info(f"=== STARTING CONTEXT PROXY SERVER {__VERSION__} ON PORT 11435 ===")

app = Flask(__name__)

DEFAULT_MODEL_CTX_TOKENS = int(os.getenv("DEFAULT_MODEL_CTX_TOKENS", "131072"))
CHARS_PER_TOKEN = 2.55

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLAMASWAP_URL = os.getenv("LLAMASWAP_URL", "http://arceus:9931/v1")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen2.5:3b")

# --- Multi-Chat Persistent Cache Configuration (SQLite) ---
CACHE_DB = "/tmp/proxy_cache.db"
MAX_CHATS = 50


def _get_db_connection():
    """Create a thread-safe SQLite connection with WAL mode for concurrent access."""
    conn = sqlite3.connect(CACHE_DB, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def _ensure_cache_table():
    """Initialize the chat_cache table if it doesn't exist."""
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


# Initialize cache table at proxy startup
_ensure_cache_table()


def get_chat_root_key(conversation_msgs):
    """Derives a stable identity key for a chat thread using its first message signature."""
    if not conversation_msgs:
        return "default_chat"
    first_msg = conversation_msgs[0]
    content = str(first_msg.get("content", ""))
    role = str(first_msg.get("role", ""))
    return hashlib.md5(f"{role}:{content}".encode("utf-8")).hexdigest()


def get_or_create_chat_state(root_hash):
    """Retrieves thread state from SQLite cache, returns default if not found."""
    conn = _get_db_connection()
    try:
        cursor = conn.execute(
            "SELECT summary, last_anchor FROM chat_cache WHERE root_hash = ?",
            (root_hash,)
        )
        row = cursor.fetchone()
        if row:
            return {"summary": row[0] or "", "last_anchor": row[1]}
        return {"summary": "", "last_anchor": None}
    finally:
        conn.close()


def update_chat_state(root_hash, summary, last_anchor):
    """Updates chat state in SQLite with LRU eviction to maintain MAX_CHATS limit."""
    conn = _get_db_connection()
    try:
        # Upsert the chat state
        conn.execute(
            "INSERT OR REPLACE INTO chat_cache (root_hash, summary, last_anchor, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
            (root_hash, summary, last_anchor)
        )
        
        # LRU eviction: remove oldest chats beyond MAX_CHATS
        conn.execute("""
            DELETE FROM chat_cache WHERE root_hash NOT IN (
                SELECT root_hash FROM chat_cache ORDER BY updated_at DESC LIMIT ?
            )
        """, (MAX_CHATS,))
        
        conn.commit()
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

SYSTEM_ANCHOR_PROMPT = (
    "\n\n---\n"
    "[SYSTEM DIRECTIVE: ACTIVE EXECUTION RULES & CONSTRAINTS]\n"
    "1. ENGRAFIS MANDATE: You MUST query Engraphis (`engraphis_recall` / `engraphis_remember`) to retrieve repository architecture, "
    "historical code decisions, and previous diagnostic conclusions before attempting code changes. Save new key findings or architectural decisions back into Engraphis.\n"
    "2. EXACTNESS & COMPLETENESS: Always return complete code blocks for edited files. Do NOT use placeholder comments like '// ... rest of code remains the same'.\n"
    "3. TOOL VERIFICATION: Verify environment parameters and file paths directly via tools before making assumptions.\n"
    "---"
)


def calculate_dynamic_thresholds(body):
    options = body.get("options", {}) if isinstance(body.get("options"), dict) else {}
    num_ctx = body.get("num_ctx") or options.get("num_ctx") or DEFAULT_MODEL_CTX_TOKENS

    total_ctx_chars = num_ctx * CHARS_PER_TOKEN

    trigger_chars = int(total_ctx_chars * 0.80)        # 80% Float Trigger
    target_active_chars = int(total_ctx_chars * 0.45)  # 45% Active Window
    summary_budget_chars = max(1000, int(total_ctx_chars * 0.12)) # 12% Summary Budget
    chunk_limit_chars = int(total_ctx_chars * 0.25)    # 25% Multi-Pass Bite Size
    tool_truncate_limit = int(total_ctx_chars * 0.05)  # 5% Max per Tool Output

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
            else:
                content_len += len(str(block))
    elif content is not None:
        content_len += len(str(content))

    if "tool_calls" in msg:
        content_len += len(str(msg["tool_calls"]))

    return content_len


def strip_think_tags(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


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
                removed_bytes = len(content) - max_tool_limit

                msg["content"] = (
                    f"{head}\n\n"
                    f"[SYSTEM WARNING: TOOL OUTPUT EXCEEDED SAFETY LIMIT. "
                    f"TRUNCATED {removed_bytes} CHARACTERS FROM THE MIDDLE.]\n\n"
                    f"{tail}"
                )
                truncated_count += 1

        sanitized.append(msg)

    if truncated_count > 0:
        logger.info(f"[Tool Guard]: Truncated {truncated_count} massive tool response(s).")

    return sanitized


def deduplicate_tool_outputs(messages):
    seen_signatures = set()
    sanitized = []
    pruned_count = 0

    for msg in reversed(messages):
        m = dict(msg)
        role = m.get("role")

        if role == "tool":
            content = str(m.get("content", "")).strip()
            content_hash = hashlib.md5(content.encode("utf-8")).hexdigest()

            if content_hash in seen_signatures:
                m["content"] = f"[SYSTEM NOTICE: Duplicate tool execution response pruned (Hash: {content_hash[:8]}...)]"
                pruned_count += 1
            else:
                seen_signatures.add(content_hash)

        sanitized.append(m)

    if pruned_count > 0:
        logger.info(f"[Deduplication Guard]: Pruned {pruned_count} duplicate tool output(s).")

    return list(reversed(sanitized))


def run_anchored_compaction(existing_summary, new_messages_to_compact, target_model, summary_budget_chars, is_llamaswap=True):
    target_tokens = max(500, int(abs(summary_budget_chars) // CHARS_PER_TOKEN))
    logger.info(f"[Compactor Pass]: Compacting {len(new_messages_to_compact)} msgs via '{target_model}' (Budget: ~{target_tokens} tokens)...")

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

    try:
        if is_llamaswap:
            target_endpoint = f"{LLAMASWAP_URL}/chat/completions"
            payload = {
                "model": target_model,
                "messages": [
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": user_payload_text}
                ],
                "temperature": 0.2,
                "stream": False
            }
            res = requests.post(target_endpoint, json=payload, timeout=900)
            if res.status_code == 200:
                updated_summary = res.json()["choices"][0]["message"]["content"].strip()
                logger.info("[Compactor Pass via llama-swap]: State updated successfully.")
                return updated_summary
            else:
                logger.error(f"[Compactor Error via llama-swap]: HTTP {res.status_code} - {res.text}")
        else:
            target_endpoint = f"{OLLAMA_URL}/api/generate"
            payload = {
                "model": target_model,
                "prompt": f"{system_instruction}\n{user_payload_text}",
                "stream": False,
                "options": {"temperature": 0.2},
                "keep_alive": "5m"
            }
            res = requests.post(target_endpoint, json=payload, timeout=900)
            if res.status_code == 200:
                updated_summary = res.json().get("response", "").strip()
                logger.info("[Compactor Pass via Ollama]: State updated successfully.")
                return updated_summary
            else:
                logger.error(f"[Compactor Error via Ollama]: HTTP {res.status_code} - {res.text}")

    except Exception as e:
        logger.error(f"[Compactor Exception]: Request failed: {e}")

    return existing_summary


def run_chunked_compaction(existing_summary, overflow_messages, target_model, summary_budget_chars, chunk_limit_chars, is_llamaswap=True):
    logger.info(f"[Multi-Pass Compactor]: Digesting {len(overflow_messages)} overflow msgs in ~{chunk_limit_chars // CHARS_PER_TOKEN} token bites...")

    current_chunk = []
    current_chars = 0
    chunk_index = 1

    for msg in overflow_messages:
        msg_len = estimate_length(msg)

        if current_chars + msg_len > chunk_limit_chars and current_chunk:
            logger.info(f"[Multi-Pass Compactor]: Pass {chunk_index} ({current_chars} chars, {len(current_chunk)} msgs)...")
            existing_summary = run_anchored_compaction(
                existing_summary, current_chunk, target_model, summary_budget_chars, is_llamaswap
            )
            current_chunk = []
            current_chars = 0
            chunk_index += 1

        current_chunk.append(msg)
        current_chars += msg_len

    if current_chunk:
        logger.info(f"[Multi-Pass Compactor]: Final Pass {chunk_index} ({current_chars} chars, {len(current_chunk)} msgs)...")
        existing_summary = run_anchored_compaction(
            existing_summary, current_chunk, target_model, summary_budget_chars, is_llamaswap
        )

    return existing_summary


def get_message_hash(msg):
    """Generates a stable signature for a message to detect compaction boundaries."""
    content = str(msg.get("content", ""))
    role = str(msg.get("role", ""))
    return hashlib.md5(f"{role}:{content}".encode("utf-8")).hexdigest()


def process_and_tier_messages(messages, body, model_name=None, is_llamaswap=True):
    # 1. Identify active thread root key (per-chat state isolation)
    root_key = get_chat_root_key(messages)
    chat_state = get_or_create_chat_state(root_key)

    # Load chat-specific state from cache
    latest_summary = chat_state["summary"]
    last_compacted_hash = chat_state["last_anchor"]

    logger.info(f"[Chat Thread]: {root_key[:8]}... (PID {os.getpid()})")

    num_ctx, TRIGGER_CHARS, TARGET_ACTIVE_CHARS, SUMMARY_BUDGET_CHARS, CHUNK_LIMIT_CHARS, TOOL_TRUNCATE_LIMIT = calculate_dynamic_thresholds(body)

    # 1. Basic Tool Sanitation
    messages = sanitize_tool_outputs(messages, TOOL_TRUNCATE_LIMIT)
    messages = deduplicate_tool_outputs(messages)

    system_msgs = [m for m in messages if m.get("role") == "system"]
    conversation = [m for m in messages if m.get("role") != "system"]

    if not conversation and not system_msgs:
        return messages

    # 2. STEP 1: Strip Previously Summarized History FIRST via Delta Guard
    uncompacted_conversation = conversation
    if last_compacted_hash:
        for idx, msg in enumerate(conversation):
            if get_message_hash(msg) == last_compacted_hash:
                uncompacted_conversation = conversation[idx + 1:]
                logger.info(f"[Delta Guard]: Found anchor at msg {idx}. Stripping {idx + 1} historical messages before audit.")
                break

    # 3. STEP 2: Calculate EFFECTIVE Footprint (Summary + Uncompacted Delta)
    summary_len = len(latest_summary)
    uncompacted_chars = sum(estimate_length(m) for m in uncompacted_conversation)
    system_chars = sum(estimate_length(m) for m in system_msgs)

    effective_total_chars = summary_len + uncompacted_chars + system_chars
    effective_tokens = effective_total_chars // CHARS_PER_TOKEN

    logger.info(
        f"[Context Audit]: Model Ctx: {num_ctx} tokens | Effective Payload: {effective_total_chars} chars (~{effective_tokens} tokens) | "
        f"Float Trigger (80%): {TRIGGER_CHARS} chars | Active Target (45%): {TARGET_ACTIVE_CHARS} chars"
    )

    # 4. STEP 3: Evaluate Trigger on EFFECTIVE Context Only
    if effective_total_chars >= TRIGGER_CHARS:
        logger.info(f"[Float Trigger Hit]: Effective footprint ({effective_total_chars} chars) EXCEEDED 80% threshold. Compacting...")

        # Calculate Active Working Window against uncompacted messages
        active_budget = TARGET_ACTIVE_CHARS
        active_idx = len(uncompacted_conversation)

        for i in range(len(uncompacted_conversation) - 1, -1, -1):
            msg_len = estimate_length(uncompacted_conversation[i])
            if active_budget - msg_len >= 0:
                active_budget -= msg_len
                active_idx = i
            else:
                break

        overflow_delta = uncompacted_conversation[:active_idx]
        recent_untouched = [dict(m) for m in uncompacted_conversation[active_idx:]]

        if overflow_delta:
            logger.info(f"[Delta Compactor]: Digesting {len(overflow_delta)} NEW overflow message(s)...")
            compact_model = model_name or SUMMARY_MODEL
            updated_summary = run_chunked_compaction(
                latest_summary, overflow_delta, compact_model, SUMMARY_BUDGET_CHARS, CHUNK_LIMIT_CHARS, is_llamaswap=is_llamaswap
            )
            # Update hash anchor to the end of the newly compacted delta
            new_anchor_hash = get_message_hash(uncompacted_conversation[active_idx - 1])
            # Persist updated state to shared cache + disk
            update_chat_state(root_key, updated_summary, new_anchor_hash)
            # Update local references
            latest_summary = updated_summary
            last_compacted_hash = new_anchor_hash
    else:
        logger.info("[Context Audit]: Effective footprint within safe boundaries. Bypassing compaction.")
        recent_untouched = [dict(m) for m in uncompacted_conversation]

    sys_content = system_msgs[0].get("content", "") if system_msgs else "You are a helpful AI assistant."
    structured_system_prompt = sys_content

    if latest_summary:
        structured_system_prompt += f"{HEADER_SUMMARY}{latest_summary}\n"

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

    for m in reversed(recent_untouched):
        if m.get("role") == "user":
            user_content = str(m.get("content", ""))
            if "[SYSTEM DIRECTIVE: ACTIVE EXECUTION RULES" not in user_content:
                m["content"] = f"{user_content}{SYSTEM_ANCHOR_PROMPT}"
                logger.info("[Anchor Directive]: Appended Engraphis mandate to final user message.")
            break

    final_payload = [{"role": "system", "content": structured_system_prompt}] + recent_untouched
    final_chars = sum(estimate_length(m) for m in final_payload)
    logger.info(f"[Proxy Complete]: Returning cleaned payload -> {final_chars} chars (~{final_chars // CHARS_PER_TOKEN} tokens) across {len(final_payload)} msgs.")

    return final_payload


# --- ROUTE HANDLERS ---

def extract_payload_specs(
    body: dict,
    default_model: str = "qwen2.5:3b",
    default_ctx: int = 32768,
) -> tuple[str, int]:
  """Scrapes Open WebUI's body payload to extract the true llama-swap base model

  and active num_ctx.
  """
  # 1. Resolve Base Model String
  # Workspace models put base_model_id directly at top-level or inside model_info
  target_model = (
      body.get("base_model_id")
      or body.get("model_info", {}).get("base_model_id")
      or body.get("params", {}).get("model")
      or body.get("model")
      or default_model
  )

  # 2. Resolve Dynamic Context Window (num_ctx)
  # Check root body, nested params, or fallback default
  target_ctx = (
      body.get("num_ctx")
      or body.get("params", {}).get("num_ctx")
      or body.get("model_info", {}).get("params", {}).get("num_ctx")
      or default_ctx
  )

  try:
    target_ctx = int(target_ctx)
  except (ValueError, TypeError):
    target_ctx = default_ctx

  return target_model, target_ctx


@app.route("/v1/transform", methods=["POST"])
def openwebui_transform_filter():
  logger.info("---> [OPEN-WEBUI FILTER DETECTED]: POST /v1/transform ---")
  body = request.get_json() or {}
  messages = body.get("messages", [])

  # Dynamically pull base model and context window specs from Open WebUI payload
  target_model, dynamic_ctx = extract_payload_specs(
      body, default_model=SUMMARY_MODEL, default_ctx=32768
  )

  logger.info(
      f"[Payload Scraper]: Resolved Workspace Model '{body.get('id') or body.get('model')}'"
      f" -> Base Model: '{target_model}' | Target Context: {dynamic_ctx} tokens"
  )

  if messages:
    # Inject the scraped num_ctx into body so downstream audit/math calculations use 180,000
    body["num_ctx"] = dynamic_ctx

    cleaned_messages = process_and_tier_messages(
        messages, body, model_name=target_model, is_llamaswap=True
    )
    return jsonify({"messages": cleaned_messages})

  return jsonify({"messages": messages})


@app.route("/version", methods=["GET"])
def get_version():
    return jsonify({"version": __VERSION__, "status": "active"})


@app.route("/v1/chat/completions", methods=["POST"])
def llama_swap_chat():
    logger.info("--- [Direct Proxy Request]: POST /v1/chat/completions ---")
    body = request.get_json() or {}
    messages = body.get("messages", [])
    model = body.get("model") or request.headers.get("x-model-name") or SUMMARY_MODEL

    if messages:
        body["messages"] = process_and_tier_messages(messages, body, model_name=model, is_llamaswap=True)

    target_url = f"{LLAMASWAP_URL}/chat/completions"
    res = requests.post(target_url, json=body, stream=True, timeout=(15, 900))

    def generate():
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:
                yield chunk

    return Response(generate(), content_type=res.headers.get("content-type", "application/json"))


@app.route("/v1/<path:path>", methods=["GET", "POST", "PUT", "DELETE"])
def proxy_llamaswap(path):
    target_url = f"{LLAMASWAP_URL}/{path}"
    resp = requests.request(
        method=request.method,
        url=target_url,
        headers={k: v for k, v in request.headers if k.lower() != "host"},
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False,
    )
    return Response(resp.content, resp.status_code, headers=dict(resp.headers))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11435)
