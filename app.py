"""
app.py - Dynamic Multi-Pass Context Compaction Proxy with Persistent Multi-Chat Cache
Version: v2.3.0-multi-chat-cache

Features:
- Delta-based compaction (only summarizes NEW overflow messages)
- Multi-chat persistent cache with per-thread state management
- Gunicorn worker synchronization via multiprocessing.Manager
- Disk-backed eviction with LRU policy (50 chat limit)
"""

import hashlib
import json
import logging
import os
import re
import sys
import time
from collections import OrderedDict
from multiprocessing import Manager
from flask import Flask, jsonify, request, Response
import requests

__VERSION__ = "v2.3.0-multi-chat-cache"

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

# --- Configuration Constants ---
DEFAULT_MODEL_CTX_TOKENS = int(os.getenv("DEFAULT_MODEL_CTX_TOKENS", "131072"))
CHARS_PER_TOKEN = 2.55

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
LLAMASWAP_URL = os.getenv("LLAMASWAP_URL", "http://arceus:9931/v1")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "qwen2.5:3b")

# Multi-chat cache configuration
CACHE_DIR = "./chat_cache"
MAX_CHATS = 50

os.makedirs(CACHE_DIR, exist_ok=True)

# --- Multi-Worker Shared State (Gunicorn Synchronization) ---
_manager = Manager()
# Structure: SHARED_STATE["chats"] = { root_hash: {"summary": str, "last_anchor": str} }
SHARED_STATE = _manager.dict({"chats": {}})

# --- Header Templates ---
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


# ====================================================================
# MULTI-CHAT PERSISTENT CACHE MANAGEMENT
# ====================================================================

def load_persistent_cache():
    """Loads up to 50 most recently accessed chats from disk into shared memory at startup."""
    loaded_chats = {}

    files = [
        os.path.join(CACHE_DIR, f)
        for f in os.listdir(CACHE_DIR)
        if f.endswith(".json")
    ]

    # Sort by last modified time (newest first)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    for file_path in files[:MAX_CHATS]:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                root_hash = data.get("root_hash")
                if root_hash:
                    loaded_chats[root_hash] = {
                        "summary": data.get("summary", ""),
                        "last_anchor": data.get("last_anchor", None),
                    }
        except Exception as e:
            logger.warning(f"[Smart Cache]: Error loading cache file {file_path}: {e}")

    # Initialize Manager Dict
    SHARED_STATE["chats"] = loaded_chats
    logger.info(f"[Smart Cache]: Loaded {len(loaded_chats)} chat contexts from disk.")


def save_chat_to_disk(root_hash, summary, last_anchor):
    """Persists an updated chat state to disk."""
    file_path = os.path.join(CACHE_DIR, f"{root_hash}.json")
    try:
        with open(file_path, "w") as f:
            json.dump(
                {"root_hash": root_hash, "summary": summary, "last_anchor": last_anchor},
                f,
                indent=2,
            )
    except Exception as e:
        logger.warning(f"[Smart Cache Error]: Failed to save chat {root_hash[:8]}...: {e}")


def get_chat_root_key(conversation_msgs):
    """Derives a stable identity key for a chat thread using its first message signature."""
    if not conversation_msgs:
        return "default_chat"
    first_msg = conversation_msgs[0]
    content = str(first_msg.get("content", ""))
    role = str(first_msg.get("role", ""))
    return hashlib.md5(f"{role}:{content}".encode("utf-8")).hexdigest()


def get_or_create_chat_state(root_hash):
    """Retrieves thread state from shared RAM, loading from disk if evicted."""
    chats = dict(SHARED_STATE["chats"])

    if root_hash in chats:
        return chats[root_hash]

    # Check if file exists on disk (if evicted from memory previously)
    file_path = os.path.join(CACHE_DIR, f"{root_hash}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                state = {
                    "summary": data.get("summary", ""),
                    "last_anchor": data.get("last_anchor", None),
                }
                chats[root_hash] = state
                SHARED_STATE["chats"] = chats
                return state
        except Exception:
            pass

    # Brand new chat thread
    return {"summary": "", "last_anchor": None}


def update_chat_state(root_hash, summary, last_anchor):
    """Updates state in shared memory and flushes to disk with LRU eviction."""
    chats = dict(SHARED_STATE["chats"])

    # LRU Eviction if memory store exceeds MAX_CHATS
    if len(chats) >= MAX_CHATS and root_hash not in chats:
        oldest_key = next(iter(chats))  # Evict oldest
        del chats[oldest_key]
        logger.info(f"[Smart Cache]: Evicted chat {oldest_key[:8]}... to make room.")

    chats[root_hash] = {"summary": summary, "last_anchor": last_anchor}

    # Write back to multiprocessing manager
    SHARED_STATE["chats"] = chats

    # Persist to disk
    save_chat_to_disk(root_hash, summary, last_anchor)


# Initialize cache at proxy startup
load_persistent_cache()


# ====================================================================
# CONTEXT ANALYSIS & DYNAMIC THRESHOLDS
# ====================================================================

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


def sanitize_content(content):
    if isinstance(content, str):
        return strip_think_tags(content)
    elif isinstance(content, list):
        cleaned_blocks = []
        for block in content:
            if isinstance(block, dict):
                cleaned = {k: v for k, v in block.items() if k != "type"}
                cleaned["text"] = strip_think_tags(block.get("text", ""))
                cleaned_blocks.append(cleaned)
            else:
                cleaned_blocks.append(block)
        return cleaned_blocks
    return content


def get_message_hash(msg):
    """Generates a stable signature for a message to detect compaction boundaries."""
    content = str(msg.get("content", ""))
    role = str(msg.get("role", ""))
    return hashlib.md5(f"{role}:{content}".encode("utf-8")).hexdigest()


# ====================================================================
# WEBUI NOISE FILTERING
# ====================================================================

WEBUI_NOISE_PATTERNS = [
    r"System message from assistant",
    r"Assistant message from assistant",
    r"Assistant message from user",
    r"Assistant message from system",
    r"User message from assistant",
    r"User message from user",
    r"User message from system",
    r"System message from user",
]

WEBUI_NOISE_REGEX = re.compile("|".join(WEBUI_NOISE_PATTERNS))


def is_webui_noise(content):
    if not isinstance(content, str):
        return False
    return bool(WEBUI_NOISE_REGEX.search(content))


def filter_webui_noise(messages):
    """Strip synthetic wrapper messages inserted by Open WebUI."""
    filtered = []
    stripped_count = 0
    for msg in messages:
        if not is_webui_noise(msg.get("content", "")):
            filtered.append(msg)
        else:
            stripped_count += 1
            logger.debug(f"[WebUI Filter]: Stripped synthetic noise message")
    if stripped_count:
        logger.info(f"[WebUI Filter]: Stripped {stripped_count} synthetic wrapper messages")
    return filtered


def truncate_long_tool_outputs(messages, limit_chars):
    """Clamp excessively long tool outputs to preserve context budget."""
    for msg in messages:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = msg.get("content")
            if isinstance(content, str) and len(content) > limit_chars:
                truncated = content[: limit_chars - 50] + "\n... [TRUNCATED BY PROXY]"
                msg["content"] = truncated
                logger.info(f"[Tool Truncate]: Clamped tool output to {limit_chars} chars")
    return messages


def deduplicate_tool_outputs(messages):
    """Collapse repeated tool outputs into a single instance."""
    seen = {}
    deduped = []
    for msg in messages:
        if msg.get("role") == "tool" or msg.get("role") == "function":
            content = str(msg.get("content", ""))
            if content in seen:
                continue
            seen[content] = True
        deduped.append(msg)
    return deduped


# ====================================================================
# COMPACTION LOGIC
# ====================================================================

def build_compaction_payload(existing_summary, overflow_messages, summary_budget_chars, model_name=None):
    """Construct the OpenAI-compatible compaction payload for Ollama."""
    logger.info(f"[Compactor]: Constructing payload for {len(overflow_messages)} overflow messages...")

    prompt = "You are a context compaction engine. Summarize the following conversation history into a dense, structured, bullet-point summary suitable for preserving critical context across tool calls, API results, and user interactions.\n\n"

    if existing_summary:
        prompt += f"## Previous Summary\n{existing_summary}\n\n"

    prompt += "## New Conversation Segment\n"
    for msg in overflow_messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(block.get("text", "") for block in content if isinstance(block, dict))
        prompt += f"[{role}] {content}\n\n"

    prompt += f"## Instructions\n"
    prompt += f"1. Consolidate previous and new content.\n"
    prompt += f"2. Maintain critical facts: filenames, paths, commands, errors, and user requests.\n"
    prompt += f"3. Output must be less than {summary_budget_chars} characters.\n"
    prompt += f"4. Use bullet points for efficiency.\n"

    return [{"role": "user", "content": prompt}]


def run_anchored_compaction(existing_summary, overflow_messages, target_model, summary_budget_chars, is_llamaswap=True):
    """Run a single anchored compaction pass."""
    logger.info(f"[Compactor]: Running anchored pass for {len(overflow_messages)} messages...")

    try:
        if is_llamaswap:
            return run_anchored_compaction_via_llamaswap(existing_summary, overflow_messages, target_model, summary_budget_chars)
        else:
            return run_anchored_compaction_via_ollama(existing_summary, overflow_messages, target_model, summary_budget_chars)
    except Exception as e:
        logger.error(f"[Compactor Exception]: {e}")
        return existing_summary


def run_anchored_compaction_via_llamaswap(existing_summary, overflow_messages, target_model, summary_budget_chars):
    """Run compaction through llama-swap."""
    try:
        logger.info(f"[Compactor via llama-swap]: Sending {len(overflow_messages)} messages to {target_model}...")
        payload = build_compaction_payload(existing_summary, overflow_messages, summary_budget_chars)

        llamaswap_body = {
            "model": target_model,
            "messages": payload,
            "stream": False,
            "options": {"temperature": 0.3, "max_tokens": 4096}
        }

        res = requests.post(f"{LLAMASWAP_URL}/chat/completions", json=llamaswap_body, timeout=60)
        res.raise_for_status()
        data = res.json()

        summary = data["choices"][0]["message"]["content"]
        logger.info(f"[Compactor via llama-swap]: Success. Generated {len(summary)} char summary.")
        return summary

    except requests.exceptions.HTTPError as e:
        logger.error(f"[Compactor Error via llama-swap]: HTTP {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"[Compactor Exception via llama-swap]: {e}")

    return existing_summary


def run_anchored_compaction_via_ollama(existing_summary, overflow_messages, target_model, summary_budget_chars):
    """Run compaction directly through Ollama."""
    try:
        logger.info(f"[Compactor via Ollama]: Sending {len(overflow_messages)} messages to {target_model}...")
        payload = build_compaction_payload(existing_summary, overflow_messages, summary_budget_chars)

        ollama_body = {
            "model": target_model,
            "messages": payload,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 4096}
        }

        res = requests.post(f"{OLLAMA_URL}/api/chat", json=ollama_body, timeout=60)
        res.raise_for_status()
        data = res.json()

        summary = data["message"]["content"]
        logger.info(f"[Compactor via Ollama]: Success. Generated {len(summary)} char summary.")
        return summary

    except requests.exceptions.HTTPError as e:
        logger.error(f"[Compactor Error via Ollama]: HTTP {e.response.status_code} - {e.response.text}")
    except Exception as e:
        logger.error(f"[Compactor Exception via Ollama]: {e}")

    return existing_summary


def run_chunked_compaction(existing_summary, overflow_messages, target_model, summary_budget_chars, chunk_limit_chars, is_llamaswap=True):
    """Run multi-pass chunked compaction for large overflow segments."""
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


# ====================================================================
# CORE PROCESSING PIPELINE
# ====================================================================

def process_and_tier_messages(messages, body, model_name=None, is_llamaswap=True):
    """Main context processing pipeline with multi-chat cache support."""

    # 1. Identify active thread root key (per-chat state isolation)
    root_key = get_chat_root_key(messages)
    chat_state = get_or_create_chat_state(root_key)

    # Load chat-specific state from cache
    latest_summary = chat_state["summary"]
    last_compacted_hash = chat_state["last_anchor"]

    logger.info(f"[Chat Thread]: {root_key[:8]}... (PID {os.getpid()})")

    # Pre-filter noise and sanitize content
    messages = filter_webui_noise(messages)
    messages = [dict(m) for m in messages]
    for m in messages:
        m["content"] = sanitize_content(m.get("content"))

    # Calculate dynamic thresholds based on target model's context window
    num_ctx, trigger_chars, target_active_chars, summary_budget_chars, chunk_limit_chars, tool_truncate_limit = calculate_dynamic_thresholds(body)

    # Truncate excessive tool outputs
    messages = truncate_long_tool_outputs(messages, tool_truncate_limit)
    messages = deduplicate_tool_outputs(messages)

    # Separate system messages from conversation
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
        f"Float Trigger (80%): {trigger_chars} chars | Active Target (45%): {target_active_chars} chars"
    )

    # 4. STEP 3: Evaluate Trigger on EFFECTIVE Context Only
    if effective_total_chars >= trigger_chars:
        logger.info(f"[Float Trigger Hit]: Effective footprint ({effective_total_chars} chars) EXCEEDED 80% threshold. Compacting...")

        # Calculate Active Working Window against uncompacted messages
        active_budget = target_active_chars
        active_idx = len(uncompacted_conversation)

        for i in range(len(uncompacted_conversation) - 1, -1, -1):
            msg_len = estimate_length(uncompacted_conversation[i])
            if active_budget - msg_len >= 0:
                active_budget -= msg_len
            else:
                active_idx = i
                break

        recent_untouched = uncompacted_conversation[active_idx:]
        overflow_delta = uncompacted_conversation[:active_idx]

        if overflow_delta:
            logger.info(f"[Delta Slice]: Found {len(overflow_delta)} NEW overflow messages. Compacting only these.")
            logger.info(f"[Active Window]: Preserving {len(recent_untouched)} recent messages untouched.")

            # Resolve target model for compaction
            compact_model = model_name or SUMMARY_MODEL

            # Run compaction on overflow only
            updated_summary = run_chunked_compaction(
                latest_summary, overflow_delta, compact_model, summary_budget_chars, chunk_limit_chars, is_llamaswap=is_llamaswap
            )

            # Persist updated state to shared cache + disk
            new_anchor_hash = get_message_hash(uncompacted_conversation[active_idx - 1])
            update_chat_state(root_key, updated_summary, new_anchor_hash)

            # Update local references
            latest_summary = updated_summary
            last_compacted_hash = new_anchor_hash

        else:
            logger.warning("[Delta Slice]: No overflow delta found. Trigger may be miscalibrated.")

    # 5. STEP 4: Reconstruct Final Payload
    if not last_compacted_hash:
        recent_untouched = [dict(m) for m in conversation]
    else:
        recent_untouched = [dict(m) for m in uncompacted_conversation]

    # Construct structured system prompt
    sys_content = system_msgs[0].get("content", "") if system_msgs else "You are a helpful AI assistant."
    structured_system_prompt = sys_content

    if latest_summary:
        structured_system_prompt += f"{HEADER_SUMMARY}{latest_summary}\n"

    if recent_untouched and latest_summary:
        first_active_content = str(recent_untouched[0].get("content", ""))
        if HEADER_ACTIVE not in first_active_content:
            recent_untouched[0]["content"] = f"{HEADER_ACTIVE}\n{first_active_content}"

    # Ensure valid user message exists
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
            recent_untouched.append(fallback_user)
            logger.info("[Fallback User]: Appended last user message as valid continuation anchor.")

    # Append Engraphis system directive
    for m in reversed(recent_untouched):
        if m.get("role") == "user" and str(m.get("content", "")).strip():
            user_content = str(m.get("content", ""))
            if "[SYSTEM DIRECTIVE: ACTIVE EXECUTION RULES" not in user_content:
                m["content"] = f"{user_content}{SYSTEM_ANCHOR_PROMPT}"
                logger.info("[Anchor Directive]: Appended Engraphis mandate to final user message.")
            break

    final_payload = [{"role": "system", "content": structured_system_prompt}] + recent_untouched
    final_chars = sum(estimate_length(m) for m in final_payload)
    logger.info(f"[Proxy Complete]: Returning cleaned payload -> {final_chars} chars (~{final_chars // CHARS_PER_TOKEN} tokens) across {len(final_payload)} msgs.")

    return final_payload


# ====================================================================
# ROUTE HANDLERS
# ====================================================================

def extract_payload_specs(
    body: dict,
    default_model: str = "qwen2.5:3b",
    default_ctx: int = 32768,
) -> tuple[str, int]:
    """Scrapes Open WebUI's body payload to extract the true llama-swap base model
    and active num_ctx.
    """
    # 1. Resolve Base Model String
    target_model = (
        body.get("base_model_id")
        or body.get("model_info", {}).get("base_model_id")
        or body.get("params", {}).get("model")
        or body.get("model")
        or default_model
    )

    # 2. Resolve Dynamic Context Window (num_ctx)
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


@app.route("/api/chat", methods=["POST"])
def chat_completions():
    """Open WebUI-compatible chat endpoint with context compaction."""
    start_time = time.time()
    try:
        body = request.get_json()

        if not body or "messages" not in body:
            return jsonify({"error": "Missing messages in request body"}), 400

        # Extract model and context window specifications
        target_model, target_ctx = extract_payload_specs(
            body,
            default_model="qwen2.5:3b",
            default_ctx=131072,
        )

        # Process and tier messages with compaction
        processed_messages = process_and_tier_messages(
            body["messages"],
            body,
            model_name=target_model,
            is_llamaswap=True,
        )

        # Build final payload for llama-swap
        llamaswap_body = {
            "model": target_model,
            "messages": processed_messages,
            "stream": body.get("stream", False),
            "options": body.get("options", {}),
        }

        # Forward to llama-swap
        res = requests.post(
            f"{LLAMASWAP_URL}/chat/completions",
            json=llamaswap_body,
            timeout=120,
            stream=body.get("stream", False),
        )
        res.raise_for_status()

        if body.get("stream", False):
            def generate():
                for chunk in res.iter_lines():
                    if chunk:
                        yield chunk.decode("utf-8") + "\n"

            return Response(generate(), content_type="text/event-stream")
        else:
            return jsonify(res.json())

    except requests.exceptions.RequestException as e:
        logger.error(f"[Route Error]: {e}")
        return jsonify({"error": f"Proxy error: {str(e)}"}), 502
    except Exception as e:
        logger.error(f"[Route Error]: {e}")
        return jsonify({"error": f"Internal error: {str(e)}"}), 500
    finally:
        elapsed = time.time() - start_time
        logger.info(f"[Request Complete]: {elapsed:.2f}s")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "version": __VERSION__,
        "cache_size": len(dict(SHARED_STATE.get("chats", {}))),
        "pid": os.getpid(),
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=11435, debug=False)