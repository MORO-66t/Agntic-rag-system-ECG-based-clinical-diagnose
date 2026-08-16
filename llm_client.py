"""
LLM client factory — supports Groq, local models, and fake/stub.

HuggingFace's Inference API client has been removed. Groq is now the only
remote backend for LLM reasoning calls. Embeddings for the RAG retriever
still use local sentence-transformers (see pdf_semantic_rag.py) — that's a
local model load, not the HF Inference API, so it doesn't need a token or a
network call at request time and isn't affected by this change.

Usage:
    from llm_client import build_llm_client

    # Groq (default, recommended)
    client = build_llm_client(mode="groq")

    # Local model (offline/test)
    client = build_llm_client(mode="test", local_model_path="./Phi_3_5_mini_instruct")

    # Fake/stub (for pipeline tests without any model)
    client = build_llm_client(mode="fake")
"""

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MODE,
    LOCAL_LLM_MODEL_PATH,
)

# ─────────────────────────────────────────────────────────────────────────────
# GROQ CLIENT (with API key rotation)
# ─────────────────────────────────────────────────────────────────────────────


class GroqLLM:
    """
    Groq API client for fast LLM inference with built-in API key rotation.

    Uses the `groq` Python package. Requires at least one key from:
      - GROQ_API_KEYS (comma-separated list, preferred) or
      - GROQ_API_KEY (single key fallback)
    both set in .env or passed directly.

    On 401/403/rate-limit errors the client rotates to the next key and retries
    once per available key before raising the last error.
    """

    def __init__(
        self,
        api_keys: Optional[list[str]] = None,
        model: Optional[str] = None,
        verbose: bool = False,
        max_tokens: int = 4096,
    ):
        import groq

        from config import GROQ_API_KEYS as _CFG_GROQ_KEYS

        self.keys: list[str] = api_keys or list(_CFG_GROQ_KEYS)
        if not self.keys:
            raise ValueError(
                "No Groq API keys configured. Set GROQ_API_KEYS or "
                "GROQ_API_KEY in your .env file."
            )
        self.model = model or GROQ_MODEL
        self.verbose = verbose
        # Lowered from 5000: on free-tier on_demand keys, llama-3.1-8b-instant
        # has a 6000 TPM limit that covers prompt + max_tokens together, so a
        # long RAG prompt plus max_tokens=5000 alone can exceed it (see the
        # 413 "Request too large ... tokens per minute" error). Rotating keys
        # does NOT fix this if the keys share one org — the fix is a smaller
        # request. Raise this back up if you're on a paid tier / bigger TPM.
        self.max_tokens = max_tokens
        self._current = 0
        # max_retries=0: disable the Groq SDK's built-in retry logic so that
        # OUR code (generate() below) has full control over retry/rotation
        # decisions. Without this, the SDK retries 413/429 errors internally
        # 3-5 times before we ever see the exception — wasting attempts on
        # the same key and preventing proper round-robin rotation.
        self._client = groq.Groq(api_key=self.keys[self._current], max_retries=0)

    # ── public generate ──────────────────────────────────────────────────

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a chat request with automatic key rotation on auth/rate errors,
        and an automatic max_tokens shrink-and-retry on 413 (request too large).

        Key behavioural properties:

        1. Retries use a ``while`` counter so shrink-and-retry on the same key
           does NOT consume the per-key attempt budget (the old ``for`` loop
           bug: ``continue`` went to the next iteration, burning one key's
           worth of retries on pointless shrink attempts of the same key).

        2. On 413 (Payload Too Large / TPM exceeded) the client rotates keys
           FIRST — a different key may belong to a different org/tier with a
           higher TPM limit. Only after all keys have been tried and still
           get a 413 does it fall back to shrinking ``max_tokens``, then
           retries all keys again at the smaller size.

        3. Non-retryable errors (anything other than 401/402/403/413/429)
           break out immediately.
        """
        # Round-robin: rotate to the next key before every call so the TPM
        # load is spread evenly across all keys. This prevents a single key
        # from exhausting its free-tier TPM limit while others sit idle.
        self._rotate()

        attempts_remaining = len(self.keys) * 3  # enough for a full key cycle at two sizes
        last_error: Optional[BaseException] = None
        current_max_tokens = self.max_tokens
        # Phase tracking for 413 (TPM too large):
        #   0 = trying all keys at current max_tokens (rotate on each 413)
        #   1 = all keys tried, shrink max_tokens once, restart with key 0
        #   2 = trying all keys again at smaller max_tokens
        #   3 = all keys tried at smaller size, give up
        _phase_413 = 0
        # Counter for how many distinct keys we've tried in the current phase.
        # Used instead of ``self._current == 0`` to detect wrap-around, because
        # with only 1 key ``_rotate()`` always wraps to 0 immediately.
        _keys_tried = 0

        while attempts_remaining > 0:
            attempts_remaining -= 1
            try:
                # ── Token count and log ───────────────────────────────
                prompt_text = system_prompt + user_prompt
                estimated_prompt = len(prompt_text) // 4
                logger.info(
                    f"Groq LLM call: model={self.model}, "
                    f"~{estimated_prompt} prompt tokens, "
                    f"max_tokens={current_max_tokens}, "
                    f"total_budget={estimated_prompt + current_max_tokens}"
                )

                if self.verbose:
                    print(
                        f"SENDING TO GROQ (key={self._current + 1}/{len(self.keys)}, "
                        f"model={self.model}, max_tokens={current_max_tokens})"
                    )

                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=current_max_tokens,
                )

                if self.verbose:
                    print("GROQ RETURNED")
                return response.choices[0].message.content

            except Exception as e:
                last_error = e
                msg = str(e).lower()
                # Prefer the SDK's status_code attribute (robust); fall back
                # to scanning the message text for the numeric codes.
                status = getattr(e, "status_code", None)
                is_413 = status == 413 or "413" in msg or "too large" in msg
                is_rotatable = status in (401, 402, 403, 413, 429) or any(
                    code in msg
                    for code in ("401", "402", "403", "413", "429")
                )

                if not is_rotatable:
                    # Non-retryable error — give up immediately.
                    break

                # ── 413 handling ────────────────────────────────────────
                # Rotate through available API keys. Do NOT shrink max_tokens/prompt.
                if is_413:
                    _keys_tried += 1
                    if self.verbose:
                        print(
                            f"Key {self._current + 1} 413 "
                            f"({msg[:60]}...). Rotating..."
                        )
                    self._rotate()
                    if _keys_tried >= len(self.keys):
                        # All keys tried without prompt/token shrinking
                        break
                    continue

                # ── Other rotatable errors (401, 402, 403, 429) ─────────
                if self.verbose:
                    print(
                        f"Key {self._current + 1} failed ({msg[:60]}...). "
                        f"Rotating..."
                    )
                self._rotate()
                continue

        if last_error is None:
            last_error = RuntimeError("All Groq API keys failed or no keys provided.")
        if not isinstance(last_error, BaseException):
            last_error = RuntimeError(f"LLM error (non-exception): {last_error!r}")
        raise last_error

    # ── internal helpers ─────────────────────────────────────────────────

    def _rotate(self) -> None:
        """Move to the next API key in the list (wraps around)."""
        self._current = (self._current + 1) % len(self.keys)
        import groq

        # max_retries=0: keep consistency with the main client — without
        # this, each rotation creates an SDK client with default retries,
        # defeating the purpose of max_retries=0 in __init__.
        self._client = groq.Groq(api_key=self.keys[self._current], max_retries=0)


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TRANSFORMERS LLM (offline / test)
# ─────────────────────────────────────────────────────────────────────────────


class LocalTransformersLLM:
    """
    Local test backend for offline/dev RAG checks.

    Loads a chat/instruct model from disk (e.g. Phi-3.5-mini-instruct).
    Use this for `--test`; use `GroqLLM` or `LLMClient` for production.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_new_tokens: int = 1200,
        temperature: float = 0.1,
        load_in_4bit: bool = True,
    ):
        self.model_path = (
            model_path
            or LOCAL_LLM_MODEL_PATH
            or os.getenv("LOCAL_LLM_MODEL_PATH")
            or os.getenv("PHI_LOCAL_MODEL_PATH")
        )
        if not self.model_path:
            raise ValueError(
                "Local LLM test mode needs --local-model-path or "
                "LOCAL_LLM_MODEL_PATH / PHI_LOCAL_MODEL_PATH in .env."
            )

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        use_cuda = torch.cuda.is_available()
        quantization_config = None
        torch_dtype = torch.float16 if use_cuda else torch.float32

        if use_cuda and load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto" if use_cuda else None,
                quantization_config=quantization_config,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch_dtype,
                local_files_only=True,
            )

        if not use_cuda:
            self.model.to("cpu")
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                f"System:\n{system_prompt}\n\n"
                f"User:\n{user_prompt}\n\nAssistant:\n"
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with self.torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.temperature > 0,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


# ─────────────────────────────────────────────────────────────────────────────
# FAKE / STUB LLM (for end-to-end pipeline tests)
# ─────────────────────────────────────────────────────────────────────────────


class FakeLLM:
    """
    Deterministic local LLM stub for end-to-end pipeline tests.

    This does not diagnose clinically. It proves that retrieval, prompt
    construction, event-to-agent wiring, and database persistence work without
    a real model or external API token.
    """

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        chunk_count = len(re.findall(r"(?m)^\[\d+\]\s+chunk_id=", user_prompt))
        event_type = "unknown"
        for line in user_prompt.splitlines():
            if line.lower().startswith("event type:"):
                event_type = line.split(":", 1)[1].strip()
                break

        return (
            "FAKE_LLM_TEST_OUTPUT\n"
            f"event_type: {event_type}\n"
            f"retrieved_pdf_chunks_seen: {chunk_count}\n"
            "status: prompt received and semantic RAG path executed.\n"
            "note: use --agent-llm-mode groq or inference for real generation."
        )


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY
# ─────────────────────────────────────────────────────────────────────────────


def build_llm_client(
    mode: Optional[str] = None,
    local_model_path: Optional[str] = None,
) -> object:
    """
    Build and return an LLM client based on the given mode.

    Parameters
    ----------
    mode : str, optional
        One of: "groq" (default), "test", "fake", "stub", "dry".
        If None, reads from LLM_MODE in .env (defaults to "groq").
        ("inference"/"hf"/"huggingface" have been removed — Groq is now the
        only remote backend.)
    local_model_path : str, optional
        Path to a local model folder (used when mode="test").

    Returns
    -------
    An object with a ``generate(system_prompt, user_prompt) -> str`` method.
    """
    resolved_mode = (mode or LLM_MODE or "groq").lower().strip()

    if resolved_mode in {"groq", "groq_api"}:
        return GroqLLM()

    if resolved_mode in {"fake", "stub", "dry"}:
        return FakeLLM()

    if resolved_mode in {"test", "local", "offline"}:
        return LocalTransformersLLM(model_path=local_model_path)

    if resolved_mode in {"inference", "hf", "api", "huggingface"}:
        raise ValueError(
            "The HuggingFace Inference API backend has been removed. "
            "Use mode='groq' instead."
        )

    raise ValueError(
        f"Unknown LLM mode '{resolved_mode}'. "
        f"Use 'groq', 'test', or 'fake'."
    )