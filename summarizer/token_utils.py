"""Token estimation utilities and small caches.

This module provides estimate_tokens and count_tokens used across the project.
"""
import hashlib
from summarizer.cache import fetch_token_count_from_cache, commit_token_count_to_cache
from typing import Any

# Caches
_tiktoken_encoders: dict = {}
_token_count_cache: dict = {}


async def estimate_tokens(text: str, model_name: str | None = None) -> int:
    if not text:
        return 0
    try:
        key = hashlib.sha1(((model_name or '') + ':' + text).encode('utf-8')).hexdigest()
    except Exception:
        key = None

    # First check in-process memory cache
    if key is not None and key in _token_count_cache:
        return _token_count_cache[key]

    # Then check persistent SQL cache (best-effort)
    if model_name is not None and key is not None:
        try:
            cached_val = await fetch_token_count_from_cache(text, model_name)
            if cached_val is not None:
                _token_count_cache[key] = int(cached_val)
                return int(cached_val)
        except Exception:
            pass

    try:
        import tiktoken
        enc = None
        enc_key = model_name or '__default__'
        if enc_key in _tiktoken_encoders:
            enc = _tiktoken_encoders[enc_key]
        else:
            try:
                if model_name:
                    enc = tiktoken.encoding_for_model(model_name)
                else:
                    enc = tiktoken.get_encoding('o200k_harmony')
            except Exception:
                try:
                    enc = tiktoken.get_encoding('o200k_harmony')
                except Exception:
                    enc = None
            if enc is not None:
                _tiktoken_encoders[enc_key] = enc

        if enc is not None:
            tok_count = len(enc.encode(text))
            if key is not None:
                _token_count_cache[key] = tok_count
            # store in SQL cache as well (best-effort)
            try:
                if model_name is not None:
                    await commit_token_count_to_cache(text, model_name, int(tok_count))
            except Exception:
                pass
            return tok_count
    except Exception:
        pass

    tok_count = max(1, int(len(text) / 4))
    if key is not None:
        _token_count_cache[key] = tok_count
    try:
        if model_name is not None:
            await commit_token_count_to_cache(text, model_name, int(tok_count))
    except Exception:
        pass
    return tok_count


async def count_tokens(obj: Any, model_name: str | None = None, per_message_overhead: bool = True) -> int:
    OVERHEAD_PER_MESSAGE = 4 if per_message_overhead else 0

    if isinstance(obj, str):
        return await estimate_tokens(obj, model_name) + OVERHEAD_PER_MESSAGE

    if isinstance(obj, dict):
        role = obj.get('role', '')
        content = obj.get('content', '')
        return await estimate_tokens(str(role), model_name) + await estimate_tokens(str(content), model_name) + OVERHEAD_PER_MESSAGE

    if isinstance(obj, list):
        total = 0
        for elem in obj:
            total += await count_tokens(elem, model_name=model_name, per_message_overhead=per_message_overhead)
        return total

    return await estimate_tokens(str(obj), model_name) + OVERHEAD_PER_MESSAGE
