from __future__ import annotations

import os
from functools import lru_cache

from llama_cpp import Llama

from zerocredits.categories import Category

LOCAL_CONTEXT_SIZE = 1024
LOCAL_SAFETY_MARGIN = 128

# Only sentiment uses the local CPU model.
LOCAL_CATEGORIES = {
    Category.SENTIMENT,
}


def estimate_tokens(text: str) -> int:
    """Conservatively estimate tokens for short English prompts."""
    return max(1, len(text) // 3)


def local_output_budget(category: str, visible_budget: int) -> int:
    """Limit local generation while leaving room for JSON-formatted answers."""
    limits = {
        Category.SENTIMENT: 32,
    }

    limit = limits.get(category)
    if limit is None:
        return 0

    return min(visible_budget, limit)


def can_use_local_model(
    category: str,
    system_prompt: str,
    user_prompt: str,
    visible_budget: int,
) -> bool:
    if category not in LOCAL_CATEGORIES:
        return False

    model_path = os.environ.get("LOCAL_MODEL_PATH")
    if not model_path or not os.path.isfile(model_path):
        return False

    input_tokens = estimate_tokens(system_prompt) + estimate_tokens(user_prompt)
    output_tokens = local_output_budget(category, visible_budget)
    required_context = input_tokens + output_tokens + LOCAL_SAFETY_MARGIN

    return required_context <= LOCAL_CONTEXT_SIZE


@lru_cache(maxsize=1)
def get_local_model() -> Llama:
    """Load one shared local model instance for serial inference."""
    model_path = os.environ.get("LOCAL_MODEL_PATH")
    if not model_path:
        raise RuntimeError("LOCAL_MODEL_PATH is not configured")
    if not os.path.isfile(model_path):
        raise RuntimeError(f"local model file does not exist: {model_path}")

    local_threads = max(1, int(os.environ.get("LOCAL_THREADS", "2")))

    return Llama(
        model_path=model_path,
        n_ctx=LOCAL_CONTEXT_SIZE,
        n_threads=local_threads,
        n_threads_batch=local_threads,
        n_batch=64,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )


def call_local_model(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
) -> str:
    """Run exactly one task on the local model."""
    response = get_local_model().create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
    )

    choices = response.get("choices", [])
    if not choices:
        raise RuntimeError("local model returned no choices")

    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("local model returned empty content")

    return content.strip()
