from __future__ import annotations

import logging
import os
from functools import lru_cache

from openai import OpenAI

LOGGER = logging.getLogger("zerocredits.fireworks_client")


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ["FIREWORKS_BASE_URL"],
        timeout=float(os.environ.get("FIREWORKS_TIMEOUT_SECONDS", "12")),
        max_retries=0,
    )


def get_allowed_models() -> list[str]:
    raw_models = os.environ.get("ALLOWED_MODELS", "")

    models = [model.strip() for model in raw_models.split(",") if model.strip()]

    if not models:
        raise RuntimeError("ALLOWED_MODELS is missing or empty")

    return models


def _build_messages(
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> list[dict[str, str]]:
    model_name = model.casefold()

    if "minimax" in model_name or "kimi" in model_name:
        reinforced_user_prompt = f"{system_prompt}\n\nTask:\n{user_prompt}"

        return [
            {
                "role": "user",
                "content": reinforced_user_prompt,
            }
        ]

    return [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]


def call_fireworks(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
) -> str:
    response = get_client().chat.completions.create(
        model=model,
        messages=_build_messages(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        ),
        temperature=0,
        max_tokens=max_tokens,
    )

    if not response.choices:
        raise RuntimeError(f"{model} returned no choices")

    choice = response.choices[0]
    content = choice.message.content

    if not isinstance(content, str) or not content.strip():
        reasoning = getattr(
            choice.message,
            "reasoning_content",
            None,
        )

        raise RuntimeError(
            f"{model} returned no visible answer "
            f"(finish_reason={choice.finish_reason!r}, "
            f"reasoning_present={bool(reasoning)}, "
            f"max_tokens={max_tokens})"
        )

    answer = content.strip()

    if choice.finish_reason == "length":
        LOGGER.warning(
            "model=%s reached max_tokens=%d but returned visible content",
            model,
            max_tokens,
        )

    return answer
