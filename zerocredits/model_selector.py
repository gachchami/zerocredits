from __future__ import annotations

from zerocredits.categories import Category
from zerocredits.fireworks_client import get_allowed_models


def _name(model: str) -> str:
    return model.lower().rstrip("/").rsplit("/", 1)[-1]


def _family(model: str) -> str:
    name = _name(model)
    if "kimi" in name:
        return "kimi"
    if "minimax" in name:
        return "minimax"
    if "gemma" in name:
        return "gemma"
    return "other"


def _rank(model: str, category: str) -> tuple[int, int]:
    """Rank by task suitability, not by whichever model appears first.

    Current contest models have clear specializations:
      * Kimi Code: code generation and code debugging.
      * MiniMax M3: general language, extraction, summarization, math, logic.
      * Gemma: preferred general model when the runner actually exposes it.

    Lower tuples are preferred. The second element preserves input order only
    for otherwise-equivalent models.
    """
    family = _family(model)
    name = _name(model)

    if category in {Category.CODE_GENERATION, Category.CODE_DEBUG}:
        order = {"kimi": 0, "gemma": 1, "minimax": 2, "other": 3}
        return order[family], 0

    # All non-code tasks belong on a general-purpose model. Never route them to
    # a code-specialized model merely because a general model needs more tokens.
    order = {"gemma": 0, "minimax": 1, "kimi": 2, "other": 3}

    # Prefer the smaller Gemma for extraction/classification and the stronger
    # variant for math/logic if multiple Gemmas are available.
    if family == "gemma":
        if category in {Category.MATH, Category.LOGIC}:
            variant = 0 if ("31b" in name or "nvfp4" in name) else 1
        else:
            variant = 0 if ("26b" in name or "a4b" in name) else 1
    else:
        variant = 0
    return order[family], variant


def ranked_models(category: str) -> list[str]:
    allowed_models = get_allowed_models()
    if not allowed_models:
        raise RuntimeError("ALLOWED_MODELS is empty or missing")
    return sorted(allowed_models, key=lambda model: _rank(model, category))


def choose_model(category: str) -> str:
    return ranked_models(category)[0]
