from __future__ import annotations

import math
import re

from zerocredits.categories import Category

COMMON_INSTRUCTION = (
    "Follow every output-format constraint literally. "
    "Do not add headings, examples, explanations, or commentary "
    "unless requested."
)


SYSTEM_PROMPTS: dict[str, str] = {
    Category.CODE_GENERATION: (
        f"Instruction: Write correct, complete code. Return ONLY the final code and direct answer. Do not wrap your response in an explanation, do not repeat your internal thoughts, and do not provide step-by-step commentary in the final output markdown.{COMMON_INSTRUCTION}"
    ),
    Category.CODE_DEBUG: (
        f"Diagnose and fix the supplied code. Only put the code in reply and nothing else unless specifically asked {COMMON_INSTRUCTION}"
    ),
    Category.MATH: (
        f"Solve accurately and concisely. Give exact answer. No explanation unless specifically asked to explain {COMMON_INSTRUCTION}"
    ),
    Category.SENTIMENT: (
        "Perform the requested sentiment or tone classification. "
        "Return only the exact label, word, or JSON schema requested by the user. "
        "Do not explain your reasoning. Do not restate the task. "
        "Do not use markdown. "
        "Return only allowed labels"
        f"{COMMON_INSTRUCTION}"
    ),
    Category.NER: (
        "Extract only entities present in the text. Preserve exact spelling "
        "and use the requested labels or schema. "
        f"{COMMON_INSTRUCTION}"
    ),
    Category.SUMMARIZATION: (
        "Preserve all requested facts and numbers while minimizing words. "
        "Follow the requested length and format exactly."
        f"{COMMON_INSTRUCTION}"
    ),
    Category.LOGIC: (
        "Reason accurately, then provide only the concise answer requested. No reasoning explanations unless asked"
        f"{COMMON_INSTRUCTION}"
    ),
    Category.FACTUAL: (f"Answer factually and concisely. {COMMON_INSTRUCTION}"),
}


DEFAULT_VISIBLE_BUDGETS: dict[str, int] = {
    Category.CODE_GENERATION: 384,
    Category.CODE_DEBUG: 256,
    Category.MATH: 128,
    Category.SENTIMENT: 32,
    Category.NER: 160,
    Category.SUMMARIZATION: 160,
    Category.LOGIC: 160,
    Category.FACTUAL: 160,
}


MAX_VISIBLE_BUDGETS: dict[str, int] = {
    Category.CODE_GENERATION: 1536,
    Category.CODE_DEBUG: 1536,
    Category.MATH: 512,
    Category.SENTIMENT: 64,
    Category.NER: 1024,
    Category.SUMMARIZATION: 1024,
    Category.LOGIC: 512,
    Category.FACTUAL: 512,
}


def _estimate_prompt_tokens(prompt: str) -> int:
    """
    Conservative approximation for English, JSON, and source code.

    Using roughly three characters per token intentionally overestimates
    many ordinary English prompts, which is safer than underestimating.
    """
    return max(1, math.ceil(len(prompt) / 3))


def _explicit_output_budget(prompt: str) -> int | None:
    """
    Detect explicit numeric output requirements.

    These rules estimate visible answer size only. They never reduce the
    category's default budget.
    """
    text = prompt.casefold()

    word_match = re.search(
        r"\b(?:in\s+|exactly\s+|write\s+(?:a\s+)?)"
        r"(\d+)[-\s]+words?\b",
        text,
    )
    if word_match:
        requested_words = int(word_match.group(1))
        return math.ceil(requested_words * 1.6) + 16

    sentence_match = re.search(
        r"\b(?:in\s+|exactly\s+|write\s+)"
        r"(\d+)[-\s]+sentences?\b",
        text,
    )
    if sentence_match:
        requested_sentences = int(sentence_match.group(1))
        return requested_sentences * 48

    bullet_match = re.search(
        r"\b(?:in\s+|exactly\s+)?"
        r"(\d+)[-\s]+bullet(?:\s+points?)?\b",
        text,
    )
    if bullet_match:
        requested_bullets = int(bullet_match.group(1))
        return requested_bullets * 48

    return None


def _estimate_category_output(
    category: str,
    prompt: str,
    prompt_tokens: int,
) -> int:
    if category == Category.SENTIMENT:
        return 32

    if category == Category.NER:
        # JSON and repeated entity labels consume additional output tokens.
        multiplier = 0.75 if "json" in prompt.casefold() else 0.60
        return max(
            DEFAULT_VISIBLE_BUDGETS[Category.NER],
            math.ceil(prompt_tokens * multiplier),
        )

    if category == Category.SUMMARIZATION:
        # When no explicit length is supplied, allow a summary up to roughly
        # 40% of the source prompt size.
        return max(
            DEFAULT_VISIBLE_BUDGETS[Category.SUMMARIZATION],
            math.ceil(prompt_tokens * 0.40),
        )

    if category == Category.CODE_DEBUG:
        # Corrected code may be roughly as large as, or slightly larger than,
        # the supplied source code.
        return max(
            DEFAULT_VISIBLE_BUDGETS[Category.CODE_DEBUG],
            math.ceil(prompt_tokens * 1.25),
        )

    if category == Category.CODE_GENERATION:
        # Prompt length is not a perfect predictor of generated code size.
        # Use a meaningful minimum, then scale upward for larger specifications.
        return max(
            DEFAULT_VISIBLE_BUDGETS[Category.CODE_GENERATION],
            math.ceil(prompt_tokens * 1.75),
        )

    if category == Category.LOGIC:
        return max(
            DEFAULT_VISIBLE_BUDGETS[Category.LOGIC],
            math.ceil(prompt_tokens * 0.75),
        )

    if category == Category.MATH:
        return max(
            DEFAULT_VISIBLE_BUDGETS[Category.MATH],
            math.ceil(prompt_tokens * 0.60),
        )

    return max(
        DEFAULT_VISIBLE_BUDGETS[Category.FACTUAL],
        math.ceil(prompt_tokens * 0.60),
    )


def estimate_visible_budget(
    category: str,
    prompt: str,
) -> int:
    default_budget = DEFAULT_VISIBLE_BUDGETS.get(
        category,
        DEFAULT_VISIBLE_BUDGETS[Category.FACTUAL],
    )

    maximum_budget = MAX_VISIBLE_BUDGETS.get(
        category,
        MAX_VISIBLE_BUDGETS[Category.FACTUAL],
    )

    prompt_tokens = _estimate_prompt_tokens(prompt)

    category_estimate = _estimate_category_output(
        category=category,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
    )

    explicit_estimate = _explicit_output_budget(prompt)

    estimates = [
        default_budget,
        category_estimate,
    ]

    if explicit_estimate is not None:
        estimates.append(explicit_estimate)

    return min(
        maximum_budget,
        max(estimates),
    )


def build_prompts(
    category: str,
    prompt: str,
) -> tuple[str, str, int]:
    system_prompt = SYSTEM_PROMPTS.get(
        category,
        SYSTEM_PROMPTS[Category.FACTUAL],
    )

    visible_budget = estimate_visible_budget(
        category=category,
        prompt=prompt,
    )

    return system_prompt, prompt, visible_budget
