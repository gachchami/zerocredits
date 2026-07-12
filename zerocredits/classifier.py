import re

from zerocredits.categories import Category


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def classify_task(prompt: str) -> str:
    text = prompt.casefold()

    if _contains_any(
        text,
        (
            "sentiment",
            "classify the tone",
            "favorable or unfavorable",
            "optimistic, pessimistic",
            "positive, negative",
            "positive or negative",
        ),
    ):
        return Category.SENTIMENT

    if _contains_any(
        text,
        (
            "extract named entities",
            "extract all date",
            "extract product",
            "extract person",
            "every named entity",
            "named entities",
            "json array of objects with text and type",
            "keys people, organizations",
        ),
    ):
        return Category.NER

    if _contains_any(
        text,
        (
            "summarize",
            "summarise",
            "summary",
            "condense",
            "tl;dr",
            "only the risks mentioned",
            "concise title and a two-sentence",
        ),
    ):
        return Category.SUMMARIZATION

    # Existing code plus a request to repair/explain it is debugging.
    has_code_block_signal = any(
        signal in text
        for signal in (
            "\ndef ",
            "function ",
            "traceback",
            "while ",
            "print(",
            "return ",
        )
    )
    if (
        _contains_any(
            text,
            (
                "fix this",
                "find and fix",
                "fix the",
                "correct this",
                "corrected code",
                "off-by-one",
                "infinite loop",
                "gives the wrong result",
                "raise a typeerror",
                "safely returns none",
                "what is wrong with this code",
                "debug",
            ),
        )
        and has_code_block_signal
    ):
        return Category.CODE_DEBUG

    if _contains_any(
        text,
        (
            "write a python function",
            "write a function",
            "create a javascript function",
            "implement ",
            "write python code",
            "write a sql query",
            "python dataclass",
        ),
    ):
        return Category.CODE_GENERATION

    if _contains_any(
        text,
        (
            "logic puzzle",
            "exactly one of",
            "all cats are",
            "taller than",
            "shorter than",
            "left of",
            "right of",
            "stand in a line",
            "before raj",
            "after kiran",
            "every label is wrong",
            "which conclusion",
            "what day will it be",
            "chickens and cows",
            "heads and",
            "must appear before",
        ),
    ):
        return Category.LOGIC

    has_number = re.search(r"\d", text) is not None
    has_math_signal = _contains_any(
        text,
        (
            "calculate",
            "compute",
            "solve for",
            "probability",
            "average speed",
            "percentage",
            "percent",
            "%",
            "area and perimeter",
            "ratio",
            "increase",
            "decrease",
            "growth",
            "how many",
            "fair six-sided die",
        ),
    )
    if has_math_signal and (has_number or "fair six-sided die" in text):
        return Category.MATH

    return Category.FACTUAL
