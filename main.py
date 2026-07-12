from __future__ import annotations

import argparse
import json
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from zerocredits.cache import RuntimeCache
from zerocredits.categories import Category
from zerocredits.classifier import classify_task
from zerocredits.fireworks_client import call_fireworks, get_allowed_models
from zerocredits.local_model import (
    call_local_model,
    can_use_local_model,
    local_output_budget,
)
from zerocredits.prompt_builder import build_prompts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
LOGGER = logging.getLogger("zerocredits")

REMOTE_WORKERS = int(os.environ.get("REMOTE_WORKERS", "6"))
REMOTE_MODEL_ATTEMPTS = int(os.environ.get("REMOTE_MODEL_ATTEMPTS", "2"))
FAILURE_ANSWER = "Unable to generate an answer because all eligible models failed."


@dataclass(frozen=True)
class PlannedTask:
    index: int
    task_id: str
    prompt: str
    category: str
    ranked_models: tuple[str, ...]


def load_tasks(input_path: str) -> list[dict[str, str]]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as file:
        data: Any = json.load(file)

    if not isinstance(data, list):
        raise ValueError("input file must contain a JSON array")

    tasks: list[dict[str, str]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"task at index {index} must be a JSON object")

        task_id = item.get("task_id")
        prompt = item.get("prompt")

        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"task at index {index} has an invalid task_id")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"task {task_id!r} has an invalid prompt")

        tasks.append({"task_id": task_id.strip(), "prompt": prompt.strip()})

    return tasks


def write_results(output_path: str, results: list[dict[str, str]]) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)


def model_family(model: str) -> str:
    name = model.casefold()
    if "kimi" in name:
        return "kimi"
    if "minimax" in name:
        return "minimax"
    if "gemma-4-26b" in name:
        return "gemma_26b"
    if "gemma-4-31b" in name and "nvfp4" in name:
        return "gemma_31b_nvfp4"
    if "gemma-4-31b" in name:
        return "gemma_31b"
    return "other"


def ranked_models(category: str, allowed_models: list[str]) -> list[str]:
    preferences: dict[str, list[str]] = {
        Category.CODE_GENERATION: [
            "kimi", "minimax", "gemma_31b", "gemma_31b_nvfp4", "gemma_26b"
        ],
        Category.CODE_DEBUG: [
            "kimi", "minimax", "gemma_31b", "gemma_31b_nvfp4", "gemma_26b"
        ],
        Category.MATH: [
            "gemma_31b", "minimax", "gemma_31b_nvfp4", "kimi", "gemma_26b"
        ],
        Category.LOGIC: [
            "gemma_31b", "minimax", "gemma_31b_nvfp4", "kimi", "gemma_26b"
        ],
        Category.FACTUAL: [
            "gemma_26b", "gemma_31b", "gemma_31b_nvfp4", "minimax", "kimi"
        ],
        Category.SUMMARIZATION: [
            "gemma_26b", "gemma_31b", "gemma_31b_nvfp4", "minimax", "kimi"
        ],
        Category.NER: [
            "gemma_26b", "gemma_31b", "gemma_31b_nvfp4", "minimax", "kimi"
        ],
        Category.SENTIMENT: [
            "gemma_26b", "gemma_31b_nvfp4", "gemma_31b", "minimax", "kimi"
        ],
    }

    preferred = preferences.get(
        category,
        ["minimax", "kimi", "gemma_31b", "gemma_31b_nvfp4", "gemma_26b"],
    )
    rank = {family: index for index, family in enumerate(preferred)}

    return sorted(
        allowed_models,
        key=lambda model: rank.get(model_family(model), len(preferred)),
    )


def provider_max_tokens_for(model: str, category: str, visible_budget: int) -> int:
    family = model_family(model)
    if family not in {"kimi", "minimax"}:
        return visible_budget

    # MiniMax and Kimi may consume part of this budget as reasoning tokens.
    headroom = {
        Category.SENTIMENT: 256,
        Category.NER: 64,
        Category.FACTUAL: 96,
        Category.SUMMARIZATION: 128,
        Category.MATH: 160,
        Category.LOGIC: 192,
        Category.CODE_DEBUG: 192,
        Category.CODE_GENERATION: 256,
    }.get(category, 96)

    return min(1536, visible_budget + headroom)


def build_plan(
    tasks: list[dict[str, str]],
    allowed_models: list[str],
) -> list[PlannedTask]:
    plan: list[PlannedTask] = []

    for index, task in enumerate(tasks):
        category = classify_task(task["prompt"])
        plan.append(
            PlannedTask(
                index=index,
                task_id=task["task_id"],
                prompt=task["prompt"],
                category=category,
                ranked_models=tuple(ranked_models(category, allowed_models)),
            )
        )

    return plan


def run_remote_task(task: PlannedTask, cache: RuntimeCache) -> str:
    cached = cache.get(task.prompt)
    if cached is not None:
        LOGGER.info("task=%s route=cache", task.task_id)
        return cached

    system_prompt, user_prompt, visible_budget = build_prompts(
        category=task.category,
        prompt=task.prompt,
    )

    errors: list[str] = []
    models = task.ranked_models[: max(1, REMOTE_MODEL_ATTEMPTS)]

    for model in models:
        try:
            max_tokens = provider_max_tokens_for(
                model=model,
                category=task.category,
                visible_budget=visible_budget,
            )
            answer = call_fireworks(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            cache.set(task.prompt, answer)
            LOGGER.info(
                "task=%s category=%s route=remote model=%s max_tokens=%d",
                task.task_id,
                task.category,
                model,
                max_tokens,
            )
            return answer
        except Exception as exc:
            errors.append(f"{model}: {exc}")
            LOGGER.warning(
                "task=%s category=%s route=remote model=%s failed: %s",
                task.task_id,
                task.category,
                model,
                exc,
            )

    raise RuntimeError("all attempted remote models failed; " + " | ".join(errors))


def run_local_task(task: PlannedTask, cache: RuntimeCache) -> str:
    """Run one local task synchronously; never called from the executor."""
    cached = cache.get(task.prompt)
    if cached is not None:
        LOGGER.info("task=%s route=cache", task.task_id)
        return cached

    system_prompt, user_prompt, visible_budget = build_prompts(
        category=task.category,
        prompt=task.prompt,
    )

    if not can_use_local_model(
        category=task.category,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        visible_budget=visible_budget,
    ):
        raise RuntimeError("task is not eligible for local inference")

    max_tokens = local_output_budget(
        category=task.category,
        visible_budget=visible_budget,
    )
    answer = call_local_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )

    cache.set(task.prompt, answer)
    LOGGER.info(
        "task=%s category=%s route=local max_tokens=%d",
        task.task_id,
        task.category,
        max_tokens,
    )
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="/input/tasks.json")
    parser.add_argument("--output", default="/output/results.json")
    args = parser.parse_args()

    allowed_models = get_allowed_models()
    tasks = load_tasks(args.input)
    plan = build_plan(tasks, allowed_models)
    cache = RuntimeCache()
    answers: dict[str, str] = {}

    local_tasks = [task for task in plan if task.category == Category.SENTIMENT]
    remote_tasks = [task for task in plan if task.category != Category.SENTIMENT]

    LOGGER.info(
        "tasks=%d local_sentiment=%d remote=%d remote_workers=%d",
        len(plan),
        len(local_tasks),
        len(remote_tasks),
        REMOTE_WORKERS,
    )

    workers = max(1, min(REMOTE_WORKERS, max(1, len(remote_tasks))))

    # Only remote API calls use worker threads. Local Qwen calls run serially
    # in the main thread, one task at a time.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[str], PlannedTask] = {
            executor.submit(run_remote_task, task, cache): task
            for task in remote_tasks
        }

        local_fallback: list[PlannedTask] = []

        for task in local_tasks:
            try:
                answers[task.task_id] = run_local_task(task, cache)
            except Exception as exc:
                LOGGER.warning(
                    "task=%s category=%s route=local failed: %s",
                    task.task_id,
                    task.category,
                    exc,
                )
                local_fallback.append(task)

        # Failed local tasks are submitted remotely only after all local calls
        # have run serially. No local inference occurs in a worker thread.
        for task in local_fallback:
            futures[executor.submit(run_remote_task, task, cache)] = task

        for future in as_completed(futures):
            task = futures[future]
            try:
                answers[task.task_id] = future.result()
            except Exception:
                LOGGER.exception("task=%s failed", task.task_id)
                answers[task.task_id] = FAILURE_ANSWER

    results = [
        {
            "task_id": task.task_id,
            "answer": answers.get(task.task_id, FAILURE_ANSWER),
        }
        for task in plan
    ]
    write_results(args.output, results)


if __name__ == "__main__":
    main()
