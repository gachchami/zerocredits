from zerocredits.classifier import classify_task
from zerocredits.model_selector import choose_model, ranked_models


def test_logic_07_is_logic():
    prompt = (
        "Five people stand in a line. Mira is before Raj, Raj is before Kiran, "
        "and Dev is after Kiran. Which person must appear before Dev?"
    )
    assert classify_task(prompt) == "logic"


def test_logic_routes_to_versioned_gemma(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        "accounts/fireworks/models/minimax-m3,"
        "accounts/fireworks/models/gemma-4-31b-it-nvfp4-v2,"
        "accounts/fireworks/models/kimi-k2p7-code",
    )
    assert "gemma" in choose_model("logic")
    assert "nvfp4" in choose_model("logic")


def test_small_tasks_prefer_small_gemma(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        "accounts/fireworks/models/gemma-4-31b-it,"
        "accounts/fireworks/models/gemma-4-26b-a4b-it-preview,"
        "accounts/fireworks/models/kimi-k2p7-code",
    )
    assert "26b" in choose_model("factual")


def test_logic_prefers_minimax_over_kimi(monkeypatch):
    monkeypatch.setenv(
        "ALLOWED_MODELS",
        "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code",
    )
    models = ranked_models("logic")
    assert "minimax" in models[0]


def test_minimax_gets_reasoning_headroom():
    from zerocredits.fireworks_client import output_budget_for_model

    model = "accounts/fireworks/models/minimax-m3"
    assert output_budget_for_model(model, 8) == 512
    assert output_budget_for_model(model, 80) == 768


def test_non_reasoning_model_keeps_requested_budget():
    from zerocredits.fireworks_client import output_budget_for_model

    model = "accounts/fireworks/models/gemma-4-31b-it-nvfp4"
    assert output_budget_for_model(model, 80) == 80
