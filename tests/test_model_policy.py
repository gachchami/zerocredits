import os

from zerocredits.categories import Category
from zerocredits.fireworks_client import output_budget_for_model
from zerocredits.model_selector import ranked_models

MINIMAX = "accounts/fireworks/models/minimax-m3"
KIMI = "accounts/fireworks/models/kimi-k2p7-code"


def setup_module():
    os.environ["ALLOWED_MODELS"] = f"{MINIMAX},{KIMI}"


def test_kimi_is_used_only_for_code_categories():
    assert ranked_models(Category.CODE_GENERATION)[0] == KIMI
    assert ranked_models(Category.CODE_DEBUG)[0] == KIMI


def test_minimax_is_used_for_every_non_code_category():
    categories = [
        Category.FACTUAL,
        Category.MATH,
        Category.SENTIMENT,
        Category.SUMMARIZATION,
        Category.NER,
        Category.LOGIC,
    ]
    for category in categories:
        assert ranked_models(category)[0] == MINIMAX


def test_minimax_gets_reasoning_headroom():
    assert output_budget_for_model(MINIMAX, 8) == 512
    assert output_budget_for_model(MINIMAX, 24) == 512
    assert output_budget_for_model(MINIMAX, 80) == 768
    assert output_budget_for_model(MINIMAX, 120) == 768


def test_kimi_keeps_requested_budget():
    assert output_budget_for_model(KIMI, 80) == 80
