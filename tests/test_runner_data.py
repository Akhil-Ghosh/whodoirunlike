from collections import Counter
from pathlib import Path

import yaml


def test_runner_seed_list_has_expected_bucket_counts():
    data = yaml.safe_load(Path("data/runners.yml").read_text())
    buckets = Counter(runner["primary_bucket"] for runner in data["runners"])

    assert len(data["runners"]) == 30
    assert buckets == {"800_1500": 10, "5k_10k": 10, "marathon": 10}


def test_runner_slugs_are_unique():
    data = yaml.safe_load(Path("data/runners.yml").read_text())
    slugs = [runner["slug"] for runner in data["runners"]]

    assert len(slugs) == len(set(slugs))

