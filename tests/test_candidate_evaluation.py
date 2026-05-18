from whodoirunlike.evaluation import score_candidate


def test_score_rewards_runner_named_in_relevant_race_clip():
    scored = score_candidate(
        {
            "runner_name": "Faith Kipyegon",
            "primary_bucket": "800_1500",
            "title": "Faith Kipyegon Breaks Olympic Record Women's 1500m highlights",
            "channel": "Olympics",
            "duration_seconds": 181,
            "view_count": 1_000_000,
            "query": "Faith Kipyegon 1500m",
        }
    )

    assert scored.recommendation == "review_first"
    assert scored.score >= 85


def test_score_penalizes_candidate_that_does_not_name_runner():
    scored = score_candidate(
        {
            "runner_name": "Faith Kipyegon",
            "primary_bucket": "800_1500",
            "title": "Perfect Running Form - Joshua Cheptegei is Built Different",
            "channel": "James Dunne",
            "duration_seconds": 488,
            "view_count": 500_000,
            "query": "Faith Kipyegon running form",
        }
    )

    assert scored.recommendation in {"maybe", "skip"}
    assert scored.score < 65


def test_score_caps_search_bleed_through_for_same_event_wrong_runner():
    scored = score_candidate(
        {
            "runner_name": "Emmanuel Wanyonyi",
            "primary_bucket": "800_1500",
            "title": "Rudisha Breaks World Record - Men's 800m Final | London 2012 Olympics",
            "channel": "Olympics",
            "duration_seconds": 579,
            "view_count": 14_000_000,
            "query": "Emmanuel Wanyonyi slow motion running form",
        }
    )

    assert scored.recommendation == "maybe"
    assert scored.score < 65
