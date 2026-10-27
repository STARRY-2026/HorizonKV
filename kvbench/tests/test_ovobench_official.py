from sparsemm_native.run_ovobench import hermes_correct, hermes_queries, official_correct, queries, summary


def test_official_query_expansion_and_prompts():
    items = queries()
    assert len(items) == 3035
    assert len({item["query_id"] for item in items}) == 3035
    assert {item["task"] for item in items} == {
        "EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"
    }
    assert "Respond only with the letter" in items[0]["prompt"]
    assert all(item["video_path"].endswith(".mp4") for item in items)


def test_official_scoring_rules():
    assert official_correct("EPM", "The answer is C.", "C")
    assert not official_correct("EPM", None, "C")
    assert official_correct("REC", "There were 1 and 2 actions", "12")
    assert not official_correct("REC", "12.0", "12")
    assert official_correct("SSR", "Y", "Yes")
    assert official_correct("CRR", "No", "No")
    assert not official_correct("SSR", "Maybe", "Yes")


def test_equal_task_and_group_averaging():
    rows = [
        {"task": "EPM", "correct": True}, {"task": "EPM", "correct": False},
        {"task": "HLD", "correct": True}, {"task": "STU", "correct": False},
        {"task": "REC", "correct": True},
    ]
    result = summary(rows, len(rows))
    assert result["group_accuracy"]["backward"] == .75
    assert result["official_total_average"] == (.75 + 0 + 1) / 3


def test_hermes_nine_task_annotation_and_letter_scoring():
    items = hermes_queries()
    assert len(items) == 1468
    assert [int(item["query_id"]) for item in items] == list(range(1468))
    assert {item["task"] for item in items} == {
        "EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD"
    }
    assert "(A)" in items[0]["prompt"]
    # HERMES has a duplicate answer string at ID 1303 and scores the first letter.
    assert items[1303]["target"] == "A"
    assert queries()[1303]["target"] == "D"
    assert hermes_correct("Best option: (A)", "A")
    assert not hermes_correct("D)", "A")
