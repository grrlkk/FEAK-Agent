from scripts.prepare_corruption_human_review import _filter_accepted_operators


def test_filter_accepted_operators_disables_other_transitions_without_mutation():
    rows = [
        {"corruption_op": "DELETE_SPECIFICS", "accepted": True},
        {"corruption_op": "INSERT_OFFTOPIC", "accepted": True},
        {"corruption_op": "INSERT_OFFTOPIC", "accepted": False},
    ]

    filtered = _filter_accepted_operators(rows, {"INSERT_OFFTOPIC"})

    assert [row["accepted"] for row in filtered] == [False, True, False]
    assert rows[0]["accepted"] is True
