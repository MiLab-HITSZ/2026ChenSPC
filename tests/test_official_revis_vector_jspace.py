from analyze_official_revis_vector_jspace import _target_records, _token_record


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"yes": [1], " no": [2, 3]}[text]

    def convert_ids_to_tokens(self, token_id):
        return f"token-{token_id}"

    def decode(self, token_ids):
        return ":".join(str(value) for value in token_ids)


def test_token_record_is_json_ready():
    record = _token_record(FakeTokenizer(), 2, -1.25)
    assert record == {
        "token_id": 2,
        "token": "token-2",
        "decoded": "2",
        "score": -1.25,
    }


def test_target_records_preserve_text_and_average_scores():
    records = _target_records(FakeTokenizer(), [0.0, 2.0, -1.0, 3.0], ["yes", " no", "yes"])
    assert records == [
        {
            "text": "yes",
            "token_ids": [1],
            "tokens": ["token-1"],
            "scores": [2.0],
            "mean_score": 2.0,
        },
        {
            "text": " no",
            "token_ids": [2, 3],
            "tokens": ["token-2", "token-3"],
            "scores": [-1.0, 3.0],
            "mean_score": 1.0,
        },
    ]
