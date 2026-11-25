"""Synthetic smoke tests for the frozen Task3 mixture experiment helpers."""

import numpy as np

import run as experiment


def main():
    scores = np.asarray([[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        experiment.stable_topk(scores, 2),
        np.asarray([[0, 1], [1, 2]], dtype=np.int64),
    )

    images = np.zeros((3, 64, 64), dtype=np.float32)
    images[1] = 1.0
    images[2, :32] = 1.0
    ranked = experiment.stable_topk(scores, 3)
    ranked_scores = np.take_along_axis(scores, ranked, axis=1)
    tied = experiment.render_mixtures(
        images,
        ranked[:1],
        ranked_scores[:1],
        k=2,
        temperature=1.0,
        threshold=0.5,
    )
    np.testing.assert_array_equal(tied[0], np.ones((64, 64), dtype=np.float32))
    tied_strict = experiment.render_mixtures(
        images,
        ranked[:1],
        ranked_scores[:1],
        k=2,
        temperature=1.0,
        threshold=0.75,
    )
    np.testing.assert_array_equal(tied_strict[0], np.zeros((64, 64), dtype=np.float32))

    rows = [
        {"tune": {"ocf_mean": 1.0}},
        {"tune": {"ocf_mean": 2.0}},
        {"tune": {"ocf_mean": 2.0}},
    ]
    assert experiment.select_locked_index(rows) == 1
    decision = experiment.advancement(
        {"mice1": 0.1, "mice2": 0.2},
        {f"group{index}": value for index, value in enumerate([1, 1, 1, 1, 1, -1, -1, -1])},
    )
    assert decision["advance"] and decision["positive_group_count"] == 5
    assert not experiment.advancement(
        {"mice1": 0.0, "mice2": 0.2},
        {f"group{index}": 1.0 for index in range(8)},
    )["advance"]
    assert not experiment.advancement(
        {"mice1": 0.1, "mice2": 0.2},
        {f"group{index}": value for index, value in enumerate([1, 1, 1, 1, -1, -1, -1, -1])},
    )["advance"]

    rng = np.random.default_rng(7)
    targets = (rng.random((2, 64, 64)) >= 0.5).astype(np.float32)
    predictions = np.stack([targets[0], 1.0 - targets[1]])
    official = experiment.official_scores(predictions, targets, "cpu", 2)
    assert official.shape == (2,) and np.isfinite(official).all()
    print("Synthetic smoke tests passed; official OCF outputs:", official.tolist())


if __name__ == "__main__":
    main()
