from train import VARIANTS


def test_experiment_matrix_is_explicit_and_complete():
    assert list(VARIANTS) == [
        "ppo",
        "icm",
        "rnd",
        "pc",
        "planner",
        "icm_planner",
    ]
