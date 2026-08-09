import os
import subprocess
import sys


def test_pseudocount_hash_is_stable_across_processes():
    script = (
        "import numpy as np; "
        "from src.pseudocount import PseudoCountExploration; "
        "state=np.arange(16,dtype=np.float32); "
        "print(PseudoCountExploration()._hash(state))"
    )
    outputs = []
    for hash_seed in ["1", "98765"]:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = hash_seed
        outputs.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                text=True,
                env=env,
            ).strip()
        )
    assert outputs[0] == outputs[1]
