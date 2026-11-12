import subprocess
import sys


def test_core_import_skips_emulators() -> None:
    code = (
        "import sys, cosmograd; "
        "assert not {'cosmopower_jax', 'baccoemu', 'cosmograd.perturbations.baccoemu', 'cosmograd.perturbations.cosmopower'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
