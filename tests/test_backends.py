import subprocess
import sys


def test_core_import_skips_emulators() -> None:
    code = (
        "import sys, cosmograd; "
        "assert not {'cosmopower_jax', 'baccoemu', 'cosmograd.perturbations.baccoemu', 'cosmograd.perturbations.cosmopower'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cosmopower_module_imports_lazily() -> None:
    code = (
        "import sys; from cosmograd import FlatLCDM; "
        "from cosmograd.perturbations.cosmopower import CosmoPower; "
        "CosmoPower(FlatLCDM(H0=67.0, omega_m=0.31, omega_b=0.049, n_s=0.965, ln10as=3.044)); "
        "assert 'cosmopower_jax' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_baccoemu_backend_imports_baccoemu_lazily() -> None:
    code = (
        "import sys; from cosmograd.perturbations.baccoemu import BaccoEmu; "
        "assert 'baccoemu' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
