import subprocess
import sys

import pytest

from cosmograd.backends import _optional
from cosmograd.backends.bacco import load_baccoemu
from cosmograd.backends.cosmopower import load_cosmopower


def test_core_import_skips_emulators() -> None:
    code = (
        "import sys, cosmograd; "
        "assert not {'cosmopower_jax', 'baccoemu', 'cosmograd.backends'} & set(sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("loader", "module", "extra"),
    [
        (load_cosmopower, "cosmopower_jax", "cosmopower"),
        (load_baccoemu, "baccoemu", "bacco"),
    ],
)
def test_missing_extra_names_it(monkeypatch, loader, module: str, extra: str) -> None:
    real_import = _optional.importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == module:
            raise ModuleNotFoundError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(_optional.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match=rf"cosmograd\[{extra}\]"):
        loader()
