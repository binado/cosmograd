"""A user-level cache for the bacco network weights.

``baccoemu`` downloads its weights into its own install directory on first use,
which is lost whenever the environment is rebuilt, is not writable on every
system, and races when several processes start at once. Here the weights are
instead fetched once into the platform's user cache directory and symlinked
into the folder where ``baccoemu`` looks for them. ``baccoemu`` finds them
present and never downloads.

Downloads are checked against pinned SHA-256 digests. The ``bacco.dipc.org``
certificate chain is incomplete, so when verification fails the download is
retried without it and the digest is what vouches for the file.
"""

import hashlib
import logging
import shutil
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path

import platformdirs

__all__ = [
    "BASE_URL",
    "CHECKSUMS",
    "NONLINEAR_EMULATORS",
    "cache_dir",
    "fetch_emulators",
]

BASE_URL = "https://bacco.dipc.org"
"""Host serving the ``<name>.tar`` archives."""

NONLINEAR_EMULATORS = {
    "Angulo2021": "nonlinear_emu_1.0.3",
    "Arico2023": "nonlinear_emu_2.0.1",
}
"""Directory name of the nonlinear emulator of each model."""

CHECKSUMS = {
    "cold_matter_linear_emu_1.0.2": (
        "566446c943207983967a084277bdf930c1121e4691fb6a3beecea49f278fcce9"
    ),
    "total_matter_linear_emu_1.0.2": (
        "a75e35d534af01c3ea547f63dfe30a659d1476f1b2896db7c42bfb1bfc624690"
    ),
    "sigma8_emu_2.0.2": (
        "de667c59a09ed8b89b0518f1c208eb547bb2ceaf2a7e312eefa89db80020d949"
    ),
    "no_wiggles_emu_1.0.2": (
        "2a2e1038ed6ba18d351c1f8e3df000b02a3af5c24a76c141a5a39ffece520597"
    ),
    "nonlinear_emu_1.0.3": (
        "30568f0f6ec138854d140813b8672660559761a7fedda17ef324df661ced018b"
    ),
    "nonlinear_emu_2.0.1": (
        "e6262ad1db34ba18f7da2752901ec095aa9adc668f01ce7806f1b2e95f8d2ec5"
    ),
}
"""SHA-256 of each ``<name>.tar`` archive."""

_TIMEOUT = 60.0

logger = logging.getLogger(__name__)


def cache_dir() -> Path:
    """Directory holding the cached weights.

    Returns
    -------
    Path
        ``baccoemu`` inside the user cache directory of ``cosmograd``, such as
        ``~/.cache/cosmograd/baccoemu`` on Linux (honouring ``XDG_CACHE_HOME``)
        or ``~/Library/Caches/cosmograd/baccoemu`` on macOS.
    """
    return Path(platformdirs.user_cache_dir("cosmograd")) / "baccoemu"


def fetch_emulators(
    names: Iterable[str],
    target_dir: Path,
    *,
    cache: Path | None = None,
    base_url: str = BASE_URL,
    checksums: Mapping[str, str] = CHECKSUMS,
) -> None:
    """Make emulator directories available in ``target_dir`` without ``baccoemu``
    having to download them.

    Each missing emulator is downloaded into the cache, then symlinked into
    ``target_dir``. Existing entries of ``target_dir`` are left alone, so a
    previous download by ``baccoemu`` itself is used as is. Safe to call from
    several processes at once.

    Parameters
    ----------
    names : iterable of str
        Emulator directory names, keys of ``checksums``.
    target_dir : Path
        Directory ``baccoemu`` reads the emulators from.
    cache : Path, optional
        Where to keep the downloads. Defaults to `cache_dir`.
    base_url : str
        Location of the ``<name>.tar`` archives.
    checksums : mapping of str to str
        SHA-256 hex digest of each archive.

    Raises
    ------
    ValueError
        If a downloaded archive does not match its digest or has an unexpected
        layout.
    urllib.error.URLError
        If a download fails.

    Warns
    -----
    UserWarning
        If ``target_dir`` cannot be linked into, in which case ``baccoemu``
        downloads the emulator itself.
    """
    cache = cache_dir() if cache is None else cache
    for name in names:
        link = target_dir / name
        if link.exists():
            continue
        source = cache / name
        if not source.is_dir():
            _download(name, source, f"{base_url}/{name}.tar", checksums[name])
        _link(link, source)


def _link(link: Path, source: Path) -> None:
    """Symlink ``link`` to ``source``, replacing a dangling link."""
    try:
        if link.is_symlink():
            link.unlink()  # dangling: ``exists`` follows links
        link.symlink_to(source, target_is_directory=True)
    except FileExistsError:
        pass  # another process got there first
    except OSError as err:
        warnings.warn(
            f"Could not link {source} into {link.parent} ({err}); "
            "'baccoemu' will download the emulator itself.",
            stacklevel=3,
        )


def _download(name: str, destination: Path, url: str, sha256: str) -> None:
    """Download and unpack one archive to ``destination``, atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading bacco emulator %s to %s", name, destination.parent)
    # Staging inside the cache keeps the final rename on one filesystem.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".tmp-") as tmp:
        archive = Path(tmp) / f"{name}.tar"
        _retrieve(url, archive)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"{url} has SHA-256 {digest}, expected {sha256}.")
        unpacked = Path(tmp) / "unpacked"
        with tarfile.open(archive) as tar:
            tar.extractall(unpacked, filter="data")
        if not (unpacked / name).is_dir():
            raise ValueError(f"{url} does not contain a '{name}' directory.")
        try:
            (unpacked / name).rename(destination)
        except OSError:
            # Another process finished first; theirs is identical.
            if not destination.is_dir():
                raise


def _retrieve(url: str, path: Path) -> None:
    """Save ``url`` to ``path``, tolerating an unverifiable certificate chain."""
    try:
        _save(url, path, ssl.create_default_context())
    except urllib.error.URLError as err:
        if not isinstance(err.reason, ssl.SSLCertVerificationError):
            raise
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        _save(url, path, context)


def _save(url: str, path: Path, context: ssl.SSLContext) -> None:
    # A private opener: ``urlopen`` would build and cache a global one with
    # verification on, which makes the unverified context that ``baccoemu``
    # installs for its own downloads (see the module docstring) have no effect.
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=context))
    with opener.open(url, timeout=_TIMEOUT) as response:
        with path.open("wb") as out:
            shutil.copyfileobj(response, out)
