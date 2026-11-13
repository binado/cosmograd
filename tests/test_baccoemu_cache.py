"""The user-level cache of bacco weights, served from local archives."""

import hashlib
import tarfile
from pathlib import Path

import pytest

from cosmograd.perturbations.baccoemu._cache import fetch_emulators

NAME = "toy_emu_1.0.0"


@pytest.fixture
def archive_dir(tmp_path: Path) -> Path:
    """A directory holding ``toy_emu_1.0.0.tar`` as the bacco host would."""
    source = tmp_path / "build" / NAME
    source.mkdir(parents=True)
    (source / "details.pickle").write_bytes(b"details")
    served = tmp_path / "served"
    served.mkdir()
    with tarfile.open(served / f"{NAME}.tar", "w") as tar:
        tar.add(source, arcname=NAME)
    return served


@pytest.fixture
def checksums(archive_dir: Path) -> dict[str, str]:
    digest = hashlib.sha256((archive_dir / f"{NAME}.tar").read_bytes()).hexdigest()
    return {NAME: digest}


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """The cache and the directory ``baccoemu`` would read from."""
    target = tmp_path / "site-packages"
    target.mkdir()
    return tmp_path / "cache", target


def test_downloads_into_cache_and_links(
    archive_dir: Path, checksums: dict[str, str], dirs: tuple[Path, Path]
) -> None:
    cache, target = dirs
    fetch_emulators(
        [NAME], target, cache=cache, base_url=archive_dir.as_uri(), checksums=checksums
    )
    assert (cache / NAME / "details.pickle").read_bytes() == b"details"
    assert (target / NAME).is_symlink()
    assert (target / NAME / "details.pickle").read_bytes() == b"details"
    assert not [path for path in cache.iterdir() if path.name.startswith(".tmp-")]


def test_relinks_from_cache_without_downloading(
    archive_dir: Path, checksums: dict[str, str], dirs: tuple[Path, Path]
) -> None:
    """A rebuilt environment loses the links, not the weights."""
    cache, target = dirs
    fetch_emulators(
        [NAME], target, cache=cache, base_url=archive_dir.as_uri(), checksums=checksums
    )
    (target / NAME).unlink()
    fetch_emulators(
        [NAME], target, cache=cache, base_url="file:///nowhere", checksums=checksums
    )
    assert (target / NAME / "details.pickle").is_file()


def test_keeps_existing_install(
    checksums: dict[str, str], dirs: tuple[Path, Path]
) -> None:
    """Weights ``baccoemu`` already holds are used as they are."""
    cache, target = dirs
    (target / NAME).mkdir()
    fetch_emulators(
        [NAME], target, cache=cache, base_url="file:///nowhere", checksums=checksums
    )
    assert not (target / NAME).is_symlink()
    assert not cache.exists()


def test_rejects_mismatched_checksum(
    archive_dir: Path, dirs: tuple[Path, Path]
) -> None:
    cache, target = dirs
    with pytest.raises(ValueError, match="SHA-256"):
        fetch_emulators(
            [NAME],
            target,
            cache=cache,
            base_url=archive_dir.as_uri(),
            checksums={NAME: "0" * 64},
        )
    assert not (cache / NAME).exists()
    assert not (target / NAME).exists()
