"""Access to the perturbation parameters stored on a cosmology."""

__all__ = ["require"]


def require(cosmology: object, *names: str) -> None:
    """Raise if any of the optional parameters ``names`` is unset.

    Call it when evaluating a spectrum rather than at construction: JAX
    rebuilds pytrees with placeholder leaves, including ``None``, in
    transforms.

    Parameters
    ----------
    cosmology : object
        A cosmology such as :class:`~cosmograd.FlatLCDM`.
    *names : str
        Field names, e.g. ``"omega_b"``.

    Raises
    ------
    ValueError
        Listing the fields that are ``None``.
    """
    missing = [name for name in names if getattr(cosmology, name) is None]
    if missing:
        raise ValueError(
            f"{type(cosmology).__name__} does not set {', '.join(missing)}, "
            "which this perturbations backend needs."
        )
