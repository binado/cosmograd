"""Emulator backends, each behind an optional extra.

Nothing here is imported by ``import cosmograd``. Import a backend module
explicitly; its emulator package is imported on first use, and a missing one
raises an :class:`ImportError` naming the extra to install.
"""
