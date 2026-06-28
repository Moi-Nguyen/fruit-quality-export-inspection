"""Smoke tests for the Tkinter GUI module."""

from __future__ import annotations


def test_gui_app_can_be_imported() -> None:
    import src.gui_app as gui_app

    assert gui_app is not None


def test_run_gui_exists() -> None:
    from src.gui_app import run_gui

    assert callable(run_gui)
