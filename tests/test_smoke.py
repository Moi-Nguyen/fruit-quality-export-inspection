"""Smoke tests for the project skeleton."""


def test_project_imports() -> None:
    """Verify that the main package can be imported."""
    import src

    assert src is not None
