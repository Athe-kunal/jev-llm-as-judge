"""Prompt templates (Jinja) for the judge and the rewriter."""

import pathlib
from typing import Any

import jinja2

_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(pathlib.Path(__file__).parent / "prompts"),
    undefined=jinja2.StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    autoescape=False,
)


def render(name: str, **values: Any) -> str:
    """Renders prompts/<name>.j2 with the given values.

    Args:
      name: Template name without the extension.
      **values: Template variables; a missing one raises an error.

    Returns:
      The prompt text, without a trailing newline.
    """
    return _ENV.get_template(f"{name}.j2").render(**values).rstrip("\n")


def source(name: str) -> str:
    """Returns the raw text of prompts/<name>.j2 (for the viewer)."""
    path = pathlib.Path(__file__).parent / "prompts" / f"{name}.j2"
    return path.read_text()
