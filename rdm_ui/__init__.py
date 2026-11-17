"""Presentation entry point sharing the current analysis and native dashboard."""
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from runpy import run_path

import streamlit as st


@contextmanager
def readable_layout() -> Generator[None, None, None]:
    """Scope the CSS to a public container key and preserve native widgets."""
    st.html(Path(__file__).with_name("readability.css"))
    with st.container(key="rdm_readability"):
        yield


def run_dashboard() -> None:
    """Run the current dashboard on every rerun; never copy analysis or widgets.

    The dashboard owns its result-file cache, Korean model/feature labels,
    English metric terms and session state. This layer only adds readability.
    """
    dashboard = Path(__file__).with_name("dashboard.py")
    with readable_layout():
        run_path(str(dashboard), run_name="__main__")
