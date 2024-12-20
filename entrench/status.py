import contextlib
import contextvars
import sys
from typing import Generator

import rich
import rich.live
import rich.spinner

_STATUS_VAR: contextvars.ContextVar["Status"] = contextvars.ContextVar("status")


class Status:
    def __init__(self):
        self._spinner = rich.spinner.Spinner("dots", text="Working...")

    @staticmethod
    def get() -> "Status":
        return _STATUS_VAR.get()

    def update(self, status: str) -> None:
        self._spinner.update(text=status)


@contextlib.contextmanager
def make_status() -> Generator[Status, None, None]:
    status = Status()

    with rich.live.Live(status._spinner, console=rich.console.Console(file=sys.stderr)):
        token = _STATUS_VAR.set(status)
        yield status
        _STATUS_VAR.reset(token)
