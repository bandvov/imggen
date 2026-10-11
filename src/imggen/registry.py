"""Name -> implementation registries, one per component kind.

Adding a component = one new module that decorates its class with
``@KIND.register("name")``. ``discover()`` imports every module under the
component subpackages, so no central file needs editing.
"""

from __future__ import annotations

import difflib
import importlib
import pkgutil
import re
from collections.abc import Callable, Iterator
from typing import Any, Generic, TypeVar

T = TypeVar("T")
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")


class RegistryError(KeyError):
    """Raised when a name is not registered."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0]) if self.args else ""


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def add(self, name: str, obj: T) -> T:
        if not _NAME_RE.match(name):
            raise ValueError(f"invalid {self.kind} name {name!r}: use lowercase [a-z0-9_-]")
        existing = self._items.get(name)
        if existing is not None and existing is not obj:
            raise ValueError(f"{self.kind} {name!r} already registered as {existing!r}")
        self._items[name] = obj  # re-registering the same object is a no-op
        return obj

    def register(self, name: str) -> Callable[[T], T]:
        def decorator(obj: T) -> T:
            return self.add(name, obj)

        return decorator

    def unregister(self, name: str) -> None:
        self._items.pop(name, None)

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            msg = f"unknown {self.kind} {name!r}"
            close = difflib.get_close_matches(name, self._items, n=3)
            if close:
                msg += f"; did you mean {close}?"
            msg += f" (available: {self.names() or 'none'})"
            raise RegistryError(msg) from None

    def build(self, name: str, /, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)  # type: ignore[operator]

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}, {self.names()})"


ATTENTION: Registry = Registry("attention")
FFN: Registry = Registry("ffn")
BACKBONE: Registry = Registry("backbone")
OBJECTIVE: Registry = Registry("objective")
AUTOENCODER: Registry = Registry("autoencoder")
TEXT_ENCODER: Registry = Registry("text_encoder")
SAMPLER: Registry = Registry("sampler")
DATASET: Registry = Registry("dataset")
METRIC: Registry = Registry("metric")

REGISTRIES: dict[str, Registry] = {
    r.kind: r
    for r in (
        ATTENTION,
        FFN,
        BACKBONE,
        OBJECTIVE,
        AUTOENCODER,
        TEXT_ENCODER,
        SAMPLER,
        DATASET,
        METRIC,
    )
}

DISCOVERY_PACKAGES = (
    "layers",
    "backbones",
    "objectives",
    "autoencoders",
    "text",
    "samplers",
    "data",
    "eval",
)


def discover(packages: tuple[str, ...] | None = None, base: str = "imggen") -> list[str]:
    """Import every module under the component subpackages so they self-register."""
    imported: list[str] = []
    for sub in packages if packages is not None else DISCOVERY_PACKAGES:
        pkg = importlib.import_module(f"{base}.{sub}")
        for info in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg.__name__}."):
            importlib.import_module(info.name)
            imported.append(info.name)
    return imported
