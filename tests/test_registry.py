import sys

import pytest

from imggen.registry import REGISTRIES, Registry, RegistryError, discover


def test_register_get_build():
    reg = Registry("thing")

    @reg.register("a")
    class A:
        def __init__(self, x=1):
            self.x = x

    assert reg.get("a") is A
    assert reg.build("a", x=5).x == 5
    assert "a" in reg and len(reg) == 1 and reg.names() == ["a"]


def test_duplicate_different_object_rejected_same_object_ok():
    reg = Registry("thing")
    first = reg.add("a", object())
    reg.add("a", first)  # idempotent (module re-import)
    with pytest.raises(ValueError, match="already registered"):
        reg.add("a", object())


def test_unknown_name_suggests_close_match():
    reg = Registry("attention")
    reg.add("softmax", object())
    with pytest.raises(RegistryError, match="did you mean.*softmax"):
        reg.get("softmaks")


@pytest.mark.parametrize("bad", ["", "Upper", "has space", "-lead"])
def test_invalid_names_rejected(bad):
    with pytest.raises(ValueError, match="invalid"):
        Registry("x").add(bad, object())


def test_build_allows_name_kwarg():
    reg = Registry("x")
    reg.add("a", lambda **kw: kw)
    assert reg.build("a", name="n") == {"name": "n"}


def test_global_registries_are_distinct_and_keyed_by_kind():
    assert len({id(r) for r in REGISTRIES.values()}) == len(REGISTRIES)
    assert all(k == r.kind for k, r in REGISTRIES.items())


def test_discover_skeleton_imports_cleanly():
    discover()


def test_discover_registers_from_new_module(tmp_path, monkeypatch):
    pkg = tmp_path / "fakepkg"
    (pkg / "layers").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "layers" / "__init__.py").write_text("")
    (pkg / "layers" / "dummy.py").write_text(
        "from imggen.registry import ATTENTION\n"
        "@ATTENTION.register('dummy-attn')\n"
        "class Dummy: pass\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from imggen.registry import ATTENTION

    try:
        imported = discover(packages=("layers",), base="fakepkg")
        assert imported == ["fakepkg.layers.dummy"]
        assert "dummy-attn" in ATTENTION
    finally:
        ATTENTION.unregister("dummy-attn")
        sys.modules.pop("fakepkg.layers.dummy", None)
