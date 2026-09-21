"""Tests for compiled selector caching and per-call scope isolation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import random

import bs4
import pytest

import soupsieve as sv
from soupsieve import css_parser as cp


@dataclass(frozen=True)
class Context:
    """A self-contained selector context."""

    name: str
    mode: str
    root: bs4.Tag
    namespaces: dict[str, str]
    custom: dict[str, str]
    target_id: str
    marker_id: str
    feature_selector: str
    scope_selector: str


def xml_tag(name, namespace, *children, attrs=None):
    """Create a detached XML tag without depending on an installed XML parser."""

    tag = bs4.Tag(
        name=name, namespace=namespace, prefix="ns", is_xml=True, attrs=attrs or {}
    )
    for child in children:
        tag.append(child)
    return tag


def html_tag(name, *children, attrs=None):
    """Create a detached HTML tag."""

    tag = bs4.Tag(name=name, is_xml=False, attrs=attrs or {})
    for child in children:
        tag.append(child)
    return tag


def make_context(name, mode):
    """Build two contexts that deliberately reuse every public selector name."""

    other = "beta" if name == "alpha" else "alpha"
    target_id = f"{name}-target"
    marker_id = f"{name}-marker"
    own_anchor = f"anchor-{name}"
    other_anchor = f"anchor-{other}"

    feature_selector = (
        "item:--target:--nested:has(> marker)"
        ":nth-child(2 of item.candidate)"
    )
    scope_selector = ":scope > item"
    custom = {
        ":--target": f".{own_anchor}",
        ":--nested": ":--anchor[kind]",
        ":--anchor": f".{own_anchor}",
    }

    if mode == "html":
        target = html_tag(
            "item",
            html_tag("marker", attrs={"id": marker_id}),
            attrs={"id": target_id, "class": ["candidate", own_anchor], "kind": "yes"},
        )
        decoy = html_tag(
            "item",
            html_tag("marker", attrs={"id": f"{other}-marker"}),
            attrs={"id": f"{other}-target", "class": ["candidate", other_anchor], "kind": "no"},
        )
        root = html_tag("root", decoy, target, attrs={"id": "root"})
        namespaces = {"shared": f"urn:{name}", "": "http://www.w3.org/1999/xhtml"}
    else:
        namespace = f"urn:{name}"
        target = xml_tag(
            "item",
            namespace,
            xml_tag("marker", namespace, attrs={"id": marker_id}),
            attrs={"id": target_id, "class": f"candidate {own_anchor}", "kind": "yes"},
        )
        decoy = xml_tag(
            "item",
            namespace,
            xml_tag("marker", namespace, attrs={"id": f"{other}-marker"}),
            attrs={"id": f"{other}-target", "class": f"candidate {other_anchor}", "kind": "no"},
        )
        root = xml_tag("root", namespace, decoy, target, attrs={"id": "root"})
        namespaces = {"shared": namespace, "": namespace}

    return Context(
        name=name,
        mode=mode,
        root=root,
        namespaces=namespaces,
        custom=custom,
        target_id=target_id,
        marker_id=marker_id,
        feature_selector=feature_selector,
        scope_selector=scope_selector,
    )


def operation_calls(context):
    """Return calls covering the complete public selector workflow."""

    return [
        ("compile", context.feature_selector, lambda: sv.compile(
            context.feature_selector, namespaces=context.namespaces, custom=context.custom
        )),
        ("select", context.feature_selector, lambda: sv.select(
            context.feature_selector, tag=context.root, namespaces=context.namespaces,
            limit=1, custom=context.custom
        )),
        ("iselect", context.feature_selector, lambda: list(sv.iselect(
            context.feature_selector, tag=context.root, namespaces=context.namespaces,
            limit=1, custom=context.custom
        ))),
        ("select_one", context.feature_selector, lambda: sv.select_one(
            context.feature_selector, context.root,
            namespaces=context.namespaces, custom=context.custom
        )),
        ("match", context.feature_selector, lambda: sv.match(
            context.feature_selector, context.root.find(id=context.target_id),
            namespaces=context.namespaces, custom=context.custom
        )),
        ("closest", context.feature_selector, lambda: sv.closest(
            context.feature_selector, context.root.find(id=context.marker_id),
            namespaces=context.namespaces, custom=context.custom
        )),
        ("filter", context.feature_selector, lambda: sv.filter(
            context.feature_selector,
            iterable=[
                context.root.find(id=context.marker_id),
                context.root.find(id=context.target_id),
            ],
            namespaces=context.namespaces, custom=context.custom
        )),
        ("scope", context.scope_selector, lambda: sv.select(
            context.scope_selector, context.root,
            namespaces=context.namespaces, custom=context.custom
        )),
    ]


def assert_operation_result(context, operation, selector, result, sequence):
    """Assert one operation and include enough context to locate a leak."""

    prefix = f"context={context.name} mode={context.mode} operation={operation} selector={selector!r}"
    failure_context = f"{prefix}\ncall sequence:\n" + "\n".join(sequence)

    if operation == "compile":
        assert result.pattern == selector, failure_context
    elif operation in ("select", "iselect"):
        assert [tag.attrs["id"] for tag in result] == [context.target_id], failure_context
    elif operation == "select_one":
        assert result is not None and result.attrs["id"] == context.target_id, failure_context
    elif operation == "match":
        assert result is True, failure_context
    elif operation == "closest":
        assert result is not None and result.attrs["id"] == context.target_id, failure_context
    elif operation == "filter":
        assert [tag.attrs["id"] for tag in result] == [context.target_id], failure_context
    else:
        decoy = "beta-target" if context.name == "alpha" else "alpha-target"
        assert [tag.attrs["id"] for tag in result] == [decoy, context.target_id], failure_context


def run_context(context, order, seed=None):
    """Run one context's operations and report every executed call."""

    calls = operation_calls(context)
    if order == "reverse":
        calls = list(reversed(calls))
    elif order == "random":
        random.Random(seed).shuffle(calls)

    sequence = []
    for index, (operation, selector, call) in enumerate(calls):
        sequence.append(f"{index}: {context.name}:{operation}:{selector}")
        assert_operation_result(context, operation, selector, call(), sequence)
    return sequence


@pytest.fixture(autouse=True)
def clear_selector_cache():
    """Keep cache accounting local to each test."""

    sv.purge()
    yield
    sv.purge()


@pytest.mark.parametrize("mode", ["html", "xml"])
@pytest.mark.parametrize("order", ["forward", "reverse"])
def test_forward_and_reverse_calls_isolate_contexts(mode, order):
    """Reused selector names and mappings must not leak across separate trees."""

    for name in ("alpha", "beta"):
        run_context(make_context(name, mode), order)


@pytest.mark.parametrize("mode", ["html", "xml"])
@pytest.mark.parametrize("seed", [17, 43, 91])
def test_randomized_parallel_contexts_do_not_share_dynamic_state(mode, seed):
    """Run randomized call sequences concurrently on independent contexts."""

    contexts = [make_context("alpha", mode), make_context("beta", mode)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_context, context, "random", seed + index)
            for index, context in enumerate(contexts)
        ]
        sequences = [future.result() for future in futures]

    for context, sequence in zip(contexts, sequences):
        selector = context.feature_selector
        assert sv.select(
            selector, context.root, limit=1,
            namespaces=context.namespaces, custom=context.custom
        )[0].attrs["id"] == context.target_id, (
            f"selector {selector!r} leaked from another context\n"
            f"call sequence:\n" + "\n".join(sequence)
        )


@pytest.mark.parametrize("mode", ["html", "xml"])
def test_equal_mappings_reuse_cache_despite_insertion_order(mode):
    """Mapping equality, not object identity or insertion order, is the cache key."""

    alpha = make_context("alpha", mode)
    beta = make_context("beta", mode)
    beta.namespaces.clear()
    beta.namespaces.update(reversed(list(alpha.namespaces.items())))
    beta.custom.clear()
    beta.custom.update(reversed(list(alpha.custom.items())))

    first = sv.compile(alpha.feature_selector, alpha.namespaces, custom=alpha.custom)
    assert cp._cached_css_compile.cache_info().misses == 1

    second = sv.compile(beta.feature_selector, beta.namespaces, custom=beta.custom)
    assert cp._cached_css_compile.cache_info().hits == 1
    assert second is first
    assert second.selectors is first.selectors

    assert [tag.attrs["id"] for tag in second.select(alpha.root, limit=1)] == [alpha.target_id]
    beta_selector = sv.compile(
        beta.feature_selector, beta.namespaces, custom=beta.custom
    )
    assert beta_selector is second


@pytest.mark.parametrize("mode", ["html", "xml"])
def test_changed_mapping_content_is_not_reused_with_same_object_identity(mode):
    """Mutating a caller-owned mapping after a compile changes the next cache key."""

    alpha = make_context("alpha", mode)
    beta = make_context("beta", mode)
    namespaces = dict(alpha.namespaces)
    custom = dict(alpha.custom)
    selector = alpha.feature_selector

    old = sv.compile(selector, namespaces, custom=custom)
    namespaces["shared"] = beta.namespaces["shared"]
    namespaces[""] = beta.namespaces[""]
    custom[":--target"] = beta.custom[":--target"]
    custom[":--anchor"] = beta.custom[":--anchor"]

    new = sv.compile(selector, namespaces, custom=custom)
    assert new is not old
    assert cp._cached_css_compile.cache_info().misses == 2
    assert [tag.attrs["id"] for tag in old.select(alpha.root)] == [alpha.target_id]
    assert [tag.attrs["id"] for tag in new.select(beta.root)] == [beta.target_id]


def test_xml_prefix_rebinding_is_not_reused_with_same_dict_identity():
    """Rebinding a prefix to another URI changes the explicit prefix selector."""

    alpha = make_context("alpha", "xml")
    beta = make_context("beta", "xml")
    namespaces = {"shared": alpha.namespaces["shared"]}
    custom = dict(alpha.custom)
    selector = "shared|item:--target:--nested"

    old = sv.compile(selector, namespaces, custom=custom)
    assert old.match(alpha.root.find(id=alpha.target_id))

    namespaces["shared"] = beta.namespaces["shared"]
    custom[":--target"] = beta.custom[":--target"]
    custom[":--anchor"] = beta.custom[":--anchor"]
    new = sv.compile(selector, namespaces, custom=custom)

    assert new is not old
    assert new.match(beta.root.find(id=beta.target_id))
    assert not new.match(alpha.root.find(id=alpha.target_id))
    assert old.match(alpha.root.find(id=alpha.target_id))


@pytest.mark.parametrize("mode", ["html", "xml"])
def test_compiled_selectors_keep_immutable_mapping_snapshots(mode):
    """Later caller mutations cannot rewrite an already compiled selector."""

    context = make_context("alpha", mode)
    baseline = make_context("alpha", mode)
    namespaces = dict(context.namespaces)
    custom = dict(context.custom)
    selector = context.feature_selector

    compiled = sv.compile(selector, namespaces, custom=custom)
    namespaces.clear()
    custom.clear()

    assert dict(compiled.namespaces) == baseline.namespaces
    assert dict(compiled.custom) == baseline.custom
    assert [tag.attrs["id"] for tag in compiled.select(context.root)] == [context.target_id]


@pytest.mark.parametrize("mode", ["html", "xml"])
def test_scope_and_limit_are_per_call_but_still_use_the_cache(mode):
    """Scope roots and limits affect matching but never become global cache keys."""

    alpha = make_context("alpha", mode)
    beta = make_context("beta", mode)
    compiled = sv.compile(alpha.feature_selector, alpha.namespaces, custom=alpha.custom)
    assert cp._cached_css_compile.cache_info().misses == 1

    beta_compiled = sv.compile(beta.feature_selector, beta.namespaces, custom=beta.custom)
    assert cp._cached_css_compile.cache_info().misses == 2
    assert [tag.attrs["id"] for tag in compiled.select(alpha.root, limit=1)] == [alpha.target_id]
    assert [
        tag.attrs["id"] for tag in list(compiled.iselect(alpha.root, limit=1))
    ] == [alpha.target_id]
    assert not compiled.match(alpha.root)
    assert compiled.closest(alpha.root.find(id=alpha.marker_id)).attrs["id"] == alpha.target_id
    assert [
        tag.attrs["id"] for tag in compiled.filter(
            iterable=[alpha.root.find(id=alpha.marker_id), alpha.root.find(id=alpha.target_id)]
        )
    ] == [alpha.target_id]

    assert [
        tag.attrs["id"] for tag in beta_compiled.select(beta.root, limit=1)
    ] == [beta.target_id]

    sv.select(
        alpha.feature_selector, beta.root, limit=0,
        namespaces=alpha.namespaces, custom=alpha.custom
    )
    assert cp._cached_css_compile.cache_info().hits == 1


def test_flags_are_part_of_the_compile_cache_key():
    """A flag that changes parser diagnostics must not return the other compilation."""

    selector = "item:--target"
    namespaces = {"": "urn:flags", "shared": "urn:flags"}
    custom = {":--target": ".anchor-flags"}

    normal = sv.compile(selector, namespaces, custom=custom)
    debug = sv.compile(selector, namespaces, flags=sv.DEBUG, custom=custom)

    assert normal is not debug
    assert normal.flags == 0
    assert debug.flags == sv.DEBUG
    assert cp._cached_css_compile.cache_info().misses == 2


def test_patterns_are_part_of_the_compile_cache_key():
    """Different selector syntax remains separately compiled."""

    namespaces = {"": "http://www.w3.org/1999/xhtml"}

    item = sv.compile("item", namespaces)
    marker = sv.compile("marker", namespaces)

    assert marker is not item
    assert marker.selectors is not item.selectors
    assert cp._cached_css_compile.cache_info().misses == 2
