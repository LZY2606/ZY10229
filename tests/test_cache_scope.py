"""
Tests for compile caching and per-call scope isolation.

The compile step of Soup Sieve turns a CSS selector into an immutable
`SelectorList` syntax tree and memoizes it.  Only inputs that change the
*compiled syntax* participate in the cache key:

* the pattern string,
* the namespace mapping,
* the custom selector mapping,
* the flags.

Everything else (the document node a call is scoped to, and `limit`) is
dynamic call context: a fresh `CSSMatch` is built per call against the
compiled, immutable object.  These tests prove the two layers do not bleed
into each other.
"""
from __future__ import annotations

import random
import threading

import bs4
import pytest

import soupsieve as sv
from soupsieve import css_parser as cp
from soupsieve import css_types as ct

try:
    from bs4.builder import LXMLTreeBuilderForXML  # noqa: F401
    LXML_PRESENT = True
except ImportError:  # pragma: no cover - depends on the optional lxml
    LXML_PRESENT = False


# --------------------------------------------------------------------------- #
# Two deliberately overlapping documents.
#
# Every tag name, id prefix, namespace prefix and custom pseudo-class name is
# reused between the contexts.  The *meaning* of the shared names is what
# differs, which is exactly what the cache/scope boundary must protect.
# --------------------------------------------------------------------------- #

DEFAULT_URI = 'urn:shared'

MARKUP_A_HTML = """
<root id="root-a">
  <group id="g-a">
    <item id="a-hot" class="hot"><leaf id="a-leaf"></leaf></item>
    <item id="a-cold" class="cold"></item>
  </group>
  <other id="o-a"></other>
</root>
"""

MARKUP_B_HTML = """
<root id="root-b">
  <group id="g-b">
    <item id="b-x"></item>
    <item id="b-hot" class="hot"></item>
    <item id="b-cold" class="cold"><leaf id="b-leaf"></leaf></item>
  </group>
  <other id="o-b"></other>
</root>
"""

# Identical custom pseudo-class names, intentionally different aliases.
CUSTOM_A = {
    ':--thing': 'item:has(leaf)',
    ':--deep': ':--thing',
}
CUSTOM_B = {
    ':--thing': 'item.hot',
    ':--deep': ':--thing',
}

NAMESPACES = {'': DEFAULT_URI, 'ns': DEFAULT_URI}

# Selectors shared by both contexts, with per-context expected matched ids
# when evaluated against each context's <group> scope node.
SELECTOR_CASES = [
    (':scope > item', ['a-hot', 'a-cold'], ['b-x', 'b-hot', 'b-cold']),
    ('& > item', ['a-hot', 'a-cold'], ['b-x', 'b-hot', 'b-cold']),
    ('item:has(leaf)', ['a-hot'], ['b-cold']),
    (':nth-child(2 of item)', ['a-cold'], ['b-hot']),
    (':--thing', ['a-hot'], ['b-hot']),
    (':--deep', ['a-hot'], ['b-hot']),
]
XML_SELECTOR_CASES = SELECTOR_CASES + [
    ('ns|item:nth-child(2 of ns|item)', ['a-cold'], ['b-hot']),
]

# Exercised in (seeded) randomized order in both directions.
OPERATIONS = ('select', 'iselect', 'match', 'closest', 'filter')
MODES = ('html', 'xml')


def _xml_markup(html_markup: str) -> str:
    """Promote an HTML fragment to namespaced XML markup."""

    root_id = 'root-a' if 'root-a' in html_markup else 'root-b'
    return html_markup.replace(
        f'<root id="{root_id}">',
        f'<root id="{root_id}" xmlns="{DEFAULT_URI}" xmlns:ns="{DEFAULT_URI}">',
        1,
    )


class Context:
    """An independent selection context: own tree and own compile inputs."""

    def __init__(self, name: str, mode: str):
        """Build the independent tree and compile inputs for *name*."""

        self.name = name
        self.mode = mode
        self.is_a = name == 'A'
        markup = MARKUP_A_HTML if self.is_a else MARKUP_B_HTML
        self.custom = dict(CUSTOM_A if self.is_a else CUSTOM_B)
        self.group_id = 'g-a' if self.is_a else 'g-b'
        self.leaf_id = 'a-leaf' if self.is_a else 'b-leaf'
        self.hot_id = 'a-hot' if self.is_a else 'b-hot'
        self.cold_id = 'a-cold' if self.is_a else 'b-cold'
        self.cases = SELECTOR_CASES

        if mode == 'xml':
            if not LXML_PRESENT:
                pytest.skip('XML mode requires the optional lxml dependency')
            self.parser = 'xml'
            markup = _xml_markup(markup)
            self.namespaces = dict(NAMESPACES)
            self.cases = XML_SELECTOR_CASES
        else:
            self.parser = 'html.parser'
            self.namespaces = None

        self.soup = bs4.BeautifulSoup(markup, self.parser)
        self.group = self.soup.find(id=self.group_id)
        self.leaf = self.soup.find(id=self.leaf_id)
        self.hot = self.soup.find(id=self.hot_id)

    def expected_ids(self, pattern: str) -> list[str]:
        """Expected matched ids for *pattern* against this context's group."""

        for entry in self.cases:
            if entry[0] == pattern:
                return entry[1 if self.is_a else 2]
        raise KeyError(pattern)

    def run_operation(self, operation: str, pattern: str, limit: int = 0) -> object:
        """Run one public API operation, returning a plain comparable value."""

        kwargs = {'namespaces': self.namespaces, 'custom': self.custom}
        if operation == 'select':
            return [el.attrs['id'] for el in sv.select(pattern, self.group, limit=limit, **kwargs)]
        if operation == 'iselect':
            return [el.attrs['id'] for el in sv.iselect(pattern, self.group, limit=limit, **kwargs)]
        if operation == 'match':
            if pattern == ':scope':
                return sv.match(pattern, self.group, **kwargs)
            return sv.match(pattern, self.hot, **kwargs)
        if operation == 'closest':
            found = sv.closest(pattern, self.leaf, **kwargs)
            return found.attrs['id'] if found is not None else None
        if operation == 'filter':
            compiled = sv.compile(pattern, **kwargs)
            return [el.attrs['id'] for el in compiled.filter(self.group)]
        raise AssertionError(f'unknown operation {operation!r}')

    def expected_operation(self, operation: str, pattern: str) -> object:
        """Expected value for `run_operation`."""

        if operation == 'match' and pattern == ':scope':
            return True
        matched = self.expected_ids(pattern)
        if operation in ('select', 'iselect', 'filter'):
            return matched
        if operation == 'match':
            if pattern == 'item:has(leaf)':
                return self.is_a
            if pattern in (':nth-child(2 of item)', 'ns|item:nth-child(2 of ns|item)'):
                return not self.is_a
            return None
        if operation == 'closest':
            if pattern == 'item:has(leaf)':
                return self.hot_id if self.is_a else self.cold_id
            return None
        raise AssertionError(operation)


# Patterns exercised per operation.  `match` adds a direct :scope probe,
# which can only ever match the node passed into the call.
OPERATION_PATTERNS = {
    'select': [case[0] for case in SELECTOR_CASES],
    'iselect': [case[0] for case in SELECTOR_CASES],
    'match': [':scope', 'item:has(leaf)', ':nth-child(2 of item)'],
    'closest': [':scope > item', 'item:has(leaf)', ':nth-child(2 of item)'],
    'filter': [case[0] for case in SELECTOR_CASES],
}

# XML-only replacement for the bare `:nth-child(2 of item)` probe.
_NTH_BARE = ':nth-child(2 of item)'
_NTH_NS = 'ns|item:nth-child(2 of ns|item)'


def _patterns_for(ctx: Context, operation: str) -> list[str]:
    """Pattern list valid for *ctx*; XML swaps/adds prefix-rebound variants."""

    patterns = list(OPERATION_PATTERNS[operation])
    if ctx.mode != 'xml':
        return patterns
    if operation in ('match', 'closest'):
        return [_NTH_NS if pattern == _NTH_BARE else pattern for pattern in patterns]
    if operation in ('select', 'iselect', 'filter'):
        patterns.append(_NTH_NS)
    return patterns


# --------------------------------------------------------------------------- #
# Compile-time cache: reuse of immutable syntax structures.
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _isolated_cache():
    """Start and end every test with an empty compile cache."""

    sv.purge()
    yield
    sv.purge()


def _misses() -> int:
    """Number of cache misses (actual compiles) recorded so far."""

    return cp._cached_css_compile.cache_info().misses


def _hits() -> int:
    """Number of cache hits recorded so far."""

    return cp._cached_css_compile.cache_info().hits


class TestCompileCache:
    """The cache key captures every input that affects compiled semantics."""

    def test_repeated_compile_reuses_immutable_object(self):
        """Re-compiling identical inputs returns the very same object."""

        p1 = sv.compile('p#x:has(span)', namespaces={'ns': 'urn:ns'}, custom={':--a': 'p'})
        p2 = sv.compile('p#x:has(span)', namespaces={'ns': 'urn:ns'}, custom={':--a': 'p'})
        assert p1 is p2
        assert _hits() == 1 and _misses() == 1
        # The syntax tree itself is immutable.
        with pytest.raises(AttributeError):
            p1.selectors.is_html = True  # type: ignore[misc]

    def test_namespace_insertion_order_does_not_matter(self):
        """Equal namespace mappings share a cache entry regardless of order."""

        p1 = sv.compile('ns|x', namespaces={'a': 'urn:a', 'b': 'urn:b'})
        p2 = sv.compile('ns|x', namespaces=[('b', 'urn:b'), ('a', 'urn:a')])
        assert p1 is p2
        assert _misses() == 1

    def test_custom_insertion_order_does_not_matter(self):
        """Equal custom selector mappings share a cache entry regardless of order."""

        p1 = sv.compile(':--b', custom={':--a': 'p', ':--b': ':--a'})
        p2 = sv.compile(':--b', custom=[(':--b', ':--a'), (':--a', 'p')])
        assert p1 is p2
        assert _misses() == 1

    @pytest.mark.parametrize('mode', MODES)
    def test_namespace_rebind_is_a_cache_miss(self, mode):
        """Same prefix rebound to another URI must not reuse the old compile."""

        if mode == 'xml' and not LXML_PRESENT:
            pytest.skip('XML mode requires the optional lxml dependency')
        if mode == 'xml':
            pattern = 'ns|item'
            ns_a = {'ns': 'urn:one', '': 'urn:one'}
            ns_b = {'ns': 'urn:two', '': 'urn:two'}
        else:
            # HTML carries no namespace table; custom selectors are the
            # remapped compile input for that mode.
            pattern = ':--thing'
            ns_a = None
            ns_b = None
        custom_a = CUSTOM_A if mode == 'html' else None
        custom_b = CUSTOM_B if mode == 'html' else None
        p1 = sv.compile(pattern, namespaces=ns_a, custom=custom_a)
        p2 = sv.compile(pattern, namespaces=ns_b, custom=custom_b)
        assert p1 is not p2
        assert _misses() == 2

    def test_mutating_same_dict_object_invalidates_the_cache(self):
        """A mutated mapping object, despite identity, must miss the cache."""

        namespaces = {'ns': 'urn:a'}
        p1 = sv.compile('ns|x', namespaces=namespaces)
        namespaces['ns'] = 'urn:b'
        p2 = sv.compile('ns|x', namespaces=namespaces)
        assert p1 is not p2
        assert p1.namespaces == ct.Namespaces({'ns': 'urn:a'})
        assert p2.namespaces == ct.Namespaces({'ns': 'urn:b'})
        assert _misses() == 2

    def test_post_compile_namespace_mutation_cannot_write_back(self):
        """Mutating the caller's dict after compile leaves the compile intact."""

        namespaces = {'ns': 'urn:a'}
        p1 = sv.compile('ns|x', namespaces=namespaces)
        namespaces['ns'] = 'urn:b'
        namespaces['other'] = 'urn:c'
        p_again = sv.compile('ns|x', namespaces={'ns': 'urn:a'})
        assert p_again is p1
        assert p1.namespaces == ct.Namespaces({'ns': 'urn:a'})
        assert 'other' not in p1.namespaces

    def test_post_compile_custom_mutation_cannot_write_back(self):
        """Mutating the caller's custom dict after compile leaves the compile intact."""

        custom = {':--thing': 'p', ':--deep': ':--thing'}
        p1 = sv.compile(':--deep', custom=custom)
        custom[':--thing'] = 'span'
        custom[':--added'] = 'div'
        p_again = sv.compile(':--deep', custom={':--thing': 'p', ':--deep': ':--thing'})
        assert p_again is p1
        assert dict(p1.custom) == {':--thing': 'p', ':--deep': ':--thing'}

    def test_flags_are_part_of_the_cache_key(self):
        """Different flags compile different objects; identical flags reuse."""

        p1 = sv.compile('p', flags=sv.DEBUG)
        p2 = sv.compile('p', flags=sv.DEBUG)
        p3 = sv.compile('p', flags=0)
        assert p1 is p2
        assert p1 is not p3
        assert p1.flags != p3.flags

    def test_pattern_is_part_of_the_cache_key(self):
        """Different patterns never share a compile."""

        assert sv.compile('p') is not sv.compile('div')

    def test_purge_clears_the_cache(self):
        """Controlled purge forces one fresh compile, then hits resume."""

        sv.compile('p')
        assert cp._cached_css_compile.cache_info().currsize == 1
        sv.purge()
        assert cp._cached_css_compile.cache_info().currsize == 0
        p1 = sv.compile('p')
        p2 = sv.compile('p')
        assert p1 is p2
        assert _misses() == 1 and _hits() == 1


# --------------------------------------------------------------------------- #
# Dynamic call context: scope nodes and limits never leak across calls.
# --------------------------------------------------------------------------- #

def _build_plan(ctx: Context, seed: int) -> list[tuple[str, str]]:
    """Build a seeded, shuffled (operation, pattern) plan for a context."""

    plan: list[tuple[str, str]] = []
    for operation in OPERATIONS:
        plan.extend((operation, pattern) for pattern in _patterns_for(ctx, operation))
    random.Random(seed).shuffle(plan)
    return plan


def _execute_plan(ctx: Context, plan: list[tuple[str, str]], errors: list[str],
                  barrier: threading.Event | None = None) -> None:
    """Execute a plan, recording readable failure context into *errors*."""

    if barrier is not None:
        barrier.wait()
    for step, (operation, pattern) in enumerate(plan):
        try:
            actual = ctx.run_operation(operation, pattern)
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            errors.append(
                f'ctx={ctx.name}/{ctx.mode} step={step} op={operation} '
                f'selector={pattern!r} raised {type(exc).__name__}: {exc} '
                f'(call sequence: {[op for op, _ in plan[:step + 1]]})'
            )
            continue
        expected = ctx.expected_operation(operation, pattern)
        if actual != expected:
            errors.append(
                f'ctx={ctx.name}/{ctx.mode} step={step} op={operation} '
                f'selector={pattern!r} expected={expected!r} got={actual!r} '
                f'(possible dynamic-context leak; call sequence: '
                f'{[op for op, _ in plan[:step + 1]]})'
            )


class TestScopeIsolation:
    """Shared compiled structures must not share per-call dynamic context."""

    @pytest.mark.parametrize('mode', MODES)
    @pytest.mark.parametrize('order', ('forward', 'reverse'))
    @pytest.mark.parametrize('seed', (17, 4242))
    def test_interleaved_calls_do_not_leak(self, mode, order, seed):
        """Two trees are driven in opposite orders over shared patterns."""

        ctx_a = Context('A', mode)
        ctx_b = Context('B', mode)
        plan_a = _build_plan(ctx_a, seed)
        plan_b = _build_plan(ctx_b, seed * 7 + 1)
        if order == 'reverse':
            plan_b.reverse()
            plan_a.reverse()
        # Strictly interleave: A-step, B-step, A-step, ... so each call on one
        # tree lands between two calls on the other tree.  The cache stays on
        # for the whole test; isolation must hold without purging.
        errors: list[str] = []
        for step in range(max(len(plan_a), len(plan_b))):
            if step < len(plan_a):
                _execute_plan(ctx_a, [plan_a[step]], errors)
            if step < len(plan_b):
                _execute_plan(ctx_b, [plan_b[step]], errors)
        assert not errors, '\n'.join(errors)

    @pytest.mark.parametrize('mode', MODES)
    def test_scope_node_is_dynamic_not_cached(self, mode):
        """`:scope` resolves to the node of each call, even on a shared object."""

        ctx_a = Context('A', mode)
        ctx_b = Context('B', mode)
        pattern = ':scope'
        compiled = sv.compile(pattern, namespaces=ctx_a.namespaces)
        # :scope has no custom-pseudo content, so identical inputs across both
        # trees resolve to the same cached, immutable object.
        compiled_b = sv.compile(pattern, namespaces=ctx_b.namespaces)
        assert compiled is compiled_b
        assert ctx_a.group is not ctx_b.group
        assert compiled.match(ctx_a.group)
        assert compiled.match(ctx_b.group)
        # `closest` returns the exact node passed in each call; a foreign node
        # can never leak across the per-call scope boundary.
        assert compiled.closest(ctx_a.leaf) is ctx_a.leaf
        assert compiled.closest(ctx_b.leaf) is ctx_b.leaf
        assert compiled.closest(ctx_a.leaf) is not ctx_b.leaf
        assert compiled.closest(ctx_b.leaf) is not ctx_a.leaf
        # Anchoring the shared compiled selector at different scope nodes
        # yields each node's own subtree only.
        child = sv.compile(':scope > item', namespaces=ctx_a.namespaces)
        assert [t.attrs['id'] for t in child.select(ctx_a.group)] == ctx_a.expected_ids(':scope > item')
        assert [t.attrs['id'] for t in child.select(ctx_b.group)] == ctx_b.expected_ids(':scope > item')

    @pytest.mark.parametrize('mode', MODES)
    def test_relative_selector_scope_is_per_call(self, mode):
        """Relative `:scope > item` anchored to different nodes stays separate."""

        ctx_a = Context('A', mode)
        ctx_b = Context('B', mode)
        pattern = ':scope > item'
        compiled = sv.compile(pattern, namespaces=ctx_a.namespaces,
                              custom=ctx_a.custom)
        assert [t.attrs['id'] for t in compiled.select(ctx_a.group)] == ctx_a.expected_ids(pattern)
        assert [t.attrs['id'] for t in compiled.select(ctx_b.group)] == ctx_b.expected_ids(pattern)
        # Repeat in reverse to ensure no call-local scope survives.
        assert [t.attrs['id'] for t in compiled.select(ctx_a.group)] == ctx_a.expected_ids(pattern)

    @pytest.mark.parametrize('mode', MODES)
    def test_limit_is_dynamic_and_not_cached(self, mode):
        """`limit` changes output per call without changing the cache entry."""

        ctx_a = Context('A', mode)
        ctx_b = Context('B', mode)
        pattern = ':scope > item'
        compiled = sv.compile(pattern, namespaces=ctx_a.namespaces,
                              custom=ctx_a.custom)
        size_before = cp._cached_css_compile.cache_info().currsize
        full_a = [t.attrs['id'] for t in compiled.select(ctx_a.group)]
        limited_a = [t.attrs['id'] for t in compiled.select(ctx_a.group, limit=1)]
        limited_b = [t.attrs['id'] for t in compiled.select(ctx_b.group, limit=1)]
        iter_limited = [t.attrs['id'] for t in compiled.iselect(ctx_b.group, limit=2)]
        assert limited_a == full_a[:1]
        assert limited_b == ctx_b.expected_ids(pattern)[:1]
        assert iter_limited == ctx_b.expected_ids(pattern)[:2]
        # limit=0 after limit=N must not be remembered across calls.
        assert [t.attrs['id'] for t in compiled.select(ctx_a.group)] == full_a
        assert cp._cached_css_compile.cache_info().currsize == size_before

    @pytest.mark.parametrize('mode', MODES)
    def test_shared_compile_used_against_both_trees(self, mode):
        """Identical compile inputs across trees share one immutable compile."""

        ctx_a = Context('A', mode)
        ctx_b = Context('B', mode)
        pattern = ':scope > item'
        p1 = sv.compile(pattern, namespaces=ctx_a.namespaces)
        p2 = sv.compile(pattern, namespaces=ctx_b.namespaces)
        assert p1 is p2
        assert [t.attrs['id'] for t in p1.select(ctx_a.group)] == ctx_a.expected_ids(pattern)
        assert [t.attrs['id'] for t in p2.select(ctx_b.group)] == ctx_b.expected_ids(pattern)

    @pytest.mark.parametrize('mode', MODES)
    def test_repeated_calls_hit_cache_never_recompile(self, mode):
        """Driving both trees never triggers a second compile of the same key."""

        ctx_a = Context('A', mode)
        ctx_b = Context('B', mode)
        pattern = ':scope > item'
        before = _misses()
        p1 = sv.compile(pattern, namespaces=ctx_a.namespaces)
        for ctx in (ctx_a, ctx_b, ctx_a, ctx_b):
            p_n = sv.compile(pattern, namespaces=ctx.namespaces)
            assert p_n is p1
            p_n.select(ctx.group)
        assert _misses() == before + 1
        assert _hits() >= 3

    def test_xml_default_namespace_is_a_distinct_cache_key(self):
        """A default namespace mapping must not collide with no mapping."""

        if not LXML_PRESENT:
            pytest.skip('XML mode requires the optional lxml dependency')
        p1 = sv.compile('item')
        p2 = sv.compile('item', namespaces={'': DEFAULT_URI})
        p3 = sv.compile('item', namespaces=[('', DEFAULT_URI)])
        assert p1 is not p2
        assert p2 is p3
        soup = bs4.BeautifulSoup(
            f'<root xmlns="{DEFAULT_URI}"><item id="z"></item></root>', 'xml'
        )
        assert [t.attrs['id'] for t in p2.select(soup)] == ['z']
        # The same pattern compiled with a *different* default URI must miss,
        # proving the default namespace is part of the compile semantics.
        p_other = sv.compile('item', namespaces={'': 'urn:other'})
        assert p_other is not p2
        assert p_other.select(soup) == []

    @pytest.mark.parametrize('mode', MODES)
    def test_parallel_contexts_stay_isolated(self, mode):
        """Both contexts run concurrently; no call-local state crosses threads."""

        ctx_a = Context('A', mode)
        ctx_b = Context('B', mode)
        plan_a = _build_plan(ctx_a, 101)
        plan_b = list(reversed(_build_plan(ctx_b, 202)))
        errors: list[str] = []
        barrier = threading.Event()
        threads = [
            threading.Thread(target=_execute_plan, args=(ctx_a, plan_a, errors),
                             kwargs={'barrier': barrier}),
            threading.Thread(target=_execute_plan, args=(ctx_b, plan_b, errors),
                             kwargs={'barrier': barrier}),
        ]
        for thread in threads:
            thread.start()
        barrier.set()
        for thread in threads:
            thread.join()
        assert not errors, '\n'.join(errors)


class TestConvenienceWrappers:
    """The module-level wrappers must forward `custom` like `compile` does."""

    @pytest.fixture
    def soup(self):
        """Parse a small tree with the stdlib parser (no external dependency)."""

        return bs4.BeautifulSoup(
            '<root><item id="x" class="c"><leaf></leaf></item></root>',
            'html.parser',
        )

    def test_custom_forwarded_by_every_wrapper(self, soup):
        """`custom=` reaches compile for all six public entry points."""

        custom = {':--alias': 'item.c', ':--nested': ':--alias'}
        item = soup.find(id='x')
        assert sv.select(':--nested', soup, custom=custom) == [item]
        assert list(sv.iselect(':--nested', soup, custom=custom)) == [item]
        assert sv.match(':--nested', item, custom=custom)
        assert sv.closest(':--nested', item, custom=custom) is item
        assert sv.filter(':--nested', soup.root, custom=custom) == [item]
        assert sv.select_one(':--nested', soup, custom=custom) is item

    def test_custom_forwarded_with_limit_and_flags(self, soup):
        """Forwarding works alongside the other dynamic/compile arguments."""

        custom = {':--alias': 'item.c'}
        item = soup.find(id='x')
        assert sv.select(':--alias', soup, limit=1, flags=sv.DEBUG, custom=custom) == [item]
        assert list(sv.iselect(':--alias', soup, limit=1, flags=sv.DEBUG, custom=custom)) == [item]

    def test_compiled_passthrough_rejects_custom(self, soup):
        """Feeding a compile a SoupSieve plus custom still raises clearly."""

        compiled = sv.compile('item')
        with pytest.raises(ValueError):
            sv.compile(compiled, custom={':--alias': 'item'})
