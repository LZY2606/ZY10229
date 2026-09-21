"""Test performance cases."""
import unittest
import sys
import signal
import time
import soupsieve as sv
from soupsieve.util import SelectorSyntaxError


class Timeout(Exception):
    """Timeout exception."""


@unittest.skipUnless(not sys.platform.startswith('win'), "Unix/Linux test")
class TestPerformance(unittest.TestCase):
    """Test performance."""

    def assert_performance(self, selector):
        """Test performance of of specific cases."""

        LIMIT = 5.0

        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(Timeout()))
        signal.setitimer(signal.ITIMER_REAL, LIMIT)
        dt = None
        t = 0
        try:
            t = time.perf_counter()
            sv.compile(selector)
            dt = time.perf_counter() - t
        except SelectorSyntaxError:
            dt = time.perf_counter() - t
        except Timeout:
            pass
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

        hung = dt is None
        success = not hung and dt < 1.0

        if not success:
            print('Selector:', selector)
            print(f"{'> %.0f s (HANG)' % LIMIT if hung else '%.3f s' % dt}")

        self.assertTrue(success)

    def test_performance_caase(self):
        """Test performance cases."""

        for n in (1000, 2000, 4000, 8000):
            self.assert_performance("[a=" + "a" * n)

        for n in (2000, 4000, 8000, 16000):
            self.assert_performance("a" * n + "!")

        self.assert_performance("[a=" + "a" * 12000)

        for n in (2000, 4000, 8000, 16000):
            self.assert_performance("a" + " " * n + "b")

        self.assert_performance("a" + " " * 20000 + "b")
