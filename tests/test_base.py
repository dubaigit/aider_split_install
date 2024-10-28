"""Base test classes and utilities."""

import asyncio
import time
import unittest
from contextlib import contextmanager


class AsyncTestCase(unittest.TestCase):
    """Base class for async tests with enhanced async support"""

    def setUp(self):
        """Set up test environment with proper async context"""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.addCleanup(self.cleanup_loop)

    def cleanup_loop(self):
        """Clean up the event loop"""
        self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        self.loop.close()
        asyncio.set_event_loop(None)

    async def asyncSetUp(self):
        """Optional async setup - override in subclasses"""
        pass

    async def asyncTearDown(self):
        """Optional async teardown - override in subclasses"""
        pass

    def run_async_test(self, coro):
        """Run coroutine in the test loop with enhanced error handling and timeout"""
        async def _run_with_setup():
            await self.asyncSetUp()
            try:
                result = await asyncio.wait_for(coro, timeout=5.0)
                return result
            except asyncio.TimeoutError:
                self.fail("Async test timed out after 5 seconds")
            finally:
                await self.asyncTearDown()

        try:
            return self.loop.run_until_complete(_run_with_setup())
        except Exception as e:
            self.fail(f"Async test failed with error: {type(e).__name__}: {str(e)}\n{e.__traceback__}")

    @contextmanager
    def assertNotRaises(self, exc_type):
        """Context manager to assert no exception is raised"""
        try:
            yield
        except exc_type as e:
            self.fail(f"Expected no {exc_type.__name__} but got: {e}")

    async def wait_for_condition(self, condition_func, timeout=5.0, interval=0.1):
        """Wait for a condition to become true with timeout"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if condition_func():
                return True
            await asyncio.sleep(interval)
        return False
