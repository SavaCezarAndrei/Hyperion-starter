"""Shared test setup.

The LLM clients in llm.py are bound to the event loop they first run on, and on
any other loop they hang instead of failing. Every in-process test must therefore
use this single loop: `loop.run_until_complete(...)`, never `asyncio.run(...)` or
a loop of its own.
"""

import asyncio

loop = asyncio.new_event_loop()
