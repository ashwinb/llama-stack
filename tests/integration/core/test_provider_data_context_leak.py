"""
Integration test for provider data context isolation in streaming requests.

This test verifies that PROVIDER_DATA_VAR doesn't leak between sequential
streaming requests when using a real server.

IMPORTANT: This test demonstrates the context leak bug and validates the fix.
- On main branch (without fix): This test will FAIL
- On fix_context_leak branch (with fix): This test will PASS

To run this test with the library client:
    pytest tests/integration/test_provider_data_context_leak.py::test_provider_data_isolation_library_client -xvs

NOTE: The server-based test requires custom provider registration which
is more complex. The library client test demonstrates the same bug.
"""

import json
import pytest


@pytest.mark.asyncio
async def test_provider_data_isolation_library_client():
    """
    Test using library client to demonstrate provider data context leak.

    This is a simplified version that uses the library client directly
    and mocks the provider behavior to demonstrate the bug.

    BUG DEMONSTRATION:
    - Without fix: PROVIDER_DATA_VAR leaks between streaming requests
    - With fix: Each request has isolated context
    """
    from llama_stack.core.request_headers import PROVIDER_DATA_VAR, request_provider_data_context
    from llama_stack.core.utils.context import preserve_contexts_async_generator

    async def mock_streaming_provider():
        """Simulates a streaming provider that reads PROVIDER_DATA_VAR"""
        # This simulates what a real provider does - read from context
        provider_data = PROVIDER_DATA_VAR.get()
        yield {"provider_data": provider_data, "chunk": 1}

    async def sse_generator(gen):
        """Simulates the SSE generator in the server"""
        async for item in gen:
            yield f"data: {json.dumps(item)}\n\n"

    # Request 1: Set provider data to {"key": "value1"}
    headers1 = {"X-LlamaStack-Provider-Data": json.dumps({"key": "value1"})}
    with request_provider_data_context(headers1):
        gen1 = preserve_contexts_async_generator(
            sse_generator(mock_streaming_provider()),
            [PROVIDER_DATA_VAR]
        )

    # Consume request 1 chunks
    chunks1 = [chunk async for chunk in gen1]
    data1 = json.loads(chunks1[0].split("data: ")[1])
    assert data1["provider_data"] == {"key": "value1"}, "Request 1 should see its own data"

    # After request 1 completes, context should be cleared
    # BUG: On main, PROVIDER_DATA_VAR still contains {"key": "value1"}
    # FIX: On fix branch, PROVIDER_DATA_VAR is None
    leaked_data = PROVIDER_DATA_VAR.get()
    assert leaked_data is None, f"Context leaked after request 1: {leaked_data}"

    # Request 2: Set different provider data {"key": "value2"}
    headers2 = {"X-LlamaStack-Provider-Data": json.dumps({"key": "value2"})}
    with request_provider_data_context(headers2):
        gen2 = preserve_contexts_async_generator(
            sse_generator(mock_streaming_provider()),
            [PROVIDER_DATA_VAR]
        )

    # Consume request 2 chunks
    chunks2 = [chunk async for chunk in gen2]
    data2 = json.loads(chunks2[0].split("data: ")[1])

    # BUG: On main, this fails because data2["provider_data"] == {"key": "value1"} (leaked!)
    # FIX: On fix branch, data2["provider_data"] == {"key": "value2"} (correct!)
    assert data2["provider_data"] == {"key": "value2"}, \
        f"Request 2 should see its own data, not {data2['provider_data']}"

    # Verify context is cleared after request 2
    leaked_data2 = PROVIDER_DATA_VAR.get()
    assert leaked_data2 is None, f"Context leaked after request 2: {leaked_data2}"

    # Request 3: No provider data (None)
    gen3 = preserve_contexts_async_generator(
        sse_generator(mock_streaming_provider()),
        [PROVIDER_DATA_VAR]
    )

    chunks3 = [chunk async for chunk in gen3]
    data3 = json.loads(chunks3[0].split("data: ")[1])

    # BUG: On main, this would show {"key": "value2"} (leaked from request 2)
    # FIX: On fix branch, this correctly shows None
    assert data3["provider_data"] is None, \
        f"Request 3 with no provider data should see None, not {data3['provider_data']}"


# This test is currently skipped but shows how we WOULD test with a real server
# if we had easy provider registration
@pytest.mark.skipif(
    True,
    reason="Requires custom provider registration - see fixtures/context_echo_provider.py"
)
def test_provider_data_isolation_with_server(llama_stack_client):
    """
    Test with real server and custom provider.

    To enable this test:
    1. Register the ContextEchoInferenceProvider from fixtures/context_echo_provider.py
    2. Configure a model using this provider
    3. Run with --stack-config server:ci-tests
    """

    # Request 1
    response1 = llama_stack_client.inference.chat_completion(
        model_id="context-echo-model",
        messages=[{"role": "user", "content": "test"}],
        stream=True,
        extra_headers={
            "X-LlamaStack-Provider-Data": json.dumps({"test_key": "value1"})
        },
    )

    chunks1 = []
    for chunk in response1:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks1.append(chunk.choices[0].delta.content)

    response1_data = json.loads("".join(chunks1))
    assert response1_data["provider_data"] == {"test_key": "value1"}

    # Request 2
    response2 = llama_stack_client.inference.chat_completion(
        model_id="context-echo-model",
        messages=[{"role": "user", "content": "test"}],
        stream=True,
        extra_headers={
            "X-LlamaStack-Provider-Data": json.dumps({"test_key": "value2"})
        },
    )

    chunks2 = []
    for chunk in response2:
        if chunk.choices and chunk.choices[0].delta.content:
            chunks2.append(chunk.choices[0].delta.content)

    response2_data = json.loads("".join(chunks2))
    assert response2_data["provider_data"] == {"test_key": "value2"}
