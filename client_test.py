import json
import unittest

from dotenv import load_dotenv

from src.cached_llm import (
    AssistantMessage,
    Client,
    Message,
    SystemMessage,
    ToolMessage,
    UserMessage,
)

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Returns the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for.",
                }
            },
            "required": ["location"],
            "additionalProperties": False,
        },
    },
}

PMS = [
    ("openai", "gpt-5-mini"),
    ("anthropic", "claude-sonnet-4-0"),
    ("gemini", "gemini-2.5-flash"),
]


class TestChat(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        load_dotenv()

    async def test_simple_chat(self) -> None:
        """Tests basic chat functionality."""
        for provider, model in PMS:
            with self.subTest(provider=provider, model=model):
                async with Client(provider) as client:
                    messages = [
                        SystemMessage(content="You are a helpful assistant."),
                        UserMessage(content="Say hello in exactly 3 words."),
                    ]
                    response = await client.ainvoke(model, messages)
                    self.assertIsInstance(response, AssistantMessage)
                    self.assertIsNotNone(response.content)
                    assert response.content is not None  # type narrowing
                    self.assertLessEqual(len(response.content.split()), 5)

    async def test_tool_call_invocation(self) -> None:
        """Tests tool calling."""
        for provider, model in PMS:
            with self.subTest(provider=provider, model=model):
                async with Client(provider) as client:
                    messages = [
                        SystemMessage(content="You are a helpful weather assistant."),
                        UserMessage(content="What's the weather in Paris?"),
                    ]

                    response = await client.ainvoke(
                        model, messages, tools=[WEATHER_TOOL], tool_choice="auto"
                    )
                    self.assertIsNotNone(response.tool_calls)
                    assert response.tool_calls is not None  # type narrowing
                    self.assertEqual(len(response.tool_calls), 1)
                    tool_call = response.tool_calls[0]
                    self.assertEqual(tool_call.function.name, "get_weather")

                    # Verify location argument.
                    args = json.loads(tool_call.function.arguments)
                    self.assertIn("location", args)
                    self.assertIn("paris", args["location"].lower())

    async def test_full_tool_flow(self) -> None:
        """Tests the complete tool calling flow."""
        for provider, model in PMS:
            with self.subTest(provider=provider, model=model):
                async with Client(provider) as client:
                    messages: list[Message] = [
                        SystemMessage(content="You are a helpful weather assistant."),
                        UserMessage(content="What's the weather in New York?"),
                    ]

                    # Get tool call.
                    response1 = await client.ainvoke(
                        model, messages, tools=[WEATHER_TOOL], tool_choice="auto"
                    )
                    self.assertIsNotNone(response1.tool_calls)
                    assert response1.tool_calls is not None  # type narrowing
                    tool_call = response1.tool_calls[0]

                    # Add assistant message and tool response.
                    messages.append(response1)
                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_call.id,
                            name="get_weather",
                            content=json.dumps(
                                {
                                    "temperature": "22°C",
                                    "condition": "sunny",
                                    "humidity": "60%",
                                }
                            ),
                        )
                    )

                    # Get final response.
                    response2 = await client.ainvoke(model, messages)
                    self.assertIsNotNone(response2.content)
                    assert response2.content is not None  # type narrowing

                    # Verify the response incorporates the weather data.
                    content_lower = response2.content.lower()
                    self.assertTrue(
                        any(
                            term in content_lower
                            for term in ["22", "sunny", "temperature"]
                        )
                    )


if __name__ == "__main__":
    unittest.main()
