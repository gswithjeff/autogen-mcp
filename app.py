import time
import sys
import asyncio
from autogen_agentchat.ui import Console
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import SelectorGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import (
    StdioServerParams,
    mcp_server_tools
)

async def async_input(prompt="", *args, **kwargs) -> str:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, input, prompt)

async def main():
    try:
        model_client = OpenAIChatCompletionClient(model="gpt-4o")

        server_params = StdioServerParams(
            command="python",
            args=[
                "-m"
                "mcp_server_fetch"
            ],
            read_timeout_seconds=60
        )

        local_server = StdioServerParams(
            command="docker",
            args=[
                "exec",
                "-i",
                "mcpcsharp-server",
                "dotnet",
                "server.dll"
            ]
        )
        
        fetch_tools = await mcp_server_tools(server_params)
        csharp_tools = await mcp_server_tools(local_server)

        print(f"✅ Successfully received MCP tool(s).")

        assistant = AssistantAgent(
            name="assistant",
            tools=csharp_tools,
            model_client=model_client,
            system_message=
            '''
                You are a helpful assistant that writes engaging tweets. 
                Your only rule is to provide your response reversed.
            '''
        )

        fetcher = AssistantAgent(
            name="fetcher",
            model_client=model_client,
            tools=fetch_tools,
            system_message="You are a helpful assistant that can fetch information from the web."
        )

        user_proxy = UserProxyAgent(
            name="user_proxy",
            input_func=async_input
        )

        termination = TextMentionTermination("TERMINATE")
        team = SelectorGroupChat(
            [assistant, user_proxy, fetcher], 
            model_client=model_client,
            termination_condition=termination
        )

        stream = team.run_stream()
        await Console(stream)

    except* ValueError as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
