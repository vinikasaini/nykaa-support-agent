import asyncio

from fastmcp import Client


MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"


async def main():

    client = Client(MCP_SERVER_URL)

    async with client:

        print("\nConnected to Nykaa MCP Server")

        print("\nAvailable tools:")

        tools = await client.list_tools()

        for tool in tools:
            print("-", tool.name)

        record_ids = [
            "NYK001",
            "NYK002"
        ]

        for record_id in record_ids:

            print("\n" + "=" * 60)
            print(f"Calling lookup_order for: {record_id}")
            print("=" * 60)

            result = await client.call_tool(
                "lookup_order",
                {
                    "record_id": record_id
                }
            )

            print("\nStandardized MCP Response:")
            print(result)


if __name__ == "__main__":
    asyncio.run(main())