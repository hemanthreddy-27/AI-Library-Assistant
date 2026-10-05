"""
MCP client for the AI-Powered Intelligent Library Assistant.

This client connects to the local MCP server through stdio,
discovers the available tools, and performs demonstration
tests for all registered library assistant capabilities.
"""

import anyio

from mcp import Client, StdioServerParameters


# ==================================================
# MCP Server Configuration
# ==================================================

server = StdioServerParameters(
    command="python3",
    args=["-m", "mcp_server.server"],
)


# ==================================================
# Helper Functions
# ==================================================

def print_section(title: str):
    """Print a formatted section heading."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def print_result(result):
    """Print an MCP tool result in a readable format."""
    print(result)

    if getattr(result, "is_error", False):
        print("\nSTATUS: FAILED")
    else:
        print("\nSTATUS: SUCCESS")


# ==================================================
# Main MCP Client
# ==================================================

async def main():
    """Connect to the MCP server and test all available tools."""

    print("=" * 60)
    print("AI LIBRARY ASSISTANT - MCP CLIENT")
    print("=" * 60)

    try:
        async with Client(server) as client:

            # --------------------------------------------------
            # Connection Test
            # --------------------------------------------------

            print("\nMCP server connected successfully.")

            # --------------------------------------------------
            # Discover MCP Tools
            # --------------------------------------------------

            print_section("AVAILABLE MCP TOOLS")

            tool_result = await client.list_tools()

            for tool in tool_result.tools:
                print(f"  - {tool.name}")

            print(f"\nTotal MCP tools: {len(tool_result.tools)}")

            # --------------------------------------------------
            # TEST 1: Search Books
            # --------------------------------------------------

            print_section("TEST 1: search_books")

            result = await client.call_tool(
                "search_books",
                {
                    "query": "business",
                    "limit": 5,
                },
            )

            print_result(result)

            # --------------------------------------------------
            # TEST 2: Get Book Details
            # --------------------------------------------------

            print_section("TEST 2: get_book_details")

            result = await client.call_tool(
                "get_book_details",
                {
                    "book_id": "200001",
                },
            )

            print_result(result)

            # --------------------------------------------------
            # TEST 3: Search by Author
            # --------------------------------------------------

            print_section("TEST 3: search_by_author")

            result = await client.call_tool(
                "search_by_author",
                {
                    "author": "Kotler",
                    "limit": 5,
                },
            )

            print_result(result)

            # --------------------------------------------------
            # TEST 4: Search by DDC
            # --------------------------------------------------

            print_section("TEST 4: search_by_ddc")

            result = await client.call_tool(
                "search_by_ddc",
                {
                    "ddc": "650",
                    "limit": 5,
                },
            )

            print_result(result)

            # --------------------------------------------------
            # TEST 5: User Borrowing History
            # --------------------------------------------------

            print_section("TEST 5: get_user_borrowing_history")

            result = await client.call_tool(
                "get_user_borrowing_history",
                {
                    "user_id": 10001,
                },
            )

            print_result(result)

            # --------------------------------------------------
            # TEST 6: Popular Books
            # --------------------------------------------------

            print_section("TEST 6: get_popular_books")

            result = await client.call_tool(
                "get_popular_books",
                {
                    "limit": 5,
                },
            )

            print_result(result)

            # --------------------------------------------------
            # TEST 7: Personalized Recommendations
            # --------------------------------------------------

            print_section("TEST 7: recommend_books")

            result = await client.call_tool(
                "recommend_books",
                {
                    "user_id": 10001,
                    "top_n": 5,
                },
            )

            print_result(result)

            # --------------------------------------------------
            # TEST 8: Text Summarization
            # --------------------------------------------------

            print_section("TEST 8: summarize_text")

            result = await client.call_tool(
                "summarize_text",
                {
                    "text": (
                        "Artificial intelligence can help university libraries "
                        "provide personalized book recommendations by analyzing "
                        "borrowing history, book metadata, user interests, and "
                        "previous interactions. Natural language processing can "
                        "also summarize long documents and help students quickly "
                        "understand important information."
                    ),
                    "max_input_length": 512,
                    "max_summary_length": 128,
                    "min_summary_length": 20,
                },
            )

            print_result(result)

            # --------------------------------------------------
            # Completion
            # --------------------------------------------------

            print_section("MCP CLIENT TEST COMPLETED")

            print("All eight MCP tools were invoked.")
            print("Review the STATUS lines above for individual results.")

    except Exception as error:
        print_section("MCP CLIENT ERROR")
        print(f"Error type: {type(error).__name__}")
        print(f"Error message: {error}")

        raise


# ==================================================
# Entry Point
# ==================================================

if __name__ == "__main__":
    anyio.run(main)