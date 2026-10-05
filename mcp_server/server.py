"""
MCP server for the AI-Powered Intelligent Library Assistant.

This server exposes the project's library search,
recommendation, borrowing-history, popularity, and
NLP summarization functionality through Model Context Protocol.
"""

from mcp.server import MCPServer

from mcp_server.library_tools import (
    tool_search_books,
    tool_get_book_details,
    tool_search_by_author,
    tool_search_by_ddc,
    tool_get_user_borrowing_history,
    tool_get_popular_books,
    tool_recommend_books,
    tool_summarize_text,
)


# ==================================================
# Create MCP server
# ==================================================

mcp = MCPServer(
    "AI Library Assistant",
    instructions=(
        "An AI-powered university library assistant providing "
        "book search, book details, author and DDC search, "
        "borrowing history, popular books, personalized recommendations, "
        "and NLP-based text summarization."
    ),
)


# ==================================================
# MCP Tool: Search books
# ==================================================

@mcp.tool()
def search_books(
    query: str,
    limit: int = 10,
) -> list:
    """
    Search library books by title, author, or related text.
    """
    return tool_search_books(
        query,
        limit,
    )


# ==================================================
# MCP Tool: Get book details
# ==================================================

@mcp.tool()
def get_book_details(
    book_id: str,
) -> dict:
    """
    Get detailed metadata for a specific book using its book ID.
    """
    return tool_get_book_details(
        book_id,
    )


# ==================================================
# MCP Tool: Search by author
# ==================================================

@mcp.tool()
def search_by_author(
    author: str,
    limit: int = 10,
) -> list:
    """
    Search library books by author name.
    """
    return tool_search_by_author(
        author,
        limit,
    )


# ==================================================
# MCP Tool: Search by DDC
# ==================================================

@mcp.tool()
def search_by_ddc(
    ddc: str,
    limit: int = 10,
) -> list:
    """
    Search library books using a Dewey Decimal Classification number.
    """
    return tool_search_by_ddc(
        ddc,
        limit,
    )


# ==================================================
# MCP Tool: User borrowing history
# ==================================================

@mcp.tool()
def get_user_borrowing_history(
    user_id: int,
) -> list:
    """
    Get the historical borrowing records of a library user.
    """
    return tool_get_user_borrowing_history(
        user_id,
    )


# ==================================================
# MCP Tool: Popular books
# ==================================================

@mcp.tool()
def get_popular_books(
    limit: int = 10,
) -> list:
    """
    Get the most frequently borrowed books in the
    historical library dataset.
    """
    return tool_get_popular_books(
        limit,
    )


# ==================================================
# MCP Tool: Personalized recommendations
# ==================================================

@mcp.tool()
def recommend_books(
    user_id: int,
    top_n: int = 10,
) -> list:
    """
    Generate personalized book recommendations for a library user.
    """
    return tool_recommend_books(
        user_id,
        top_n,
    )


# ==================================================
# MCP Tool: Text summarization
# ==================================================

@mcp.tool()
def summarize_text(
    text: str,
    max_input_length: int = 512,
    max_summary_length: int = 128,
    min_summary_length: int = 20,
) -> str:
    """
    Generate an abstractive summary of the supplied text
    using the trained T5-small model.
    """
    return tool_summarize_text(
        text,
        max_input_length,
        max_summary_length,
        min_summary_length,
    )


# ==================================================
# Run MCP server
# ==================================================

if __name__ == "__main__":
    mcp.run(transport="stdio")