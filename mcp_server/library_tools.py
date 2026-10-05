"""
MCP library tool functions.

This module connects the MCP server to the project's
recommendation and summarization components.
"""

from recommendation.recommend import (
    search_books,
    get_book_details,
    search_by_author,
    search_by_ddc,
    get_user_borrowing_history,
    get_popular_books,
    recommend_books,
)

from summarization.summarize import summarize_text


# ==================================================
# MCP Tool: Search books
# ==================================================

def tool_search_books(
    query: str,
    limit: int = 10,
) -> list:
    """
    Search the library catalogue by title, author, or book ID.

    Args:
        query: Search text.
        limit: Maximum number of results to return.

    Returns:
        A list of matching library books.
    """
    return search_books(
        query=query,
        limit=limit,
    )


# ==================================================
# MCP Tool: Get book details
# ==================================================

def tool_get_book_details(
    book_id: str,
) -> dict:
    """
    Get detailed information about a specific book.

    Args:
        book_id: Library book ID.

    Returns:
        A dictionary containing book metadata.
    """
    return get_book_details(
        book_id=book_id,
    )


# ==================================================
# MCP Tool: Search by author
# ==================================================

def tool_search_by_author(
    author: str,
    limit: int = 10,
) -> list:
    """
    Search library books by author name.

    Args:
        author: Author name or author search text.
        limit: Maximum number of results to return.

    Returns:
        A list of matching library books.
    """
    return search_by_author(
        author=author,
        limit=limit,
    )


# ==================================================
# MCP Tool: Search by DDC
# ==================================================

def tool_search_by_ddc(
    ddc: str,
    limit: int = 10,
) -> list:
    """
    Search library books using Dewey Decimal Classification.

    Args:
        ddc: DDC number or DDC search text.
        limit: Maximum number of results to return.

    Returns:
        A list of matching library books.
    """
    return search_by_ddc(
        ddc=ddc,
        limit=limit,
    )


# ==================================================
# MCP Tool: User borrowing history
# ==================================================

def tool_get_user_borrowing_history(
    user_id: int,
) -> list:
    """
    Get historical borrowing records for a library user.

    Args:
        user_id: Library membership/user ID.

    Returns:
        A list of the user's historical borrowing records.
    """
    return get_user_borrowing_history(
        user_id=user_id,
    )


# ==================================================
# MCP Tool: Popular books
# ==================================================

def tool_get_popular_books(
    limit: int = 10,
) -> list:
    """
    Get the most frequently borrowed books from the
    historical library dataset.

    Args:
        limit: Maximum number of popular books to return.

    Returns:
        A list of popular books.
    """
    return get_popular_books(
        limit=limit,
    )


# ==================================================
# MCP Tool: Personalized recommendations
# ==================================================

def tool_recommend_books(
    user_id: int,
    top_n: int = 10,
) -> list:
    """
    Generate personalized book recommendations for a
    library user.

    Args:
        user_id: Library membership/user ID.
        top_n: Number of recommendations to return.

    Returns:
        A list of recommended books.
    """
    return recommend_books(
        user_id=user_id,
        top_n=top_n,
    )


# ==================================================
# MCP Tool: Text summarization
# ==================================================

def tool_summarize_text(
    text: str,
    max_input_length: int = 512,
    max_summary_length: int = 128,
    min_summary_length: int = 20,
) -> str:
    """
    Generate an abstractive summary using the trained
    T5-small summarization model.

    Args:
        text: Input text to summarize.
        max_input_length: Maximum number of input tokens.
        max_summary_length: Maximum number of summary tokens.
        min_summary_length: Minimum number of summary tokens.

    Returns:
        Generated summary text.
    """
    return summarize_text(
        text=text,
        max_input_length=max_input_length,
        max_summary_length=max_summary_length,
        min_summary_length=min_summary_length,
    )