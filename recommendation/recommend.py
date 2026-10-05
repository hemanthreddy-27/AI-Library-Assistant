from pathlib import Path
import sys

import joblib
import pandas as pd


# ==================================================
# Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models" / "library_recommender_artifacts"
DATASET_PATH = (
    PROJECT_ROOT
    / "datasets"
    / "dataset_univLib_preprocessed.csv"
)


# ==================================================
# Load library dataset
# ==================================================

library_data = pd.read_csv(DATASET_PATH)


# ==================================================
# Load trained recommendation artifacts
# ==================================================

book_similarity = joblib.load(
    MODEL_DIR / "book_similarity.joblib"
)

book_factors = joblib.load(
    MODEL_DIR / "book_factors.joblib"
)

book_to_index = joblib.load(
    MODEL_DIR / "book_to_index.joblib"
)

books_metadata = joblib.load(
    MODEL_DIR / "books_metadata.joblib"
)

department_book_popularity = joblib.load(
    MODEL_DIR / "department_book_popularity.joblib"
)

global_popularity = joblib.load(
    MODEL_DIR / "global_popularity.joblib"
)

program_book_popularity = joblib.load(
    MODEL_DIR / "program_book_popularity.joblib"
)

svd_model = joblib.load(
    MODEL_DIR / "svd_model.joblib"
)

user_factors = joblib.load(
    MODEL_DIR / "user_factors.joblib"
)

user_profiles = joblib.load(
    MODEL_DIR / "user_profiles.joblib"
)

user_to_index = joblib.load(
    MODEL_DIR / "user_to_index.joblib"
)


# ==================================================
# Basic information
# ==================================================

# IMPORTANT:
# MCP stdio transport uses stdout for JSON-RPC messages.
# Diagnostic messages must therefore go to stderr.

print(
    "Recommendation models loaded successfully.",
    file=sys.stderr
)

print(
    f"Number of books: {len(book_to_index):,}",
    file=sys.stderr
)

print(
    f"Number of users: {len(user_to_index):,}",
    file=sys.stderr
)


# ==================================================
# Metadata helper
# ==================================================

def clean_author(title, author):
    """
    Clean obviously invalid author metadata for user-facing output.

    Some records in the historical dataset contain the book title
    itself in the Author field. This function preserves normal
    author values and replaces only the obvious title-as-author
    case with 'Unknown'.

    This is a presentation-layer cleanup only. It does not modify
    the dataset or any trained recommendation artifacts.
    """

    title_text = str(title).strip()
    author_text = str(author).strip()

    if not author_text:
        return "Unknown"

    if author_text.lower() == title_text.lower():
        return "Unknown"

    return author


# ==================================================
# Helper: get books already borrowed by user
# ==================================================

def get_borrowed_book_ids(user_id):
    """
    Return the set of book IDs already borrowed by a user.
    """

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return set()

    user_history = library_data[
        library_data["Membership Number"] == user_id
    ]

    return set(
        user_history["Book ID Number"]
        .astype(str)
        .str.strip()
    )


# ==================================================
# Recommend books for a user
# ==================================================

def recommend_books(user_id, top_n=10):
    """
    Recommend books for a known library user.

    Parameters:
        user_id: Membership Number
        top_n: Number of recommendations to return

    Returns:
        List of recommended book records
    """

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return []

    # Check whether the user exists
    if user_id not in user_to_index:
        return []

    user_idx = user_to_index[user_id]

    # Get the user's learned latent representation
    user_vector = user_factors[user_idx]

    # Score all books
    scores = book_factors @ user_vector

    # Get books already borrowed by this user
    borrowed_books = get_borrowed_book_ids(user_id)

    # Map index -> book ID
    index_to_book = {
        index: book_id
        for book_id, index in book_to_index.items()
    }

    # Exclude already borrowed books
    for book_id in borrowed_books:

        if book_id in book_to_index:

            book_idx = book_to_index[book_id]
            scores[book_idx] = float("-inf")

        elif book_id.isdigit():

            numeric_book_id = int(book_id)

            if numeric_book_id in book_to_index:

                book_idx = book_to_index[numeric_book_id]
                scores[book_idx] = float("-inf")

    # Get ranked book indices
    ranked_indices = scores.argsort()[::-1]

    recommendations = []

    for book_idx in ranked_indices:

        if len(recommendations) >= top_n:
            break

        book_id = index_to_book.get(book_idx)

        if book_id is None:
            continue

        if str(book_id).strip() in borrowed_books:
            continue

        book_row = books_metadata.iloc[book_idx]

        title = book_row["Book Title"]

        recommendations.append(
            {
                "book_id": book_id,
                "title": title,
                "author": clean_author(
                    title,
                    book_row["Author"],
                ),
                "ddc": book_row["DDC Number"],
            }
        )

    return recommendations


# ==================================================
# Search books
# ==================================================

def search_books(query, limit=10):
    """
    Search the library catalogue by:

    - Book title
    - Author
    - Book ID
    """

    query = str(query).strip().lower()

    if not query:
        return []

    results = []

    for _, row in books_metadata.iterrows():

        title = str(row["Book Title"]).lower()
        author = str(row["Author"]).lower()
        book_id = str(row["Book ID Number"]).lower()

        if (
            query in title
            or query in author
            or query in book_id
        ):

            results.append(
                {
                    "book_id": row["Book ID Number"],
                    "title": row["Book Title"],
                    "author": clean_author(
                        row["Book Title"],
                        row["Author"],
                    ),
                    "ddc": row["DDC Number"],
                }
            )

        if len(results) >= limit:
            break

    return results


# ==================================================
# Search books by author
# ==================================================

def search_by_author(author, limit=10):
    """
    Search the library catalogue by author name.
    """

    author = str(author).strip().lower()

    if not author:
        return []

    results = []

    for _, row in books_metadata.iterrows():

        book_author = str(row["Author"]).lower()

        if author in book_author:

            results.append(
                {
                    "book_id": row["Book ID Number"],
                    "title": row["Book Title"],
                    "author": clean_author(
                        row["Book Title"],
                        row["Author"],
                    ),
                    "ddc": row["DDC Number"],
                }
            )

        if len(results) >= limit:
            break

    return results


# ==================================================
# Search books by DDC number
# ==================================================

def search_by_ddc(ddc, limit=10):
    """
    Search the library catalogue by
    Dewey Decimal Classification number.
    """

    ddc = str(ddc).strip()

    if not ddc:
        return []

    results = []

    for _, row in books_metadata.iterrows():

        book_ddc = str(row["DDC Number"]).strip()

        if ddc in book_ddc:

            results.append(
                {
                    "book_id": row["Book ID Number"],
                    "title": row["Book Title"],
                    "author": clean_author(
                        row["Book Title"],
                        row["Author"],
                    ),
                    "ddc": row["DDC Number"],
                }
            )

        if len(results) >= limit:
            break

    return results


# ==================================================
# Get book details
# ==================================================

def get_book_details(book_id):
    """
    Return detailed information about a specific book.
    """

    book_id = str(book_id).strip()

    if not book_id:
        return None

    matches = books_metadata[
        books_metadata["Book ID Number"]
        .astype(str)
        .str.strip()
        == book_id
    ]

    if matches.empty:
        return None

    row = matches.iloc[0]

    return {
        "book_id": row["Book ID Number"],
        "title": row["Book Title"],
        "author": clean_author(
            row["Book Title"],
            row["Author"],
        ),
        "ddc": row["DDC Number"],
        "isbn": row.get("ISBN", None),
        "published_year": row.get(
            "Published Year",
            None
        ),
        "publisher": row.get(
            "Publisher Name",
            None
        ),
    }


# ==================================================
# Get user borrowing history
# ==================================================

def get_user_borrowing_history(user_id):
    """
    Return the historical borrowing records
    of a library user.
    """

    try:
        user_id = int(user_id)
    except (ValueError, TypeError):
        return []

    user_history = library_data[
        library_data["Membership Number"] == user_id
    ].sort_values("Borrow Date")

    if user_history.empty:
        return []

    results = []

    for _, row in user_history.iterrows():

        results.append(
            {
                "book_id": row["Book ID Number"],
                "title": row["Book Title"],
                "author": clean_author(
                    row["Book Title"],
                    row["Author"],
                ),
                "ddc": row["DDC Number"],
                "borrow_date": row["Borrow Date"],
                "return_date": row["Return Date"],
                "borrow_duration_days": row[
                    "Borrow Duration (days)"
                ],
            }
        )

    return results


# ==================================================
# Get popular books
# ==================================================

def get_popular_books(limit=10):
    """
    Return the most frequently borrowed books
    in the library dataset.

    Parameters:
        limit: Number of popular books to return

    Returns:
        List of popular book records
    """

    try:
        limit = int(limit)
    except (ValueError, TypeError):
        return []

    if limit <= 0:
        return []

    # Count how many borrowing records exist
    # for each book.
    popularity = (
        library_data[
            "Book ID Number"
        ]
        .astype(str)
        .str.strip()
        .value_counts()
        .head(limit)
    )

    results = []

    for book_id, borrow_count in popularity.items():

        matches = books_metadata[
            books_metadata["Book ID Number"]
            .astype(str)
            .str.strip()
            == str(book_id)
        ]

        if matches.empty:
            continue

        row = matches.iloc[0]

        results.append(
            {
                "book_id": row["Book ID Number"],
                "title": row["Book Title"],
                "author": clean_author(
                    row["Book Title"],
                    row["Author"],
                ),
                "ddc": row["DDC Number"],
                "borrow_count": int(borrow_count),
            }
        )

    return results


# ==================================================
# Test functions
# ==================================================

if __name__ == "__main__":

    # ----------------------------------------------
    # Test recommendation
    # ----------------------------------------------

    test_user = list(user_to_index.keys())[0]

    print(
        f"\nTesting recommendations for user: "
        f"{test_user}"
    )

    recommendations = recommend_books(
        test_user,
        top_n=5
    )

    for i, book in enumerate(
        recommendations,
        start=1
    ):

        print(
            f"{i}. {book['title']} | "
            f"Author: {book['author']} | "
            f"Book ID: {book['book_id']}"
        )


    # ----------------------------------------------
    # Test book search
    # ----------------------------------------------

    print("\nTesting book search...")

    search_results = search_books(
        "business",
        limit=5
    )

    for i, book in enumerate(
        search_results,
        start=1
    ):

        print(
            f"{i}. {book['title']} | "
            f"Author: {book['author']} | "
            f"Book ID: {book['book_id']}"
        )


    # ----------------------------------------------
    # Test author search
    # ----------------------------------------------

    print("\nTesting author search...")

    author_results = search_by_author(
        "Sekaran",
        limit=5
    )

    for i, book in enumerate(
        author_results,
        start=1
    ):

        print(
            f"{i}. {book['title']} | "
            f"Author: {book['author']} | "
            f"Book ID: {book['book_id']}"
        )


    # ----------------------------------------------
    # Test DDC search
    # ----------------------------------------------

    print("\nTesting DDC search...")

    ddc_results = search_by_ddc(
        "650",
        limit=5
    )

    for i, book in enumerate(
        ddc_results,
        start=1
    ):

        print(
            f"{i}. {book['title']} | "
            f"Author: {book['author']} | "
            f"DDC: {book['ddc']} | "
            f"Book ID: {book['book_id']}"
        )


    # ----------------------------------------------
    # Test book details
    # ----------------------------------------------

    print("\nTesting book details...")

    book_details = get_book_details("200001")

    if book_details:

        print(
            f"Book ID: {book_details['book_id']}"
        )

        print(
            f"Title: {book_details['title']}"
        )

        print(
            f"Author: {book_details['author']}"
        )

        print(
            f"DDC: {book_details['ddc']}"
        )

        print(
            f"ISBN: {book_details['isbn']}"
        )

        print(
            f"Published Year: "
            f"{book_details['published_year']}"
        )

        print(
            f"Publisher: {book_details['publisher']}"
        )

    else:

        print("Book not found.")


    # ----------------------------------------------
    # Test borrowing history
    # ----------------------------------------------

    print("\nTesting borrowing history...")

    history = get_user_borrowing_history(
        test_user
    )

    print(
        f"Number of records: {len(history)}"
    )

    for i, book in enumerate(
        history[:5],
        start=1
    ):

        print(
            f"{i}. {book['title']} | "
            f"Author: {book['author']} | "
            f"Book ID: {book['book_id']} | "
            f"Borrowed: {book['borrow_date']} | "
            f"Returned: {book['return_date']}"
        )


    # ----------------------------------------------
    # Test popular books
    # ----------------------------------------------

    print("\nTesting popular books...")

    popular_books = get_popular_books(
        limit=10
    )

    for i, book in enumerate(
        popular_books,
        start=1
    ):

        print(
            f"{i}. {book['title']} | "
            f"Author: {book['author']} | "
            f"Book ID: {book['book_id']} | "
            f"Borrow count: {book['borrow_count']}"
        )


    # ----------------------------------------------
    # Inspect user profile structure
    # ----------------------------------------------

    print(
        "\nInspecting user profile structure..."
    )

    print(
        "Type:",
        type(user_profiles)
    )

    print(
        "Number of profiles:",
        len(user_profiles)
    )

    print(
        "Shape:",
        user_profiles.shape
    )

    print("Columns:")

    for column in user_profiles.columns:
        print(f" - {column}")