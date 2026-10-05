# AI-Powered Intelligent Library Assistant using Model Context Protocol, Machine Learning and NLP

## 1. Project Overview

The **AI-Powered Intelligent Library Assistant** is a university library assistance system that combines **Machine Learning (ML)**, **Natural Language Processing (NLP)**, and the **Model Context Protocol (MCP)** to provide intelligent library-related services.

The system is designed to help users:

* Search for books using keywords.
* Search for books by author.
* Search for books using Dewey Decimal Classification (DDC).
* Retrieve detailed information about books.
* View borrowing history.
* Identify popular books.
* Receive personalized book recommendations.
* Generate summaries of textual content.

The recommendation component uses historical university library borrowing data, while the NLP component uses a fine-tuned **T5-small** model for text summarization. The MCP layer exposes the library capabilities as structured tools that can be accessed through an MCP client.

---

## 2. Project Objectives

The main objectives of the project are:

1. To develop an intelligent library assistant using Machine Learning and NLP.
2. To provide personalized book recommendations using historical borrowing behavior.
3. To handle both known-user and cold-start recommendation scenarios.
4. To provide multiple methods of book discovery and search.
5. To use NLP to generate concise summaries of textual content.
6. To integrate the library functionality using the Model Context Protocol.
7. To provide a modular architecture that can be extended with additional library services.

---

## 3. Key Features

### Book Search

Users can search for books using keywords from book titles and metadata.

### Book Details

The system can retrieve detailed information including:

* Book ID
* Book title
* Author
* DDC number
* ISBN
* Published year
* Publisher

### Author Search

Books can be searched using author names or author-related keywords.

### DDC Search

Books can be discovered using Dewey Decimal Classification numbers.

### Borrowing History

The system can retrieve a user's historical borrowing records, including:

* Book ID
* Book title
* Author
* DDC
* Borrow date
* Return date
* Borrow duration

### Popular Books

The system identifies frequently borrowed books from the historical library dataset.

### Personalized Recommendations

The recommendation engine combines multiple approaches to generate personalized book recommendations.

### Cold-Start Recommendations

For users without sufficient historical borrowing information, the system uses popularity information based on available user and academic metadata.

### Text Summarization

The system uses a T5-small sequence-to-sequence model to generate concise summaries of input text.

### MCP Integration

Library capabilities are exposed through eight MCP tools, allowing the MCP client to interact with the library assistant through a standardized tool interface.

---

## 4. System Architecture

```text
                         ┌─────────────────────────┐
                         │ University Library Data │
                         │        CSV Dataset      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Data Preprocessing      │
                         │ Cleaning & Preparation  │
                         └────────────┬────────────┘
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
                     ▼                                 ▼
          ┌─────────────────────┐           ┌─────────────────────┐
          │ Recommendation      │           │ NLP Summarization   │
          │ System              │           │ T5-small Model      │
          └──────────┬──────────┘           └──────────┬──────────┘
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │ MCP Library Tool Layer  │
                         │       8 MCP Tools       │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │      MCP Server         │
                         │    stdio transport      │
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │       MCP Client        │
                         └─────────────────────────┘
```

---

## 5. Dataset

The project uses the **University and Public Library Recommendation Dataset for Sparse and Cold-Start Research**.

Dataset source:

**Mendeley Data**

DOI:

`10.17632/6rd45bvxjy.1`

The dataset used in this project is the university library dataset:

`dataset_univLib.csv`

### Dataset Preparation

The original dataset contained:

* 52,394 records
* 22 columns

After preprocessing:

* **52,389 records**
* **22 columns**
* **0 exact duplicate records**

Five exact duplicate records were removed during preprocessing.

### Dataset Statistics

| Attribute                     |      Value |
| ----------------------------- | ---------: |
| Processed records             |     52,389 |
| Unique members                |     14,148 |
| Unique book IDs               |      8,369 |
| Unique book titles            |      7,302 |
| Unique user-book interactions |     39,067 |
| Repeated borrowing records    |     13,322 |
| Interaction sparsity          |    99.967% |
| Interaction density           |     0.033% |
| Earliest borrowing date       | 2018-01-02 |
| Latest borrowing date         | 2025-04-09 |

### Dataset Columns

The dataset contains information related to:

* Membership Number
* Book ID Number
* Transaction year
* Borrow Duration
* Book Title
* Borrow Date
* Return Date
* Department Code
* Cumulative GPA
* Total Credit Cumulative
* Gender
* Place of Birth
* Date of Birth
* Academic Year
* Program Code
* ISBN
* DDC Number
* Author
* Published Year
* Publisher Name
* Location Code
* Borrowed

---

## 6. Data Preprocessing

The preprocessing stage includes:

1. Removing exact duplicate records.
2. Cleaning text fields.
3. Converting numeric fields into appropriate numeric formats.
4. Converting date fields into datetime representations.
5. Handling missing publisher values.
6. Preserving other missing values where appropriate.
7. Preparing user-book interaction data for recommendation modeling.

The preprocessed dataset is stored at:

```text
datasets/dataset_univLib_preprocessed.csv
```

---

# 7. Recommendation System

The recommendation system was designed for a highly sparse user-book interaction environment.

The processed data contains:

* 13,902 users in the training data.
* 8,246 books represented in the recommendation model.
* 38,324 training interactions.
* A very sparse user-item interaction matrix.

The overall interaction matrix has approximately **99.967% sparsity**, making both personalized recommendation and cold-start handling important components of the system.

---

## 8. Recommendation Approaches

The system uses multiple recommendation techniques.

### 8.1 Item-Based Collaborative Filtering

Item-based collaborative filtering uses similarities between books based on user borrowing behavior.

Cosine similarity is used to identify books that have similar borrowing patterns.

The resulting item similarity artifact is stored as:

```text
models/library_recommender_artifacts/book_similarity.joblib
```

---

### 8.2 Singular Value Decomposition

SVD is used to learn latent representations of users and books from the user-item interaction matrix.

The trained SVD model is stored as:

```text
models/library_recommender_artifacts/svd_model.joblib
```

The model uses 50 latent components.

---

### 8.3 Content-Based Recommendation

The content-based component uses book metadata including:

* Book title
* Author
* DDC number
* Publisher name

TF-IDF representations are used to calculate content similarity between books.

---

### 8.4 Hybrid Recommendation

The system combines recommendation signals from multiple approaches.

Two hybrid configurations were evaluated:

* Original Hybrid
* Normalized Hybrid

The normalized hybrid approach improved the original hybrid across the evaluated Precision@5, Recall@5, Precision@10, and Recall@10 metrics on the known-user evaluation population.

---

## 9. Cold-Start Recommendation

Cold-start users are users for whom sufficient historical borrowing information is not available.

The evaluation identified:

* 13,902 training users
* 408 test users
* 162 known test users
* 246 cold-start test users

The cold-start strategy uses a popularity hierarchy based on available academic information:

```text
Program Popularity
        ↓
Department Popularity
        ↓
Global Popularity
```

This allows the system to provide recommendations even when personalized interaction history is unavailable.

The cold-start evaluation achieved:

* Precision@5: approximately 0.035
* Recall@5: approximately 0.109
* Precision@10: approximately 0.022
* Recall@10: approximately 0.135
* Recommendation coverage: 100% of evaluated cold-start users

The cold-start results are evaluated on a separate population from the known-user recommendation results.

---

## 10. Recommendation Evaluation

The recommendation system was evaluated using a temporal train/test split.

### Training Data

Training interactions consist of records before 2025.

* 38,324 interactions
* Through 2024-12-26

### Test Data

Testing uses 2025 interactions.

* 743 interactions
* From 2025-01-03 to 2025-04-09

### Known-User Evaluation

The known-user evaluation population contains 162 users.

| Method            | Precision@5 | Recall@5 | Precision@10 | Recall@10 |
| ----------------- | ----------: | -------: | -----------: | --------: |
| Popularity        |      0.0012 |   0.0021 |       0.0006 |    0.0021 |
| Item-CF           |      0.0115 |   0.0377 |       0.0076 |    0.0469 |
| SVD               |      0.0111 |   0.0337 |       0.0093 |    0.0635 |
| Original Hybrid   |      0.0123 |   0.0427 |       0.0080 |    0.0516 |
| Normalized Hybrid |      0.0148 |   0.0445 |       0.0099 |    0.0620 |
| Content-Based     |      0.0000 |   0.0000 |       0.0000 |    0.0000 |

The evaluation shows that different approaches perform differently across Precision and Recall metrics. Therefore, the metrics should be interpreted together rather than using a single metric to characterize the recommendation system.

---

# 11. NLP Text Summarization

The NLP component uses a **T5-small** sequence-to-sequence model.

The model was trained using the **CNN/DailyMail** dataset.

### Dataset

The full CNN/DailyMail dataset contains:

* Training: 287,113 examples
* Validation: 13,368 examples
* Test: 11,490 examples

A project subset was used for training and evaluation:

* Training: 20,000 examples
* Validation: 2,000 examples
* Test: 2,000 examples

### Model

```text
Model: google-t5/t5-small
Parameters: 60,506,624
```

### Training Configuration

| Parameter              | Value     |
| ---------------------- | --------- |
| Model                  | T5-small  |
| Training examples      | 20,000    |
| Validation examples    | 2,000     |
| Test examples          | 2,000     |
| Epochs                 | 1         |
| Learning rate          | 5e-5      |
| Batch size             | 2         |
| Gradient accumulation  | 4         |
| Effective batch size   | 8         |
| Maximum input length   | 512       |
| Maximum summary length | 128       |
| Device                 | Apple MPS |

Training took approximately **76.08 minutes**.

Average training loss:

```text
2.0664
```

Validation loss:

```text
1.7900
```

---

## 12. Summarization Evaluation

The model was evaluated on 2,000 test examples.

| Metric     |  Score |
| ---------- | -----: |
| ROUGE-1    | 0.4132 |
| ROUGE-2    | 0.1919 |
| ROUGE-L    | 0.2905 |
| ROUGE-Lsum | 0.3559 |

Evaluation took approximately **50.79 minutes**.

### Important Limitation

The summarization model was trained and evaluated using the **CNN/DailyMail news dataset**, rather than a university-library-specific dataset.

Therefore, the summarization results demonstrate general text summarization capability but should not be interpreted as a specialized evaluation of academic or library-document summarization.

---

# 13. Model Artifacts

Recommendation model artifacts are stored in:

```text
models/library_recommender_artifacts/
```

Important artifacts include:

```text
book_similarity.joblib
book_factors.joblib
book_to_index.joblib
books_metadata.joblib
department_book_popularity.joblib
global_popularity.joblib
program_book_popularity.joblib
svd_model.joblib
user_factors.joblib
user_profiles.joblib
user_to_index.joblib
```

The trained T5 summarization model is stored in:

```text
models/t5_summarizer/
```

---

# 14. Model Context Protocol Integration

The Model Context Protocol layer provides a standardized interface for accessing the library assistant capabilities.

The MCP server is implemented in:

```text
mcp_server/server.py
```

The library tool wrappers are implemented in:

```text
mcp_server/library_tools.py
```

The MCP client is implemented in:

```text
mcp_client.py
```

The MCP server uses **stdio transport**.

---

# 15. MCP Tools

The system exposes exactly eight MCP tools.

### 1. `search_books`

Searches for books using a text query.

Example:

```text
search_books(query="business", limit=5)
```

---

### 2. `get_book_details`

Retrieves detailed metadata for a specific book.

Example:

```text
get_book_details(book_id="200001")
```

---

### 3. `search_by_author`

Searches for books associated with an author.

Example:

```text
search_by_author(author="Kotler", limit=5)
```

---

### 4. `search_by_ddc`

Searches for books using a DDC classification.

Example:

```text
search_by_ddc(ddc="650", limit=5)
```

---

### 5. `get_user_borrowing_history`

Retrieves historical borrowing information for a user.

Example:

```text
get_user_borrowing_history(user_id=10001)
```

---

### 6. `get_popular_books`

Returns frequently borrowed books.

Example:

```text
get_popular_books(limit=5)
```

---

### 7. `recommend_books`

Generates personalized recommendations for a user.

Example:

```text
recommend_books(user_id=10001, top_n=5)
```

---

### 8. `summarize_text`

Generates a summary of supplied text using the T5 model.

Example:

```text
summarize_text(
    text="...",
    max_input_length=512,
    max_summary_length=128,
    min_summary_length=20
)
```

---

# 16. Project Structure

```text
AI-Library-Assistant/
│
├── datasets/
│   └── dataset_univLib_preprocessed.csv
│
├── models/
│   ├── library_recommender_artifacts/
│   │   ├── book_similarity.joblib
│   │   ├── book_factors.joblib
│   │   ├── book_to_index.joblib
│   │   ├── books_metadata.joblib
│   │   ├── department_book_popularity.joblib
│   │   ├── global_popularity.joblib
│   │   ├── program_book_popularity.joblib
│   │   ├── svd_model.joblib
│   │   ├── user_factors.joblib
│   │   ├── user_profiles.joblib
│   │   └── user_to_index.joblib
│   │
│   └── t5_summarizer/
│       ├── config.json
│       ├── generation_config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── training_metadata.json
│
├── recommendation/
│   └── recommend.py
│
├── summarization/
│   └── summarize.py
│
├── mcp_server/
│   ├── library_tools.py
│   └── server.py
│
├── notebooks/
│   └── Library_AI_Project.ipynb
│
├── mcp_client.py
│
├── requirements.txt
│
└── README.md
```

---

# 17. Installation

## Requirements

The project requires Python 3 and the dependencies listed in:

```text
requirements.txt
```

Install the dependencies using:

```bash
pip install -r requirements.txt
```

For the project environment used during development, Python 3.13.7 was used on macOS with Apple Silicon.

The recommendation and summarization components can use Apple MPS acceleration when available.

---

# 18. Running the Project

Navigate to the project directory:

```bash
cd AI-Library-Assistant
```

Run the MCP client:

```bash
python3 mcp_client.py
```

The MCP client automatically starts the local MCP server through stdio and performs demonstration calls to all eight registered tools.

---

# 19. MCP Verification

The MCP client was used to verify the complete tool interface.

The verification confirmed:

```text
MCP server connected successfully.

Total MCP tools: 8
```

All eight tools returned successful results:

```text
TEST 1: search_books
STATUS: SUCCESS

TEST 2: get_book_details
STATUS: SUCCESS

TEST 3: search_by_author
STATUS: SUCCESS

TEST 4: search_by_ddc
STATUS: SUCCESS

TEST 5: get_user_borrowing_history
STATUS: SUCCESS

TEST 6: get_popular_books
STATUS: SUCCESS

TEST 7: recommend_books
STATUS: SUCCESS

TEST 8: summarize_text
STATUS: SUCCESS
```

The MCP communication was also verified to operate correctly using JSON-RPC over stdio.

---

# 20. Technical Verification

The project implementation was additionally verified through:

* Python module import checks.
* Required file and model artifact checks.
* Recommendation function checks.
* Invalid-input handling checks.
* MCP server connection checks.
* MCP tool registration checks.
* MCP tool execution checks.
* Dependency version checks.
* T5 model loading checks.
* Apple MPS device detection.

The verification completed successfully for the implemented components.

---

# 21. Limitations

### Historical Dataset

The recommendation system is based on historical borrowing data and does not provide real-time library inventory or availability information.

### No Live Borrow/Return System

The current implementation does not directly connect to a live university library management system.

### Sparse Interactions

The dataset contains highly sparse user-book interactions, which makes personalized recommendation challenging.

### Cold-Start Users

Users with little or no historical borrowing information cannot receive fully personalized collaborative recommendations. The system therefore uses popularity-based strategies for cold-start cases.

### General-Domain Summarization Dataset

The T5 model was trained using CNN/DailyMail, which contains news articles rather than university library documents.

### Dataset Metadata Quality

Some book metadata contains inconsistencies. The recommendation output layer therefore performs limited metadata cleanup, such as replacing cases where the author field exactly duplicates the book title with `"Unknown"`.

### Static Model Artifacts

The trained models are stored locally and are not automatically updated when new borrowing records become available.

---

# 22. Future Enhancements

Possible future improvements include:

1. Integration with a live Library Management System.
2. Real-time book availability checking.
3. Live borrowing and return functionality.
4. User authentication.
5. Web or mobile user interface.
6. Conversational AI interface.
7. Fine-tuning the summarization model on academic and library-specific documents.
8. More advanced transformer-based recommendation models.
9. Continuous model retraining using new borrowing data.
10. Improved explainability for recommendations.
11. Multilingual library search and summarization.
12. Integration with additional MCP tools and external library services.
13. Improved handling of metadata inconsistencies.
14. Online evaluation using real user interactions.

---

# 23. Conclusion

The **AI-Powered Intelligent Library Assistant** demonstrates how Machine Learning, Natural Language Processing, and the Model Context Protocol can be combined to build an intelligent library assistance system.

The recommendation component provides multiple recommendation strategies, including collaborative filtering, SVD, content-based methods, hybrid recommendation, and cold-start popularity-based recommendation.

The NLP component provides text summarization using a trained T5-small model.

The MCP architecture exposes the library functionality through eight structured tools, allowing an MCP client to interact with the recommendation, search, borrowing-history, popularity, and summarization capabilities through a unified interface.

The completed implementation provides a foundation that can be extended into a real-time university library assistant by connecting the system to live library databases, authentication services, and user-facing applications.
