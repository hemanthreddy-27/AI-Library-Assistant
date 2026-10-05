from pathlib import Path
import sys

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ==================================================
# Project paths
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "t5_summarizer"
)


# ==================================================
# Select device
# ==================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ==================================================
# Load tokenizer
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)


# ==================================================
# Load trained T5 model
# ==================================================

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_DIR
)

model.to(DEVICE)
model.eval()


# IMPORTANT:
# MCP stdio transport uses stdout for JSON-RPC messages.
# Diagnostic messages must therefore go to stderr.
print(
    "T5 summarization model loaded successfully.",
    file=sys.stderr
)

print(
    f"Device: {DEVICE}",
    file=sys.stderr
)


# ==================================================
# Summarize text
# ==================================================

def summarize_text(
    text,
    max_input_length=512,
    max_summary_length=128,
    min_summary_length=20,
):
    """
    Generate a summary from input text using
    the trained T5-small summarization model.

    Parameters:
        text:
            Input text to summarize.

        max_input_length:
            Maximum number of input tokens.

        max_summary_length:
            Maximum number of tokens in the summary.

        min_summary_length:
            Minimum number of tokens in the summary.

    Returns:
        Generated summary as a string.
    """

    # Convert input to string
    text = str(text).strip()

    # Handle empty input
    if not text:
        return ""

    # T5 was trained using this prefix
    input_text = "summarize: " + text

    # Tokenize input
    inputs = tokenizer(
        input_text,
        max_length=max_input_length,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )

    # Move tensors to selected device
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Generate summary
    with torch.no_grad():

        summary_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_summary_length,
            min_length=min_summary_length,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    # Convert generated tokens to text
    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    return summary.strip()


# ==================================================
# Test summarization
# ==================================================

if __name__ == "__main__":

    test_text = """
    
    A Database Management System (DBMS) is software that allows users and applications to create, store, organize, retrieve, and manage data efficiently. Instead of storing information in separate files, a DBMS provides a structured way to maintain large collections of data while reducing duplication and improving consistency. Common DBMS operations include inserting new records, updating existing information, deleting unnecessary records, and retrieving data through queries. SQL (Structured Query Language) is widely used to communicate with relational databases. A relational DBMS stores data in tables consisting of rows and columns and uses relationships between tables to organize information. Primary keys are used to uniquely identify records, while foreign keys establish relationships between related tables. DBMSs also provide important features such as data security, integrity, backup and recovery, concurrency control, and transaction management. Transactions follow properties known as ACID: atomicity, consistency, isolation, and durability. These properties help ensure that database operations are completed reliably even when multiple users access the system at the same time. Examples of relational database systems include MySQL, PostgreSQL, Oracle Database, and Microsoft SQL Server. DBMS technology is widely used in applications such as banking, education, healthcare, e-commerce, and inventory management.

    """

    print("\nTesting text summarization...")

    summary = summarize_text(
        test_text
    )

    print("\nOriginal text:")
    print(test_text.strip())

    print("\nGenerated summary:")
    print(summary)