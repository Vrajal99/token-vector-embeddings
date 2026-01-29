# Tokenization and Embeddings for NLP - Phase 1

# Understanding Tokenization and Embeddings for NLP

## Project Description:
This project explores and compares core Natural Language Processing (NLP) techniques, including tokenization and various vector embedding methods. It leverages popular Python libraries such as NumPy, Hugging Face Transformers, OpenAI's Tiktoken, and Sentence Transformers to demonstrate advanced text representation.

## Table of Contents:
- [Introduction](#introduction)
- [Key Concepts](#key-concepts)
- [Technologies Used](#technologies-used)
- [Demonstrations](#demonstrations)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)

## Introduction:
This notebook provides a practical overview of how text data is processed for machine learning models, specifically focusing on the transformation of raw text into numerical representations (vectors/embeddings) through tokenization.

## Key Concepts:
- **Tokens**: Basic data or linguistic units derived from raw text.
- **Vectors**: Mathematical frameworks, often 1-D arrays, used for machine processing of numerical data.
- **Embeddings**: Special vectors that capture semantic meaning and context of tokens, allowing models to understand relationships between words and phrases.
- **Tokenization**: The process of breaking down text into smaller units (tokens).
- **Vector Operations**: Basic mathematical operations on vectors (addition, scalar multiplication).

## Technologies Used:
- **NumPy**: For numerical operations and array manipulations.
- **Hugging Face Transformers**: For advanced tokenization using pre-trained models (e.g., Llama 2 tokenizer).
- **OpenAI Tiktoken**: For tokenization specifically designed for OpenAI's GPT models.
- **Sentence Transformers**: For generating high-quality sentence embeddings.
- **OpenAI API**: For accessing advanced embedding models (e.g., `text-embedding-3-small`).

## Demonstrations:

The notebook walks through the following examples:

1.  **Basic Vector Operations with NumPy**: Illustrates fundamental vector arithmetic.
    - Vector creation
    - Vector addition
    - Scalar multiplication

2.  **Tokenization with Hugging Face Transformers**: Demonstrates how text is converted into tokens using a tokenizer from the Hugging Face library.
    - Example using `meta-llama/Llama-2-7b-chat-hf` tokenizer.

3.  **Tokenization with OpenAI Tiktoken**: Shows tokenization using OpenAI's `tiktoken` library, optimized for GPT models.
    - Example using `gpt-4` encoding.

4.  **Sentence Embeddings with Sentence Transformers**: Generates dense vector representations for sentences using a pre-trained sentence transformer model.
    - Example using `'sentence-transformers/all-MiniLM-L6-v2'`.

5.  **Sentence Embeddings with OpenAI API**: Utilizes the OpenAI API to create high-dimensional embeddings for sentences.
    - Example using `'text-embedding-3-small'` model.

## Setup and Installation:
To run this notebook, you'll need to install the following Python libraries:

```bash
pip install numpy transformers tiktoken sentence-transformers openai
```

For Hugging Face and OpenAI API usage, you will also need API tokens/keys:
- **Hugging Face Token**: Obtain a token from [Hugging Face](https://huggingface.co/settings/tokens) and replace `HF_TOKEN` in the code.
- **OpenAI API Key**: Obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys) and replace `Open_API+key` in the code.

## Usage:
1.  Clone this repository:
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```
2.  Open the `.ipynb` file in Google Colab or your preferred Jupyter environment.
3.  Install the required libraries.
4.  Replace placeholder API tokens/keys with your actual credentials.
5.  Run the cells sequentially to understand tokenization, vector operations, and embeddings.
