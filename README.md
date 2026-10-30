# Semantic Document Search

## Project Overview
This project implements a Semantic Document Search system using Python, Sentence Transformers, and ChromaDB. It finds relevant documents based on the meaning of a user's query rather than relying only on exact keyword matches.

## Objectives
- Load and process text documents.
- Generate document embeddings using Sentence Transformers.
- Store embeddings in ChromaDB.
- Perform semantic search for user queries.
- Calculate similarity distances.
- Compare semantic search with keyword-based search.

## Technologies Used
- Python
- Pandas
- ChromaDB
- Sentence Transformers
- Jupyter Notebook

## Dataset
The dataset contains 10 text documents covering:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Python
- Database Management Systems
- Natural Language Processing
- Cloud Computing
- Cybersecurity
- Data Science
- Internet of Things

It also includes 15 search queries.

## Project Structure
```text
Semantic_Document_Search/
├── dataset/
│   ├── documents (text files)
│   ├── queries.csv
│   ├── search_results.csv
│   ├── similarity_scores.csv
│   ├── notebook_search_results.csv
│   ├── keyword_search_results.csv
│   └── search_comparison.csv
├── chroma_db/
├── notebook/
│   └── Week12_Semantic_Document_Search.ipynb
├── semantic_search.py
├── README.md
└── requirements.txt
```

## How to Run
1. Install the required packages:
   `pip install -r requirements.txt`
2. Run the