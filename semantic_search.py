
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")

# Create or access collection
collection = client.get_or_create_collection(
    name="semantic_documents"
)

# Load documents
documents = []
ids = []

folder = "./dataset"

for file in os.listdir(folder):
    if file.endswith(".txt"):
        with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
            documents.append(f.read())
            ids.append(file)

# Add documents to ChromaDB
embeddings = model.encode(documents).tolist()

collection.upsert(
    ids=ids,
    documents=documents,
    embeddings=embeddings
)

print("Documents added successfully!")
print("Total documents:", collection.count())

# Load queries
queries_df = pd.read_csv("./dataset/queries.csv")

# Semantic search
search_results = []
similarity_scores = []

for query in queries_df["query"]:
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    for i, doc_id in enumerate(results["ids"][0]):
        distance = results["distances"][0][i]

        search_results.append({
            "query": query,
            "document": doc_id,
            "rank": i + 1
        })

        similarity_scores.append({
            "query": query,
            "document": doc_id,
            "distance": distance
        })

# Save CSV files
pd.DataFrame(search_results).to_csv(
    "./dataset/search_results.csv", index=False
)

pd.DataFrame(similarity_scores).to_csv(
    "./dataset/similarity_scores.csv", index=False
)

print("Search results saved successfully!")
print("Similarity scores saved successfully!")
