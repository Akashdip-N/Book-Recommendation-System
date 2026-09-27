import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import gradio as gr

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

# --- 1. Model & Vector Database Initialization ---
hf_embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=os.environ.get("HF_TOKEN")
)

persist_directory = "Saved Files/chroma_db"
db_books = Chroma(
    persist_directory=persist_directory,
    embedding_function=hf_embeddings
)

# --- 2. Dataset Preprocessing ---
books = pd.read_csv("Dataset/books_with_emotions.csv")
FALLBACK_IMAGE = "cover-not-found.jpeg"

def process_thumbnail(url):
    if pd.isna(url) or not isinstance(url, str) or not url.strip():
        return (
            FALLBACK_IMAGE
            if os.path.exists(FALLBACK_IMAGE)
            else "https://via.placeholder.com/300x450?text=No+Cover"
        )

    clean_url = url.strip().replace("http://", "https://")
    if "&fife=w800" not in clean_url:
        clean_url += "&fife=w800"

    return clean_url

books["large_thumbnail"] = books["thumbnail"].apply(process_thumbnail)

# --- 3. Recommendation Logic ---
TONE_MAP = {
    "Happy": "joy",
    "Surprising": "surprise",
    "Angry": "anger",
    "Suspenseful": "fear",
    "Sad": "sadness"
}

def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 16
) -> pd.DataFrame:
    # Vector similarity search
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract ISBNs from retrieved Chroma documents
    books_list = []
    for rec in recs:
        try:
            isbn = int(rec.page_content.strip('"').split()[0])
            books_list.append(isbn)
        except (ValueError, IndexError):
            continue

    # Filter initial candidate pool
    books_recs = books[books["isbn13"].isin(books_list)].copy()

    # Filter by category if requested
    if category and category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category]

    # Sort by emotional tone if requested
    if tone in TONE_MAP:
        emotion_col = TONE_MAP[tone]
        if emotion_col in books_recs.columns:
            books_recs.sort_values(by=[emotion_col], ascending=False, inplace=True)

    # Return top K results after filtering and sorting
    return books_recs.head(final_top_k)

def recommend_books(query: str, category: str, tone: str):
    if not query.strip():
        return []

    recommendations = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone
    )
    results = []

    for _, row in recommendations.iterrows():
        # Truncate description safely
        description = str(row.get("description", ""))
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + ("..." if len(truncated_desc_split) > 30 else "")

        # Format authors string safely
        authors_raw = str(row.get("authors", "Unknown Author"))
        authors_split = [a.strip() for a in authors_raw.split(";") if a.strip()]

        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        caption = f"{row.get('title', 'Untitled')} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# --- 4. Gradio Interface Construction ---
categories = ["All"] + sorted([c for c in books["simple_categories"].dropna().unique()])
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label="Please enter a description of a book:",
            placeholder="e.g., A story about forgiveness"
        )

        category_dropdown = gr.Dropdown(
            choices=categories,
            label="Select a category:",
            value="All"
        )

        tone_dropdown = gr.Dropdown(
            choices=tones,
            label="Select an emotional tone:",
            value="All"
        )

        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label="Recommended Books",
        columns=8,
        rows=2
    )

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    dashboard.launch(
        server_name="0.0.0.0",
        server_port=port,
    )