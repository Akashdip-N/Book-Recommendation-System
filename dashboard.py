import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

import gradio as gr

load_dotenv()

books = pd.read_csv("Dataset/books_with_emotions.csv")

import os

# Path to your local fallback image
FALLBACK_IMAGE = "cover-not-found.jpeg"  # Or "Saved Files/cover-not-found.jpeg"


def process_thumbnail(url):
    # Check if the URL is missing (NaN), null, or empty
    if pd.isna(url) or not isinstance(url, str) or not url.strip():
        return (
            FALLBACK_IMAGE
            if os.path.exists(FALLBACK_IMAGE)
            else "https://via.placeholder.com/300x450?text=No+Cover"
        )

    # Clean standard HTTP to HTTPS and append parameter if not present
    clean_url = url.strip().replace("http://", "https://")
    if "&fife=w800" not in clean_url:
        clean_url += "&fife=w800"

    return clean_url


books["large_thumbnail"] = books["thumbnail"].apply(process_thumbnail)

raw_documents = TextLoader("Saved Files/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

persist_directory = "Saved Files/chroma_db"
db_books = Chroma(
    persist_directory=persist_directory,
    embedding_function=hf_embeddings
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k = initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by=["joy"], ascending=False, inplace=True)
    if tone == "Surprising":
        books_recs.sort_values(by=["surprise"], ascending=False, inplace=True)
    if tone == "Angry":
        books_recs.sort_values(by=["anger"], ascending=False, inplace=True)
    if tone == "Suspenseful":
        books_recs.sort_values(by=["fear"], ascending=False, inplace=True)
    if tone == "Sad":
        books_recs.sort_values(by=["sadness"], ascending=False, inplace=True)

    return books_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone
    )
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len (authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label = "Please enter a description of a book: ",
            placeholder = "e.g., A story about forgiveness"
        )

        category_dropdown = gr.Dropdown(
            choices = categories,
            label = "Select a category: ",
            value = "All"
        )

        tone_dropdown = gr.Dropdown(
            choices = tones,
            label = "Select a emotional tone: ",
            value = "All"
        )

        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label = "Recommended Books",
        columns = 8,
        rows = 2
    )

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs = output
    )

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": False}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

import gradio as gr

load_dotenv()

books = pd.read_csv("Dataset/books_with_emotions.csv")

import os

# Path to your local fallback image
FALLBACK_IMAGE = "cover-not-found.jpeg"  # Or "Saved Files/cover-not-found.jpeg"


def process_thumbnail(url):
    # Check if the URL is missing (NaN), null, or empty
    if pd.isna(url) or not isinstance(url, str) or not url.strip():
        return (
            FALLBACK_IMAGE
            if os.path.exists(FALLBACK_IMAGE)
            else "https://via.placeholder.com/300x450?text=No+Cover"
        )

    # Clean standard HTTP to HTTPS and append parameter if not present
    clean_url = url.strip().replace("http://", "https://")
    if "&fife=w800" not in clean_url:
        clean_url += "&fife=w800"

    return clean_url


books["large_thumbnail"] = books["thumbnail"].apply(process_thumbnail)

raw_documents = TextLoader("Saved Files/tagged_description.txt").load()
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

persist_directory = "Saved Files/chroma_db"
db_books = Chroma(
    persist_directory=persist_directory,
    embedding_function=hf_embeddings
)

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16
) -> pd.DataFrame:

    recs = db_books.similarity_search(query, k = initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    books_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category != "All":
        books_recs = books_recs[books_recs["simple_categories"] == category].head(final_top_k)
    else:
        books_recs = books_recs.head(final_top_k)

    if tone == "Happy":
        books_recs.sort_values(by=["joy"], ascending=False, inplace=True)
    if tone == "Surprising":
        books_recs.sort_values(by=["surprise"], ascending=False, inplace=True)
    if tone == "Angry":
        books_recs.sort_values(by=["anger"], ascending=False, inplace=True)
    if tone == "Suspenseful":
        books_recs.sort_values(by=["fear"], ascending=False, inplace=True)
    if tone == "Sad":
        books_recs.sort_values(by=["sadness"], ascending=False, inplace=True)

    return books_recs

def recommend_books(
        query: str,
        category: str,
        tone: str
):
    recommendations = retrieve_semantic_recommendations(
        query=query,
        category=category,
        tone=tone
    )
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len (authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str} : {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")

    with gr.Row():
        user_query = gr.Textbox(
            label = "Please enter a description of a book: ",
            placeholder = "e.g., A story about forgiveness"
        )

        category_dropdown = gr.Dropdown(
            choices = categories,
            label = "Select a category: ",
            value = "All"
        )

        tone_dropdown = gr.Dropdown(
            choices = tones,
            label = "Select a emotional tone: ",
            value = "All"
        )

        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(
        label = "Recommended Books",
        columns = 8,
        rows = 2
    )

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs = output
    )

if __name__ == "__main__":
    port = os.environ.get("PORT")

    dashboard.launch(
        server_name="0.0.0.0",
        server_port=int(port)
        if port
        else None,
    )