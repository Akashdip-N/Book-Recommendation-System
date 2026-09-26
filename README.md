# 📚 Content-Based Book Recommendation System
An end-to-end Content-Based Book Recommendation Engine powered by Sentence Transformers, FAISS vector index, and a Gradio web interface. The system converts book metadata and textual descriptions into dense 384-dimensional embeddings to perform real-time semantic similarity search.

## 🏗️ Architecture Overview
```
                        ┌──────────────────────────────┐
                        │ 1. Data Acquisition          │
                        │ • Kaggle API download        │
                        │ • Cleaning & filtering       │
                        │ • Text feature synthesis     │
                        └──────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ 2. Text Embeddings           │
                        │ • SentenceTransformer        │
                        │ • Model: all-MiniLM-L6-v2    │
                        │ • 384-d L2-norm vectors      │
                        └──────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ 3. Vector Indexing           │
                        │ • FAISS IndexFlatIP          │
                        │ • Cosine similarity search   │
                        │ • Fast k-NN retrieval        │
                        └──────────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ 4. Interactive App           │
                        │ • Gradio Web UI              │
                        │ • Real-time NL Queries       │
                        │ • Display top recommendations│
                        └──────────────────────────────┘
```

## ✨ Key Features
   - Data Cleaning & Filtering: Automated dataset downloading via Kaggle API, handling missing values, and combining key text attributes (`title`, `authors`, `categories`, `description`).
   - 🧠 Semantic Embeddings: Leverages `all-MiniLM-L6-v2` to extract contextual feature vectors from metadata.
   - ⚡ Fast Similarity Search: Uses FAISS (`IndexFlatIP`) with $L_2$-normalized vectors to compute exact cosine similarities instantly.
   - 💻 Interactive UI: Built with Gradio to provide a simple web interface for natural language book discovery.


## 📁 Project Structure
```
├── Dataset/
│   ├── books.csv                      # Raw dataset from Kaggle
│   ├── books_with_categories.csv      # Filtered dataset with categories
│   ├── books_with_emotions.csv        # Filtered dataset with emotions
│   └── cleaned_books.csv              # Final cleaned dataset for embedding generation
│
├── Notebooks/
│   ├── data-exploration.ipynb         # Data downloading, cleaning & EDA
│   ├── sentimental-analysis.ipynb     # Sentiment & emotional tone analysis (joy, sadness, suspense)
│   ├── text-classification.ipynb      # Zero-shot classification (Fiction / Non-fiction tagging)
│   └── vector-search.ipynb            # Vector database setup & semantic similarity search
│
├── dashboard.py                       # Interactive Gradio web interface
├── requirements.txt                   # Dependencies
└── README.md
```
## 🚀 Getting Started
### 1️⃣ Prerequisites ⚙️
Make sure you have `Python 3.8+` installed on your system.
### 2️⃣ Installation 📦
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Akashdip-N/Book-Recommendation-System.git
cd book-recommendation-system
pip install -r requirements.txt
```
### 3️⃣ Usage & Pipeline Execution 🔄
#### Running the Notebooks 📓
To run the full end-to-end processing pipeline, execute the notebooks in the following sequential order:
1. `data-exploration.ipynb`: Downloads the dataset from Kaggle, performs exploratory data analysis (EDA), and cleans text data.
2. `vector-search.ipynb`: Converts book descriptions into vector embeddings and constructs the vector database for semantic retrieval.
3. `text-classification.ipynb`: Performs zero-shot text classification to categorize books into "Fiction" vs. "Non-fiction" filters.
4. `sentiment-analysis.ipynb`: Evaluates the emotional tone and sentiment of each book description (e.g., joyful, suspenseful, sad).

### 4️⃣ To start the Gradio app:
```bash
python dashboard.py
```

## 🛠️ Tech Stack & Dependencies
- 🤖 sentence-transformers
- ⚡ faiss-cpu (or faiss-gpu)
- 🎨 gradio
- 📊 pandas & numpy
- 📈 seaborn & matplotlib
- 📥 kaggle

## 🙏 Credits & Acknowledgments
- Course & Video Tutorial: Built following the [
LLM Course – Build a Semantic Book Recommender (Python, OpenAI, LangChain, Gradio)](https://www.youtube.com/watch?v=Q7mS1VHm3Yw) tutorial published on [freeCodeCamp.org](https://www.freecodecamp.org), created by [Dr. Jodie Burchell](https://www.linkedin.com/in/jodieburchell/) in partnership with [JetBrains](https://www.jetbrains.com/).
- Dataset: 7k Books with Metadata hosted on Kaggle by [Dylan Castillo](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata).

## 📺 Tutorial & Demo

Click the image below to watch the full course tutorial:
[![LLM Course – Build a Semantic Book Recommender](https://img.youtube.com/vi/Q7mS1VHm3Yw/maxresdefault.jpg)](https://www.youtube.com/watch?v=Q7mS1VHm3Yw "LLM Course – Build a Semantic Book Recommender")

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
✅ Contributions are welcome!
</br>✅ Please fork the repository, create a new branch, and submit a pull request with your changes.