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

## 🚀 Production Deployment & Optimization (Render Hosting)

To successfully host this application on **Render's Free Tier (512 MB RAM limit)** without triggering Out-Of-Memory (OOM) crashes or build timeouts, the following architectural optimizations were applied:

### 1. Offloaded Model Embeddings to Serverless API
* **Problem:** Running PyTorch and loading `sentence-transformers/all-MiniLM-L6-v2` locally inside the container consumed ~600–800 MB RAM during initialization, exceeding Render's 512 MB memory limit.
* **Solution:** Replaced local `HuggingFaceEmbeddings` with `HuggingFaceEndpointEmbeddings` (`langchain_huggingface`) to offload vector generation to Hugging Face's Serverless Inference API via HTTP.
* **Impact:** Reduced overall RAM footprint from **~800 MB down to ~150 MB**, eliminating OOM crashes.

### 2. Lightweight PyTorch-Free Dependency Tree
* **Problem:** Standard `pip install torch` downloads GPU/CUDA binaries, inflating the build size and container startup overhead.
* **Solution:** Removed `torch` from `requirements.txt` entirely, keeping only API-based clients (`langchain-huggingface`, `langchain-chroma`, `gradio`, `pandas`).
* **Impact:** Dramatically accelerated build times and eliminated PyTorch C++ binding allocations in memory.

### 3. Dynamic Network Port Binding
* **Problem:** Hardcoded port allocations (e.g., `7860`) cause health check failures on Render, which dynamically assigns a host port at container boot.
* **Solution:** Updated `dashboard.launch()` to bind dynamically to `0.0.0.0` and read the `PORT` environment variable injected by Render:
  ```python
  port = int(os.environ.get("PORT", 7860))
  dashboard.launch(server_name="0.0.0.0", server_port=port)

## 💻 View the Project Live

You can view the live version of this project at [Book Recommendation System](https://book-recommendation-system-x23e.onrender.com/).

📝 A small note: Since the application's backend is hosted using the free tier of Vercel, it might take one or two minutes to wake up. Sorry for the inconvenience caused.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing
✅ Contributions are welcome!
</br>✅ Please fork the repository, create a new branch, and submit a pull request with your changes.