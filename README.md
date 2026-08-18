# 🧠 My Deep Dive into Advanced Information Retrieval

> *"Understanding how to find meaning in text — from vector math to production RAG systems."*

---

## 📌 Why I Built This

**In the age of AI, retrieval is everything.** Whether it's semantic search, RAG pipelines, or recommendation engines, the ability to efficiently find semantically similar information is the backbone of modern AI applications.

This repository documents my **personal learning journey** through the mathematical foundations, practical implementations, and advanced techniques of **semantic search and retrieval-augmented generation (RAG)** . Instead of just reading theory, I built everything from scratch — starting with manual vector calculations and progressing to production-grade retrieval systems.

**What I discovered:** Simple vector search isn't enough for real-world applications. The real challenge lies in combining multiple retrieval strategies, preserving context, and making systems intelligent enough to understand both meaning and metadata.

---

## 🎯 What I Explored

### 1. The Math — Building Similarity from Scratch
**Before using any library, I needed to understand what was happening under the hood.**

I implemented similarity metrics manually:
- **L2 (Euclidean) Distance** — The straight-line distance between vectors
- **Dot Product Similarity** — Measures magnitude and direction alignment
- **Cosine Similarity** — The angle between vectors, independent of magnitude
- **Vector Normalization** — Scaling vectors to unit length

**The "Bugs" Test:** I used a clever dataset of four sentences that all start with "Bugs" but refer to different concepts (software vs insects). The result? Both L2 Distance and Cosine Similarity successfully grouped the software bugs together and the insects together — proving that semantic embeddings actually capture meaning, not just keywords.

---

### 2. Scaling Up — Semantic Search with FAISS
**From single vectors to thousands of documents.**

I built a complete semantic search engine using the 20 Newsgroups dataset (~20,000 documents):
- Preprocessed real-world text data (removing email headers, punctuation, normalizing)
- Generated embeddings using the **Universal Sentence Encoder**
- Indexed everything with **FAISS** for fast similarity search

**The Result:** A search engine that understands "motorcycle" and finds relevant posts about buying a first bike, even when the exact word "motorcycle" appears only once or twice in the text.

---

### 3. LangChain Retrievers — Smarter Than Basic Search
**Simple vector search wasn't enough for real-world scenarios.**

I explored LangChain's retriever ecosystem to handle common challenges:

| Retriever | What It Does | When I Used It |
|-----------|--------------|----------------|
| **Vector Store** | Basic semantic search | General-purpose similarity |
| **Multi-Query** | Generates query variations | When the query is ambiguous |
| **Self-Querying** | Extracts metadata filters automatically | When I needed to filter by attributes |
| **Parent Document** | Retrieves larger context chunks | For long documents where small chunks lose context |
| **BM25** | Exact keyword matching | Technical documents where specific terms matter |

**Key Insight:** The Multi-Query Retriever significantly improved recall when the query could be phrased in multiple ways. The Parent Document Retriever solved the "lost in the middle" problem for long documents.

---

### 4. LlamaIndex Advanced Retrievers — The Next Level
**This is where things got really interesting.**

I explored LlamaIndex's advanced retrieval techniques, including sophisticated fusion methods:

**Core Retrievers:**
- **Vector Index Retriever** — The semantic foundation
- **BM25 Retriever** — Keyword-based with TF-IDF saturation
- **Document Summary Index** — Document-level filtering using summaries
- **Auto Merging Retriever** — Hierarchical context preservation
- **Recursive Retriever** — Following document references and citations

**Query Fusion Retriever (The Game Changer):**
This was the most exciting discovery. It combines results from multiple retrievers using three fusion modes:

| Mode | How It Works | Best For |
|------|--------------|----------|
| **Reciprocal Rank Fusion (RRF)** | Combines rankings using reciprocal ranks | Most robust, scale-invariant |
| **Relative Score** | Normalizes scores by max score | Preserves confidence information |
| **Distribution-Based** | Statistical normalization (z-score) | Most sophisticated, handles outliers |

**My Takeaway:** Different problems require different retrieval strategies. The power lies in knowing when to use each — and when to combine them.

---

### 5. Building a Production RAG Pipeline
**The final challenge: putting everything together.**

I built a production-ready RAG pipeline that:
- Routes queries to the right retriever based on question type
- Retrieves documents from multiple sources
- Generates responses using IBM watsonx.ai (Granite-4-H-Small)
- Evaluates pipeline performance with success metrics

**The Evaluation:** The pipeline achieved 100% success rate on test queries, with automatic query routing working effectively for semantic, comprehensive, and specific question types.

---

## 🛠️ Tech Stack Used

| Category | Tools |
|----------|-------|
| **Vector Databases** | ChromaDB, FAISS |
| **Embeddings** | Universal Sentence Encoder, Sentence Transformers (`all-MiniLM-L6-v2`), HuggingFace (`BAAI/bge-small-en-v1.5`) |
| **Frameworks** | LangChain, LlamaIndex |
| **LLMs** | IBM watsonx.ai (Granite-4-H-Small, Mistral) |
| **Math** | NumPy, SciPy, PyTorch |
| **Frontend** | Gradio |
| **Language** | Python 3.11+ |

---

## 📂 Repository Structure

```
advanced-retrieval-learning/
│
├── 📓 01-similarity-search-by-hand/          # The math, built from scratch
│   └── Similarity Search by Hand.ipynb
│
├── 📓 02-semantic-search-with-faiss/         # Scaling up to thousands of docs
│   └── Semantic Similarity with FAISS.ipynb
│
├── 📓 03-langchain-retrievers/               # Smarter retrieval with LangChain
│   └── Build a Smarter Search with LangChain Context Retrieval.ipynb
│
├── 📓 04-llamaindex-advanced-retrievers/     # Advanced techniques & fusion
│   └── Explore Advanced Retrievers in LlamaIndex.ipynb
│
├── 🛒 grocery-similarity-search/             # Practical Gradio app
│   └── similarity_search.py
│
└── 📖 README.md
```

---

## 🔑 Key Insights From This Journey

### 1. Similarity Metrics Are Not Interchangeable
- **L2 Distance** works well when magnitude matters
- **Cosine Similarity** is better when only direction matters
- **Dot Product** combines both magnitude and direction

### 2. Retrieval Is a Multi-Stage Process
- Stage 1: Fast filtering (vector search, BM25)
- Stage 2: Re-ranking (cross-encoders, fusion)
- Stage 3: Context injection (RAG)

### 3. Fusion Techniques Make a Difference
- **RRF** is the most robust choice for production
- **Relative Score** works well when you trust your embedding model
- **Distribution-Based** handles complex score distributions

### 4. Context Preservation Is Critical
- Parent Document Retriever solves the "chunks losing meaning" problem
- Auto Merging Retriever automatically combines related chunks
- Recursive Retriever follows references for comprehensive retrieval

### 5. Production RAG Requires:
- Query routing
- Multiple retrieval strategies
- Evaluation metrics
- Graceful fallback handling

---

## 💡 Real-World Applications

| What I Built | Could Be Used For |
|--------------|-------------------|
| Manual similarity metrics | Understanding retrieval math |
| FAISS search engine | Content discovery, recommendation |
| LangChain retrievers | Enterprise search, customer support |
| LlamaIndex advanced retrievers | Research papers, legal documents |
| Query Fusion | Smart search engines |
| Production RAG pipeline | AI assistants, chatbots |

---

## 🚀 What I'm Building Next

- **Multi-modal retrieval** — combining text, image, and audio search
- **Real-time RAG** — streaming retrieval for interactive applications
- **Evaluation frameworks** — measuring retrieval quality at scale
- **Vector database benchmarking** — comparing ChromaDB, FAISS, and others

---

## 📬 Connect With Me

I'm always interested in:
- Discussing retrieval systems and RAG architectures
- Collaborating on open-source AI projects
- Learning from others' experiences in information retrieval

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rana-umer-05a9a9359/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/umer302203)

---

> *"The best way to understand retrieval is to build it — from the math to the production system."*

— Rana Umer

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=footer&text=Retrieval%20is%20Everything&fontSize=24&fontColor=white&fontAlignY=65" />
</p>
