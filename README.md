# 🔍 Advanced Retrieval & Semantic Search

> **From Vector Embeddings to Production-Grade RAG — A complete journey through modern information retrieval techniques.**

[![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2.1-green?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12.49-red?style=for-the-badge)](https://www.llamaindex.ai/)
[![FAISS](https://img.shields.io/badge/FAISS-1.7.4-purple?style=for-the-badge)](https://github.com/facebookresearch/faiss)
[![IBM watsonx](https://img.shields.io/badge/IBM%20watsonx-Granite--4--H--Small-blue?style=for-the-badge&logo=ibm&logoColor=white)](https://www.ibm.com/watsonx)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](LICENSE)

---

## 📌 Overview

This repository contains a **comprehensive lab series** that explores the mathematical foundations, practical implementations, and advanced techniques of **semantic search and retrieval-augmented generation (RAG)** . The labs progress from basic vector similarity concepts to production-grade retrieval systems using industry-standard frameworks.

**Why this matters:** In the age of AI, **retrieval is everything**. Whether you're building a RAG system, a search engine, or a recommendation system, understanding how to efficiently find semantically similar information is critical. These labs provide hands-on experience with the tools and techniques powering modern AI applications.

### 🧠 What You'll Learn

| Concept | Lab | What You'll Build |
|---------|-----|-------------------|
| **Vector Similarity** | Similarity Search by Hand | Manual L2 distance, dot product, cosine similarity |
| **Semantic Embeddings** | Semantic Similarity with FAISS | Universal Sentence Encoder + FAISS search engine |
| **Basic Retrieval** | LangChain Retrievers | Vector Store, MMR, Similarity Threshold retrieval |
| **Advanced Retrieval** | LangChain Context Retrieval | Multi-Query, Self-Querying, Parent Document Retrievers |
| **Multi-Query Fusion** | LlamaIndex Advanced Retrievers | Vector Index, BM25, Auto-Merging, Recursive, Query Fusion |
| **Production RAG** | LlamaIndex Exercises | Custom hybrid retrievers, production pipelines |

---

## 📁 Lab Series Breakdown

### Lab 1: Similarity Search by Hand (Jupyter Notebook)
**The Mathematical Foundation**

This lab starts from **first principles** — no libraries, just math. You'll manually implement similarity metrics to understand how machines compare information.

**Key Concepts:**
- ✅ **L2 (Euclidean) Distance** — `√Σ(ai - bi)²`
- ✅ **Dot Product Similarity** — `Σ(ai * bi)`
- ✅ **Cosine Similarity** — `(a·b) / (||a||·||b||)`
- ✅ **Vector Normalization** — Scaling vectors to unit length
- ✅ **Semantic Disambiguation** — Distinguishing "software bugs" from "insect bugs"

**Why It Matters:** Before you can use libraries, you must understand the math they implement. This lab builds the intuition that every subsequent lab relies on.

---

### Lab 2: Semantic Similarity with FAISS (Jupyter Notebook)
**From Math to Production**

This lab scales up from single vectors to **thousands of documents** using the 20 Newsgroups dataset. You'll use the Universal Sentence Encoder for embeddings and FAISS for efficient indexing.

**Key Concepts:**
- ✅ **Text Preprocessing** — Cleaning real-world text data
- ✅ **Universal Sentence Encoder** — State-of-the-art sentence embeddings
- ✅ **FAISS Indexing** — `IndexFlatL2` for fast similarity search
- ✅ **Query Processing** — Finding nearest neighbors in high-dimensional space
- ✅ **Result Interpretation** — Understanding distance metrics in context

**Why It Matters:** This is the foundational pipeline for any RAG system. You'll see how raw text becomes searchable vectors in a production-ready system.

---

### Lab 3: Build a Smarter Search with LangChain Context Retrieval (Jupyter Notebook)
**Advanced Retrievers — Part 1**

This lab explores LangChain's retriever ecosystem, moving beyond simple vector search to intelligent, context-aware retrieval.

**Key Concepts:**
- ✅ **Vector Store-Backed Retriever** — Basic semantic search with ChromaDB
- ✅ **Multi-Query Retriever** — Generating multiple query variations for better recall
- ✅ **Self-Querying Retriever** — Automatic metadata filtering
- ✅ **Parent Document Retriever** — Hierarchical chunking for context preservation
- ✅ **BM25 Retriever** — TF-IDF-based keyword search with saturation

**Why It Matters:** Simple vector search isn't enough for production systems. These retrievers address real-world challenges: query ambiguity, metadata filtering, and context loss.

---

### Lab 4: Explore Advanced Retrievers in LlamaIndex (Jupyter Notebook)
**Advanced Retrievers — Part 2**

This lab pushes retrieval to the next level with LlamaIndex's advanced techniques, including sophisticated fusion methods.

**Key Concepts:**
- ✅ **Vector Index Retriever** — Semantic foundation
- ✅ **BM25 Retriever** — Keyword search with TF-IDF improvements
- ✅ **Document Summary Index** — Document-level filtering with summaries
- ✅ **Auto Merging Retriever** — Hierarchical context preservation
- ✅ **Recursive Retriever** — Following document references
- ✅ **Query Fusion Retriever** — Three fusion modes:
  - **Reciprocal Rank Fusion (RRF)** — Most robust, rank-based
  - **Relative Score Fusion** — Preserves confidence scores
  - **Distribution-Based Fusion** — Statistical normalization

**Why It Matters:** Different problems require different retrieval strategies. This lab teaches you when and how to combine them for optimal results.

---

## 📊 Comparison: Retrieval Methods at a Glance

| Retriever | Type | Best For | Key Strength |
|-----------|------|----------|--------------|
| **Vector Store** | Semantic | General Q&A | Meaning-based similarity |
| **BM25** | Keyword | Technical docs | Exact term matching |
| **Multi-Query** | Enhancement | Ambiguous queries | Multiple query formulations |
| **Self-Querying** | Intelligent | Filtered search | Automatic metadata filtering |
| **Parent Document** | Context-Aware | Long documents | Context preservation |
| **Auto Merging** | Hierarchical | Structured docs | Multi-level context |
| **Recursive** | Reference-Following | Research papers | Citation traversal |
| **Query Fusion** | Combined | Hybrid search | Multiple strategy fusion |

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|--------------|
| **Vector Databases** | ChromaDB, FAISS |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`), Universal Sentence Encoder, HuggingFace (`BAAI/bge-small-en-v1.5`) |
| **Frameworks** | LangChain, LlamaIndex |
| **LLMs** | IBM watsonx.ai (Granite-4-H-Small, Mistral) |
| **Math Libraries** | NumPy, SciPy, PyTorch |
| **Frontend** | Gradio |
| **Language** | Python 3.11+ |

---

## 📁 Project Structure

```
advanced-retrieval-labs/
│
├── 📓 Lab 1: Similarity Search by Hand.ipynb
│   ├── Manual L2 distance implementation
│   ├── Manual dot product & cosine similarity
│   ├── Vector normalization
│   └── Semantic disambiguation with "Bugs" dataset
│
├── 📓 Lab 2: Semantic Similarity with FAISS.ipynb
│   ├── 20 Newsgroups dataset exploration
│   ├── Text preprocessing pipeline
│   ├── Universal Sentence Encoder embeddings
│   ├── FAISS index creation & search
│   └── Production query system
│
├── 📓 Lab 3: Build a Smarter Search with LangChain.ipynb
│   ├── Vector Store-Backed Retriever
│   ├── Multi-Query Retriever
│   ├── Self-Querying Retriever
│   ├── Parent Document Retriever
│   └── BM25 Retriever
│
├── 📓 Lab 4: Explore Advanced Retrievers in LlamaIndex.ipynb
│   ├── Vector Index Retriever
│   ├── BM25 Retriever (detailed)
│   ├── Document Summary Index
│   ├── Auto Merging Retriever
│   ├── Recursive Retriever
│   ├── Query Fusion Retriever (3 modes)
│   └── Production RAG Pipeline Exercise
│
├── 🛒 Grocery Similarity Search (Gradio App)
│   └── similarity_search.py
│
└── 📖 README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.11 or higher
- Jupyter Notebook or JupyterLab
- IBM Cloud account (for watsonx.ai access — optional)

### Step 1: Clone the Repository

```bash
git clone https://github.com/umer302203/advanced-retrieval-labs.git
cd advanced-retrieval-labs
```

### Step 2: Create Virtual Environment

```bash
python3.11 -m venv venv
source venv/bin/activate   # Linux/Mac
# venv\Scripts\activate    # Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Launch Jupyter

```bash
jupyter lab
```

---

## 📋 Requirements (`requirements.txt`)

```
# Core Math & Data
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0

# Machine Learning & NLP
scikit-learn>=1.3.0
tensorflow>=2.0.0
tensorflow-hub>=0.16.0
sentence-transformers>=2.2.0

# Vector Search
faiss-cpu>=1.7.4
chromadb>=0.4.24

# RAG Frameworks
langchain>=0.2.1
langchain-ibm>=0.1.11
langchain-community>=0.2.1
llama-index>=0.12.49
llama-index-embeddings-huggingface>=0.5.5
llama-index-llms-ibm>=0.4.0
llama-index-retrievers-bm25>=0.5.2
rank-bm25>=0.2.2
PyStemmer>=2.2.0

# IBM watsonx
ibm-watsonx-ai>=1.1.2

# UI & Misc
gradio>=4.44.0
lark>=1.1.9
```

---

## 🎓 Key Learnings from Each Lab

### From Lab 1: The Math
- **L2 Distance** = Euclidean distance — straight-line in vector space
- **Dot Product** = Measures magnitude and direction alignment
- **Cosine Similarity** = Angle between vectors — scale-invariant
- **Normalization** = Essential for cosine similarity

### From Lab 2: Scaling Up
- **Text preprocessing** dramatically improves embedding quality
- **Universal Sentence Encoder** produces high-quality semantic embeddings
- **FAISS** enables fast similarity search at scale
- **Distance metrics** directly correlate with semantic relevance

### From Lab 3: Advanced LangChain
- **Multi-Query Retrievers** improve recall for ambiguous queries
- **Self-Querying** automatically extracts metadata filters
- **Parent Document Retrievers** preserve context in long documents
- **BM25** excels at exact keyword matching

### From Lab 4: LlamaIndex Advanced
- **Auto Merging** automatically retrieves larger contexts when multiple child chunks match
- **Recursive Retrievers** follow reference chains (citations, links)
- **Query Fusion** combines multiple retrieval strategies for improved results
- **Fusion modes** each have unique strengths: RRF (robust), Relative Score (interpretable), Distribution-Based (statistically sophisticated)

---

## 💡 Real-World Applications

| Technique | Production Use Case |
|-----------|---------------------|
| **Vector Search** | Product recommendation, content discovery |
| **Hybrid (Vector + BM25)** | Enterprise search (e.g., Elasticsearch) |
| **Self-Querying** | E-commerce filtering (price, category, rating) |
| **Parent Document** | Legal document retrieval, research papers |
| **Multi-Query** | Customer support FAQ, voice assistants |
| **Query Fusion** | Intelligent search engines (Google, Bing) |
| **Auto Merging** | Technical documentation, textbooks |
| **Recursive** | Academic research, citation networks |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Open a Pull Request

---

## 📄 License

This project is distributed under the **MIT License** — free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **IBM Skills Network** for the original lab materials and structure
- **LangChain** and **LlamaIndex** for their powerful retrieval frameworks
- **Hugging Face** for the embedding models
- **FAISS** for efficient similarity search

---

## 📬 Connect with Me

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rana-umer-05a9a9359/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/umer302203)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-FF9D00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/Umer78786)

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&height=120&section=footer&text=Advanced%20Retrieval%20Labs&fontSize=24&fontColor=white&fontAlignY=65" />
</p>

> Built with ❤️ by [Rana Umer](https://www.linkedin.com/in/rana-umer-05a9a9359/) 🚀
