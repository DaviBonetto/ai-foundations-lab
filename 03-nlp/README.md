# 💬 Natural Language Processing

A deep dive into Natural Language Processing, exploring how machines understand and generate human language. This section covers foundational text processing techniques, attention mechanisms, and the transformer architecture that powers modern NLP systems.

## 📋 Overview

Natural Language Processing bridges the gap between human communication and machine understanding. This section progresses from fundamental text representations to state-of-the-art transformer models, implementing key components from scratch and applying them to real-world language tasks.

## 📂 Contents

### Text Processing Fundamentals

#### 🔤 **Word Embeddings Implementation**
*File: `word-embeddings-implementation.ipynb`*

Building dense vector representations of words that capture semantic meaning:
- One-hot encoding limitations and the curse of dimensionality
- Word2Vec: Skip-gram and CBOW architectures
- Training embeddings on text corpora
- Semantic relationships: king - man + woman ≈ queen
- Cosine similarity and nearest neighbor search
- Visualization with t-SNE and PCA
- Subword tokenization: handling out-of-vocabulary words

**Key Concepts:** Distributional semantics, context windows, negative sampling

---

#### 👁️ **Attention Mechanism**
*File: `attention-mechanism.ipynb`*

The fundamental building block of modern NLP:
- Sequence-to-sequence problems and their challenges
- Attention as soft alignment between input and output
- Query, Key, Value paradigm
- Attention score computation (dot-product, additive)
- Softmax normalization and weighted context vectors
- Self-attention: attending to one's own sequence
- Multi-head attention: parallel attention mechanisms

**Key Concepts:** Alignment scores, context vectors, information bottleneck

---

### Transformers

#### 🏗️ **Transformer Architecture Study**
*File: `transformer-architecture-study.ipynb`*

Understanding the architecture that revolutionized NLP:
- Positional encoding: injecting sequence order information
- Multi-head self-attention layers
- Feed-forward networks and residual connections
- Layer normalization for training stability
- Encoder-decoder structure
- Masking strategies: padding mask, look-ahead mask
- Training objectives and loss functions
- Attention visualization and interpretability

**Key Concepts:** Self-attention, positional embeddings, encoder-decoder, parallelization

---

#### 🎯 **Fine-Tuning Project**
*File: `fine-tuning-project.ipynb`*

Adapting pre-trained language models for specific tasks:
- Transfer learning in NLP: pre-training + fine-tuning paradigm
- Loading pre-trained transformers (BERT, GPT, RoBERTa)
- Task-specific heads: classification, token labeling, generation
- Fine-tuning strategies: learning rates, layer freezing
- Data preparation and tokenization
- Evaluation metrics for NLP tasks
- Practical applications: sentiment analysis, named entity recognition, text classification

**Key Concepts:** Transfer learning, task adaptation, contextual embeddings

---

## 🎯 Learning Objectives

By working through these implementations, you will understand:

- ✅ How machines represent and process natural language
- ✅ The evolution from word embeddings to contextual representations
- ✅ Why attention mechanisms are central to modern NLP
- ✅ The transformer architecture and its revolutionary impact
- ✅ How to fine-tune pre-trained models for real-world tasks

## 🛠️ Technologies Used

- **Python 3.8+**
- **PyTorch / TensorFlow** - Deep learning frameworks
- **Transformers (Hugging Face)** - Pre-trained models and tokenizers
- **NumPy** - Numerical computations
- **NLTK / spaCy** - Text preprocessing
- **Matplotlib / Seaborn** - Attention visualizations
- **Jupyter Notebooks** - Interactive experimentation

## 🚀 Getting Started
```bash
# Navigate to the NLP directory
cd 03-nlp

# Install required packages
pip install torch transformers nltk spacy numpy matplotlib seaborn jupyter

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Launch Jupyter Notebook
jupyter notebook
```

Then open any `.ipynb` file to explore the implementations.

## 📊 Repository Structure
```
03-nlp/
├── text-processing/
│   ├── attention-mechanism.ipynb
│   └── word-embeddings-implementation.ipynb
├── transformers/
│   ├── fine-tuning-project.ipynb
│   └── transformer-architecture-study.ipynb
└── README.md (you are here)
```

## 💡 Key Insights

**The Power of Attention:**
- **Context Awareness**: Attention enables models to focus on relevant parts of the input
- **Long-Range Dependencies**: Overcomes RNN limitations with sequential processing
- **Interpretability**: Attention weights reveal what the model focuses on

**Why Transformers Dominate:**
- **Parallelization**: Unlike RNNs, transformers process sequences in parallel
- **Scalability**: Larger models consistently improve performance
- **Transfer Learning**: Pre-training on massive corpora enables few-shot adaptation

**From Word2Vec to BERT:**
- Word2Vec: Static embeddings, one vector per word
- ELMo: Contextual embeddings from bidirectional LSTMs
- BERT: Deep bidirectional transformers, masked language modeling
- GPT: Autoregressive transformers for generation

## 📚 Learning Resources

These implementations are inspired by and aligned with:
- **Stanford CS224N: Natural Language Processing with Deep Learning** (Chris Manning)
- **Attention Is All You Need** (Vaswani et al., 2017) - The transformer paper
- **BERT: Pre-training of Deep Bidirectional Transformers** (Devlin et al., 2018)
- **Speech and Language Processing** (Jurafsky & Martin)

## 🔗 Navigation

[← Previous: Deep Learning](../02-deep-learning/README.md) | [Next: Reinforcement Learning →](../04-reinforcement-learning/README.md)

---

**Part of the [AI Foundations Lab](../README.md) project - A self-directed journey through ML, DL, NLP, and RL.**
