# 🤖 Ollama Streamlit Chatbot

A powerful Streamlit-based chat interface for interacting with local Ollama models. Features both free chat and RAG (Retrieval-Augmented Generation) capabilities with semantic search.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- 🗨️ **Free Chat Mode**: Direct conversation with Ollama models
- 📚 **RAG Mode**: Context-aware responses using document retrieval
- 🔍 **Semantic Search**: Uses sentence transformers for intelligent document matching
- 🧠 **Memory Management**: CPU/GPU control for systems with limited VRAM
- 📊 **Model Selection**: Easy switching between installed Ollama models
- 💾 **Chat History**: Persistent conversation tracking
- 🎨 **Clean UI**: Intuitive Streamlit interface with expandable source documents

## 📋 Prerequisites

- Python 3.8 or higher
- [Ollama](https://ollama.ai/) installed and running locally
- At least one Ollama model installed

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd ollama-streamlit-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit requests numpy sentence-transformers
```

### 3. Install Ollama

Download and install Ollama from [https://ollama.ai](https://ollama.ai)

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download the installer from the official website.

### 4. Pull an Ollama Model

```bash
# Recommended models based on your system:

# Low VRAM (< 4GB) - Small & Fast
ollama pull tinyllama    # ~637MB
ollama pull phi          # ~2.7GB

# Medium VRAM (4-8GB)
ollama pull llama2       # ~3.8GB
ollama pull mistral      # ~4.1GB

# High VRAM (8GB+)
ollama pull llama2:13b   # ~7.3GB
ollama pull codellama    # ~3.8GB
```

## 🎯 Usage

### Start Ollama Server

```bash
ollama serve
```

Leave this running in a terminal.

### Run the Streamlit App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 🎮 Using the App

### Free Chat Mode

1. Select a model from the sidebar
2. Configure memory settings if needed:
   - Check "Force CPU Mode" for systems with limited GPU memory
   - Adjust "GPU Layers" slider for hybrid CPU/GPU processing
3. Type your message in the chat input
4. Get instant responses from the AI

### Document-Based (RAG) Mode

1. Switch to "Document-based (RAG)" mode in the sidebar
2. Click "Load Documents" to initialize the knowledge base
3. Adjust the number of context documents (1-5)
4. Ask questions - the AI will answer based on the loaded documents
5. Expand "View Source Documents" to see which documents were used

## ⚙️ Configuration

### Memory Settings

**Force CPU Mode:**
- Runs entirely on CPU (no GPU)
- Slower but uses no GPU memory
- Recommended for systems with limited VRAM

**GPU Layers:**
- `-1`: Automatic (use all available GPU)
- `0`: CPU only
- `1-50`: Hybrid mode (partial GPU offload)

### Troubleshooting Memory Issues

If you see: `model requires more system memory than is currently available`

**Solutions:**
1. ✅ Check "Force CPU Mode" (easiest)
2. ✅ Use a smaller model (`tinyllama`, `phi`)
3. ✅ Reduce GPU layers to 10-20
4. ✅ Close other GPU applications
5. ✅ Set environment variable:
   ```bash
   export OLLAMA_NUM_GPU=0
   ollama serve
   ```

## 📁 Project Structure

```
ollama-streamlit-chatbot/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .gitignore            # Git ignore rules
```

## 🔧 Technical Details

### Technologies Used

- **Streamlit**: Web UI framework
- **Ollama**: Local LLM inference
- **Sentence Transformers**: Semantic embeddings (`all-MiniLM-L6-v2`)
- **NumPy**: Similarity calculations
- **Requests**: HTTP client for Ollama API

### How RAG Works

1. Documents are loaded from a JSON source
2. Each document is encoded into embeddings using SentenceTransformer
3. User queries are encoded similarly
4. Cosine similarity finds the most relevant documents
5. Retrieved documents are included in the prompt context
6. Ollama generates a response based on the context

## 📊 System Requirements

### Minimum
- **CPU**: 2+ cores
- **RAM**: 8GB
- **Disk**: 2GB free space
- **GPU**: Optional (can run on CPU)

### Recommended
- **CPU**: 4+ cores
- **RAM**: 16GB
- **Disk**: 10GB free space
- **GPU**: 4GB+ VRAM (NVIDIA/AMD)

## 🐛 Common Issues

### "Error fetching models"
**Solution**: Make sure Ollama is running
```bash
ollama serve
```

### "Connection refused"
**Solution**: Check if Ollama is accessible
```bash
curl http://localhost:11434/api/tags
```

### "Model not found"
**Solution**: Pull the model first
```bash
ollama pull llama2
```

### "Out of memory"
**Solution**: 
- Enable CPU mode
- Use smaller models
- Reduce GPU layers

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ollama](https://ollama.ai/) for local LLM inference
- [Streamlit](https://streamlit.io/) for the web framework
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Alexey Grigorev](https://github.com/alexeygrigorev) for the sample documents

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ using Ollama & Streamlit**
