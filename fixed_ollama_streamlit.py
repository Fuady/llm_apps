import streamlit as st
import requests
import json
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
OLLAMA_API_URL = "http://localhost:11434"
DOCUMENTS_URL = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = OLLAMA_API_URL):
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
        self.chat_url = f"{base_url}/api/chat"
        self.tags_url = f"{base_url}/api/tags"
    
    def chat(self, model: str, messages: List[Dict], stream: bool = False, num_gpu: int = -1):
        """Send chat request to Ollama using generate endpoint"""
        try:
            # Convert messages to a single prompt for generate endpoint
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": stream,
                "options": {}
            }
            
            # Add GPU layers control if specified
            if num_gpu >= 0:
                payload["options"]["num_gpu"] = num_gpu
            
            response = requests.post(self.generate_url, json=payload, timeout=120)
            
            # Check for specific error messages
            if response.status_code == 500:
                try:
                    error_detail = response.json()
                    error_msg = error_detail.get('error', 'Unknown error')
                    st.error(f"Ollama Error: {error_msg}")
                    
                    # Check for memory errors
                    if "memory" in error_msg.lower() or "vram" in error_msg.lower():
                        st.warning("🧠 Not enough GPU memory!")
                        st.info("""
**Solutions:**
1. Try a smaller model (phi, tinyllama)
2. Use CPU mode (set GPU Layers to 0)
3. Reduce GPU layers in sidebar
4. Close other GPU applications
                        """)
                except:
                    st.error(f"Model '{model}' may not be available. Try pulling it first: `ollama pull {model}`")
                return None
            
            response.raise_for_status()
            
            result = response.json()
            
            # Format response to match expected structure
            return {
                "message": {
                    "content": result.get("response", "")
                }
            }
            
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to Ollama: {e}")
            st.info(f"💡 Make sure the model '{model}' is installed: `ollama pull {model}`")
            return None
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert messages list to a single prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
            elif role == "system":
                prompt_parts.append(f"System: {content}")
        
        return "\n\n".join(prompt_parts)
    
    def list_models(self):
        """Get list of available models"""
        try:
            response = requests.get(self.tags_url, timeout=10)
            response.raise_for_status()
            return response.json().get('models', [])
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching models: {e}")
            return []

class DocumentRetriever:
    """Retrieves relevant documents using semantic search"""
    
    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.model = None
        
    def load_documents(self, url: str):
        """Load documents from URL"""
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            self.documents = response.json()
            return True
        except requests.exceptions.RequestException as e:
            st.error(f"Error loading documents: {e}")
            return False
    
    def initialize_embeddings(self):
        """Initialize the embedding model and create document embeddings"""
        if not self.documents:
            return False
        
        try:
            with st.spinner("Loading embedding model..."):
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            with st.spinner("Creating document embeddings..."):
                texts = [
                    f"{doc.get('question', '')} {doc.get('text', '')}" 
                    for doc in self.documents
                ]
                self.embeddings = self.model.encode(texts)
            
            return True
        except Exception as e:
            st.error(f"Error initializing embeddings: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for relevant documents"""
        if self.embeddings is None or self.model is None:
            return []
        
        try:
            query_embedding = self.model.encode([query])[0]
            
            # Calculate cosine similarity
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                doc = self.documents[idx].copy()
                doc['similarity'] = float(similarities[idx])
                results.append(doc)
            
            return results
        except Exception as e:
            st.error(f"Error searching documents: {e}")
            return []

def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'retriever' not in st.session_state:
        st.session_state.retriever = DocumentRetriever()
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'ollama_client' not in st.session_state:
        st.session_state.ollama_client = OllamaClient()

def create_rag_prompt(query: str, documents: List[Dict]) -> str:
    """Create a RAG prompt with context from documents"""
    context = "\n\n".join([
        f"Document {i+1}:\nQuestion: {doc.get('question', 'N/A')}\nAnswer: {doc.get('text', 'N/A')}"
        for i, doc in enumerate(documents)
    ])
    
    prompt = f"""You are a helpful assistant. Answer the user's question based on the following context documents.
If the answer is not in the context, say so and provide a general response.

Context:
{context}

User Question: {query}

Answer:"""
    return prompt

def test_ollama_connection():
    """Test if Ollama is accessible"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            return True, "Connected"
        else:
            return False, f"Status code: {response.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    st.set_page_config(
        page_title="Ollama Chatbot",
        page_icon="🤖",
        layout="wide"
    )
    
    initialize_session_state()
    
    st.title("🤖 Ollama Chatbot")
    st.markdown("Chat with AI using local Ollama models")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Connection status
        is_connected, status_msg = test_ollama_connection()
        if is_connected:
            st.success(f"✅ Ollama: {status_msg}")
        else:
            st.error(f"❌ Ollama: {status_msg}")
            st.info("Make sure Ollama is running: `ollama serve`")
        
        st.divider()
        
        # Model selection
        models = st.session_state.ollama_client.list_models()
        if models:
            model_names = [m['name'] for m in models]
            selected_model = st.selectbox(
                "Select Model",
                model_names,
                index=0 if model_names else None
            )
            
            # Show model info
            st.caption(f"📦 Available models: {len(model_names)}")
        else:
            st.warning("⚠️ No models found!")
            st.info("Install a model first:")
            st.code("ollama pull llama2", language="bash")
            st.code("ollama pull mistral", language="bash")
            st.code("ollama pull phi", language="bash")
            selected_model = st.text_input("Or enter model name manually:", "llama2")
        
        st.divider()
        
        # Memory management
        st.header("🧠 Memory Settings")
        
        use_cpu = st.checkbox(
            "Force CPU Mode (No GPU)",
            value=False,
            help="Run entirely on CPU - slower but uses no GPU memory"
        )
        
        if not use_cpu:
            num_gpu_layers = st.slider(
                "GPU Layers",
                min_value=0,
                max_value=50,
                value=-1,
                help="-1 = Auto (use all GPU), 0 = CPU only, 1-50 = Partial GPU offload"
            )
        else:
            num_gpu_layers = 0
        
        if num_gpu_layers == 0:
            st.info("⚠️ Running on CPU - This will be slower")
        elif num_gpu_layers > 0:
            st.info(f"🔀 Hybrid mode: {num_gpu_layers} layers on GPU, rest on CPU")
        
        st.caption("💡 Reduce GPU layers if you get memory errors")
        
        st.divider()
        
        # Mode selection
        st.header("📋 Chat Mode")
        chat_mode = st.radio(
            "Choose mode:",
            ["Free Chat", "Document-based (RAG)"],
            help="Free Chat: Direct conversation with the model\nDocument-based: Answers based on loaded documents"
        )
        
        # Document loading for RAG mode
        if chat_mode == "Document-based (RAG)":
            st.divider()
            st.subheader("📚 Document Settings")
            
            if not st.session_state.documents_loaded:
                if st.button("Load Documents", type="primary"):
                    with st.spinner("Loading documents..."):
                        if st.session_state.retriever.load_documents(DOCUMENTS_URL):
                            if st.session_state.retriever.initialize_embeddings():
                                st.session_state.documents_loaded = True
                                st.success(f"✅ Loaded {len(st.session_state.retriever.documents)} documents")
                                st.rerun()
            else:
                st.success(f"✅ {len(st.session_state.retriever.documents)} documents loaded")
                if st.button("Reload Documents"):
                    st.session_state.documents_loaded = False
                    st.session_state.retriever = DocumentRetriever()
                    st.rerun()
            
            num_docs = st.slider(
                "Number of context documents",
                min_value=1,
                max_value=5,
                value=3,
                help="How many relevant documents to use as context"
            )
        else:
            num_docs = 3
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        st.caption("Made with ❤️ using Ollama & Streamlit")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show retrieved documents if available
            if message["role"] == "assistant" and "documents" in message:
                with st.expander("📄 View Source Documents"):
                    for i, doc in enumerate(message["documents"]):
                        st.markdown(f"**Document {i+1}** (Similarity: {doc['similarity']:.3f})")
                        st.markdown(f"**Q:** {doc.get('question', 'N/A')}")
                        st.markdown(f"**A:** {doc.get('text', 'N/A')}")
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Handle RAG mode
                if chat_mode == "Document-based (RAG)":
                    if not st.session_state.documents_loaded:
                        st.error("Please load documents first using the sidebar.")
                        st.stop()
                    
                    # Retrieve relevant documents
                    relevant_docs = st.session_state.retriever.search(prompt, top_k=num_docs)
                    
                    if relevant_docs:
                        # Create RAG prompt
                        rag_prompt = create_rag_prompt(prompt, relevant_docs)
                        
                        # Get response from Ollama
                        messages = [{"role": "user", "content": rag_prompt}]
                        response = st.session_state.ollama_client.chat(
                            selected_model, 
                            messages, 
                            num_gpu=num_gpu_layers
                        )
                        
                        if response:
                            assistant_message = response['message']['content']
                            st.markdown(assistant_message)
                            
                            # Store message with documents
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": assistant_message,
                                "documents": relevant_docs
                            })
                        else:
                            st.error("Failed to get response from Ollama")
                    else:
                        st.error("No relevant documents found")
                
                # Handle Free Chat mode
                else:
                    # Prepare messages for context
                    messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                    ]

                    try:
                        # Get response from Ollama
                        response = st.session_state.ollama_client.chat(
                            model=selected_model,
                            messages=messages
                        )

                        if response and "message" in response:
                            assistant_message = response["message"]["content"]
                            st.markdown(assistant_message)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": assistant_message
                            })
                        else:
                            st.error("No valid response received from Ollama. Check model or connection.")

                    except Exception as e:
                        st.error(f"Error connecting to Ollama: {e}")


if __name__ == "__main__":
    main()