import os
import sys

# Configuração de prioridade de path para isolamento no Pop!_OS
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'venv_rag/lib/python3.10/site-packages')))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from dotenv import load_dotenv
from engine.rag_chain import RAGManager
from engine.ingestion import DocumentProcessor

# Carrega variáveis de ambiente (útil para chaves de API se decidir usar OpenAI no futuro)
load_dotenv()

# Configuração da página do Streamlit
st.set_page_config(page_title="Expert RAG Showroom", layout="wide", page_icon="🧠")

st.title("🧠 Local Mistral: Expert RAG Agent")
st.markdown("---")

# Sidebar para Configurações Técnicas e Ingestão
with st.sidebar:
    st.header("⚙️ Engineering Settings")
    chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
    chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
    top_k = st.number_input("Top-K Retrieval", 1, 10, 5)
    
    st.markdown("---")
    st.header("📂 Document Ingestion")
    uploaded_file = st.file_uploader("Upload a Technical PDF", type="pdf")
    
    if uploaded_file:
        if st.button("Index Document"):
            with st.status("🚀 Processing Document...", expanded=True) as status:
                # Salva o arquivo temporariamente para processamento
                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Instancia o processador (Configurado para ChromaDB local)
                processor = DocumentProcessor() 
                
                st.write("🔍 Splitting text into chunks...")
                chunks = processor.process_pdf("temp.pdf", chunk_size, chunk_overlap)
                
                st.write(f"📦 Generated {len(chunks)} chunks. Indexing in local Vector Store...")
                
                # CHAVE DA CORREÇÃO: Chamando o método save_to_chroma do seu ingestion.py
                processor.save_to_chroma(chunks)
                
                status.update(label="✅ Indexing Complete!", state="complete", expanded=False)
                st.success("The agent is ready to answer questions based on this PDF.")

# Interface de Chat e Inspeção (Debug)
tab1, tab2 = st.tabs(["💬 Expert Chat", "🔍 RAG X-Ray (Debug Mode)"])

# Inicialização do histórico de mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

with tab1:
    # Renderiza mensagens anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input do Usuário
    if prompt := st.chat_input("Ask about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Consulting local knowledge base..."):
                # Instancia o Gerenciador de RAG (Busca no ./db local)
                rag = RAGManager()
                response_data = rag.get_response(prompt, k=top_k)
                
                full_response = response_data["result"]
                st.markdown(full_response)
                
                # Exibição de metadados das fontes para transparência
                with st.expander("📑 View Sources"):
                    for doc in response_data["source_documents"]:
                        st.info(f"Source: {doc.metadata['source']} | Chunk ID: {doc.metadata.get('chunk_id', 'N/A')}")
                
                # Salva no estado da sessão para visualização no Tab de Debug
                st.session_state.last_sources = response_data["source_documents"]
                st.session_state.messages.append({"role": "assistant", "content": full_response})

with tab2:
    st.header("Vector Store Retrieval Inspection")
    if "last_sources" in st.session_state:
        st.write(f"Showing the top {len(st.session_state.last_sources)} retrieved chunks for the last query:")
        for i, doc in enumerate(st.session_state.last_sources):
            with st.container():
                st.subheader(f"Rank #{i+1}")
                st.write(f"**Content Snippet:**")
                st.text(doc.page_content)
                st.json(doc.metadata)
                st.markdown("---")
    else:
        st.info("Ask a question in the chat to inspect the retrieved context here.")
