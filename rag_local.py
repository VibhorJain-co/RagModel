import os
import argparse
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import multiprocessing

# === CONFIGURATION ===
MODEL_NAME = "llama3"  # Options: llama2 or llama3
MODEL_PATHS = {
    "llama2": "models/llama-2-7b-chat.Q4_K_M.gguf",
    "llama3": "models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
}
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA_DIR = "data"
EMBEDDINGS_DIR = "embeddings"

# === DOCUMENT INGESTION ===
def load_documents(data_dir):
    documents = []
    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, file))
            documents.extend(loader.load())
    return documents
 
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def create_or_load_vectorstore(split_docs, create_new=True):
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    if create_new:
        db = FAISS.from_documents(split_docs, embeddings)
        db.save_local(EMBEDDINGS_DIR)
        print(" ------- Vector store created and saved. -------")
        return db
    elif os.path.exists(os.path.join(EMBEDDINGS_DIR, "index.faiss")):
        return FAISS.load_local(EMBEDDINGS_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        raise ValueError("Vector store not found. Please create a new one or set create_new=True.")


# === LLaMA MODEL LOADING ===
# === LLaMA MODEL LOADING ===
def load_llm(model_path, model_name):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    # Choose stop tokens based on model
    if model_name == "llama3":
        stop_tokens = ["<|endoftext|>"]
    else:  # llama2
        stop_tokens = ["</s>"]

    return LlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        temperature=0.1,
        n_threads=multiprocessing.cpu_count(),
        n_gpu_layers=0,
        stop=stop_tokens,
        verbose=False
    )


# === MAIN RAG PIPELINE ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["llama2", "llama3"], default=MODEL_NAME, help="Choose LLaMA version")
    parser.add_argument("--rebuild-embeddings", action="store_true", help="Rebuild the vector store from scratch")
    args = parser.parse_args()

    print("🔎 Loading and processing documents...")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("❌ No documents found in data directory.")
        return

    chunks = split_documents(docs)
    vector_db = create_or_load_vectorstore(chunks, create_new=args.rebuild_embeddings)

    print(f"🤖 Loading {args.model} model...")
    llm = load_llm(MODEL_PATHS[args.model], args.model)
    if args.model == "llama3":
        prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
                    You are a helpful assistant. Use the following context to answer the question.
                    If the context does not provide enough information for you to generate an answer, say "I don't know."

                    Context:
                    {context}

                    Question:
                    {question}

                    Answer:"""
                )
    else:  # llama2 prompt style
        prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
                You are a helpful assistant. Use the following context to answer the question.
                If the context does not provide enough information for you to generate an answer, say "I don't know."

                Context:
                {context}

                Question:
                {question}

                Answer:"""
            )


    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

    print("\n✅ Ready! Type your questions (type 'exit' to quit):")
    while True:
        try:
            query = input("\n> ")
            if query.lower() in ["exit", "quit"]:
                break
            response = qa_chain({"query": query})
            print("\n🧠 Answer:", response["result"])
            print("\n🔍 Source Documents:")
            for doc in response["source_documents"]:
                print(f" {doc.metadata}")
        except KeyboardInterrupt:
            print("\n👋 Exiting.")
            break

    # chunks = split_documents(docs)
    # for i in range(10):
    #     print("Sample chunk text:\n", chunks[i].page_content[:500])
    #     print("-"*100)
    #     print("-"*100)
    #     print("-"*100)


if __name__ == "__main__":
    main()


# ALTERNATIVE PROMPT TEMPLATE
"""
You are a helpful assistant. Use the following context to answer the question.
If the context does not provide enough information for you to generate an answer, say "I don't know."

Context:
{context}

Question:
{question}

Answer:"""