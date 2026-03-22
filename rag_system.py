from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

llm = Ollama(
    model="llama3.2:3b",
    request_timeout=120.0,
    # Manually set the context window to limit memory usage
    context_window=8000,
)

# resp = llm.complete("Who is APJ Abdul Kalam?")
# print(resp)

Settings.llm = Ollama(model="llama3.2:3b", request_timeout=120.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

#Load my documents

documents = SimpleDirectoryReader(input_dir="/Users/mzaheen/Dev/RAG/Docs").load_data()
print(f"Loaded {len(documents)} docs")
#DO indexing on the data
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

while True:
    query = input("Enter here:")
    response = query_engine.query(query)
    print(response)
