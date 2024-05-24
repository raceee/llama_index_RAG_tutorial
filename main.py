from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
import psycopg2
from sqlalchemy import make_url
from llama_index.vector_stores.postgres import PGVectorStore
from pathlib import Path
from llama_index.readers.file import PyMuPDFReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import VectorStoreQuery
from retriever_class import VectorDBRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
model_url = "https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF/resolve/main/llama-2-13b-chat.Q4_0.gguf"

llm = LlamaCPP(
    model_url=model_url,
    model_path=None,
    temperature=0.1,
    max_new_tokens=256,
    context_window=3900,
    generate_kwargs={},
    model_kwargs={"n_gpu_layers": 1},
    verbose=True,
)


db_name = "vector_db"
host = "localhost"
password = "newuser"
port = "5432"
user = "newuser"
# conn = psycopg2.connect(connection_string)
conn = psycopg2.connect(
    dbname="postgres",
    host=host,
    password=password,
    port=port,
    user=user,
)
conn.autocommit = True

with conn.cursor() as c:
    c.execute(f"SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = '{db_name}' AND pid <> pg_backend_pid();")
    c.execute(f"DROP DATABASE IF EXISTS {db_name}")
    c.execute(f"CREATE DATABASE {db_name}")


vector_store = PGVectorStore.from_params(
    database=db_name,
    host=host,
    password=password,
    port=port,
    user=user,
    table_name="dei",
    embed_dim=384,  # openai embedding dimension
)

loader = PyMuPDFReader()
documents = loader.load(Path("./data/test.pdf"))

# use a text splitter to split the text into chunks

text_parser = SentenceSplitter(
    chunk_size=1024,
)

text_chunks = []

doc_idxs = []
for i, doc in enumerate(documents):
    cur_text_chunks = text_parser.split_text(doc.text)
    text_chunks.extend(cur_text_chunks)
    doc_idxs.extend([i] * len(cur_text_chunks))

# make 'nodes' out of chunks
print("doc_idxs checkpoint: ", doc_idxs)
nodes = []
for idx, text_chunk in enumerate(text_chunks):
    node = TextNode(
        text=text_chunk,
    )
    src_doc = documents[doc_idxs[idx]]
    node.metadata = src_doc.metadata
    nodes.append(node)

for node in nodes:
    node_embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
    node.embedding = node_embedding

vector_store.add(nodes)
# query_str = "Give me a summary of the diversity equity and inclusion policy."

# query_embedding = embed_model.get_query_embedding(query_str)

# # construct vector store query
# from llama_index.core.vector_stores import VectorStoreQuery

# query_mode = "default"
# # query_mode = "sparse"
# # query_mode = "hybrid"

# vector_store_query = VectorStoreQuery(query_embedding=query_embedding, similarity_top_k=2, mode=query_mode)

# # returns a VectorStoreQueryResult
# query_result = vector_store.query(vector_store_query)
# print(query_result.nodes[0].get_content())

# from llama_index.core.schema import NodeWithScore
# from typing import Optional

# nodes_with_scores = []
# for index, node in enumerate(query_result.nodes):
#     score: Optional[float] = None
#     if query_result.similarities is not None:
#         score = query_result.similarities[index]
#     nodes_with_scores.append(NodeWithScore(node=node, score=score))


if __name__ == "__main__":
    query_str = "Give me a summary of the diversity equity and inclusion policy."
    retriever = VectorDBRetriever(vector_store=vector_store, embed_model=llm, query_mode="default", similarity_top_k=2)
    # response = retriever._retrieve(query_str)
    query_engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
    response = query_engine.query(query_str)
    print(str(response))
    print("done!")
