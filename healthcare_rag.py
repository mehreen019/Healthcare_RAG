from datasets import load_dataset
from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss
import os
import torch
import re

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device="cuda" if torch.cuda.is_available() else "cpu",
    embed_batch_size=128,
    normalize=True
)

Settings.llm = None
Settings.chunk_size=512
Settings.chunk_overlap=64

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

model_path = "/content/drive/MyDrive/phi3-pubmedqa-finetuned"

model = AutoModelForCausalLM.from_pretrained(
    "./phi3-pubmedqa-finetuned",
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained("./phi3-pubmedqa-finetuned")

pubmed_dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:1000]")
medical_contexts = [
    " ".join(ex["context"]["contexts"])
    for ex in pubmed_dataset
]

documents = [Document(text=ctx) for ctx in medical_contexts]

persist_dir = "./vector_store"
dimension = 384

if os.path.exists(persist_dir):
    vector_store = FaissVectorStore.from_persist_dir(persist_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store, persist_dir=persist_dir
    )
    index = VectorStoreIndex.load(storage_context=storage_context)
    print("Loaded existing FAISS index.")
else:

    faiss_index = faiss.IndexHNSWFlat(dimension, 32)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        show_progress=True
    )

    storage_context.persist(persist_dir=persist_dir)
    print("Created new FAISS index.")

query_engine = index.as_query_engine(
    similarity_top_k=3,
    node_postprocessors=[
        SimilarityPostprocessor(similarity_cutoff=0.65)
    ]
)

def extract_answer(full_response):
    match = re.search(r'ANSWER:(.*?)(?:QUESTION:|$)', full_response, re.DOTALL)
    return match.group(1).strip() if match else full_response.strip()

def medical_rag(question):
    nodes = query_engine.retrieve(question)
    contexts = [n.text for n in nodes]

    prompt = f"""MEDICAL CONTEXT: {' '.join(contexts)}
**INSTRUCTIONS**
1. Answer ONLY the question asked
2. Use only information from the context
3. If unsure, ONLY say "I cannot determine from available evidence"
4. Answer in 2-3 simple sentences

QUESTION: {question}
ANSWER:"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True
    )

    full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    answer = full_response.split("ANSWER:")[-1]
    answer = answer.split("QUESTION:")[0].strip()
    answer = answer.split("EOS")[0].strip()
    answer = answer.split("\n")[0].strip()

    return extract_answer(tokenizer.decode(outputs[0], skip_special_tokens=True))