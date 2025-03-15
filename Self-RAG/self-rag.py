import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer

# 質問が書かれたテキストファイルのパス
file_path = "../question.txt"
# RAG ありの場合の出力ファイルのパス
output_rag_path = "../output/self-rag.txt"

# Load the Elyza model
model_name = "elyza/ELYZA-japanese-Llama-2-13b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

loader = PyPDFDirectoryLoader("../data/")
document = loader.load()

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=20,
)

splitted_texts = text_splitter.split_documents(document)

# Add to vectorDB
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
db = FAISS.from_documents(splitted_texts, embeddings)

# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Chain
def generate_answer(context, question):
    input_text = f"Context: {context}\n\nQuestion: {question}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=1024)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Run
i = 1
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        question = line.strip()
        docs = db.similarity_search(question, k=3)
        context = format_docs(docs)


        with open(output_rag_path, "a", encoding="utf-8") as f:
            f.write("\n" + str(i) + "\n")
        
        answer = generate_answer(context, question)

        with open(output_rag_path, "a", encoding="utf-8") as f:
            f.write(answer + "\n")
        
        # break
        i += 1
        if(i%10 == 0):
            print(i)