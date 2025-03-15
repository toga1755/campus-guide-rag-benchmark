# torch                     2.5.0
import torch
# langchain                 0.2.1
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
# langchain-community       0.2.1
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# langchain-huggingface     0.0.1
from langchain_huggingface.llms import HuggingFacePipeline
# transformers              4.42.3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 質問が書かれたテキストファイルのパス
file_path = "../question.txt"
# RAG ありの場合の出力ファイルのパス
output_rag_path = "../output/rag.txt"
# RAG なしの場合の出力ファイルのパス
output_norag_path = "../output/no-rag.txt"


###################################################
# 前半: RAGによる文書生成
###################################################
# ベクトルデータベースの準備
loader = PyPDFDirectoryLoader("../data/")
document = loader.load()

# 全文章を決まった長さの文章（チャンク）に分割して、文章データベースを作成
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=20,
)

splitted_texts = text_splitter.split_documents(document)

# ベクトルデータベースの作成
# 文章からベクトルに変換するためのモデルを用意
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
# 文章データベースからベクトルデータベースを作成。チャンク単位で文章からベクトルに変換。
db = FAISS.from_documents(splitted_texts, embeddings)

###################################################
# RAG の準備

# 生成モデルの準備
device = "cuda:0"
model_name = "elyza/ELYZA-japanese-Llama-2-13b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model = model.to(device)

# RAG のためのLangChainのインタフェース準備
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    top_k=20,
    temperature=0.7,
    device=device,
)
llm = HuggingFacePipeline(pipeline=pipe)

# プロンプトの準備（ELYZA 用）
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = "参考情報を元に、ユーザーからの質問に簡潔に正確に答えてください。"
text = "{context}\nユーザからの質問は次のとおりです。{question}"
template = "{bos_token}{b_inst} {system}{prompt} {e_inst} ".format(
    bos_token=tokenizer.bos_token,
    b_inst=B_INST,
    system=f"{B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}",
    prompt=text,
    e_inst=E_INST,
)
rag_prompt_custom = PromptTemplate(
    template=template, input_variables=["context", "question"]
)

# チェーンの準備
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt_custom)

###################################################
# ベクトル検索

i = 1
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        question = line.strip()
        # 質問に対して、データベース中の類似度上位3件を抽出。質問の文章はこの関数でベクトル化され利用される
        docs = db.similarity_search(question, k=3)

        ###################################################
        # 生成

        with open(output_rag_path, "a", encoding="utf-8") as f:
            f.write("\n" + str(i) + "\n")

        # RAG ありの場合
        inputs = {"input_documents": docs, "question": question}
        output_rag = chain.run(inputs)
        # 不要なタグを削除
        output_rag = output_rag.replace(B_INST, "").replace(E_INST, "").replace(B_SYS, "").replace(E_SYS, "").strip()
        output_rag = output_rag.split("ユーザからの質問は次のとおりです。")[1].strip()
        # answer.txt に出力
        with open(output_rag_path, "a", encoding="utf-8") as f:
            f.write(output_rag + "\n")

        with open(output_norag_path, "a", encoding="utf-8") as f:
            f.write("\n" + str(i) + "\n")
        
        # RAG なしの場合
        inputs = template.format(context="", question=question)
        output = llm(inputs)
        # 不要なタグを削除
        output = output.replace(B_INST, "").replace(E_INST, "").replace(B_SYS, "").replace(E_SYS, "").strip()
        output = output.split("ユーザからの質問は次のとおりです。")[1].strip()
        # answer.txt に出力
        with open(output_norag_path, "a", encoding="utf-8") as f:
            f.write(output + "\n")
        
        # break
        i += 1
        if(i%10 == 0):
            print(i)


###################################################
# メモリの解放

del model, tokenizer, pipe, llm, chain
torch.cuda.empty_cache()
