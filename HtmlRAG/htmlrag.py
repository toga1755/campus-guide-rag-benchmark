# これを実行する前に"HtmlRAG/pdfocr.py"を実行してPDFを画像に変換、「Yomitoku」パッケージをターミナルで実行してHTMLに変換しておくこと

#################################################
# 日本語用_V4
#################################################
import os
# torch                     2.5.0
import torch
# bs4                       0.0.2
from bs4 import BeautifulSoup
# sentence-transformers     3.3.1
from sentence_transformers import SentenceTransformer, util
# transformers              4.46.3
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# 質問が書かれたテキストファイルのパス
file_path = "../question.txt"
# RAG ありの場合の出力ファイルのパス
output_rag_path = "../output/htmlrag.txt"

def clean_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    for script_or_style in soup(["script", "style", "meta", "link"]):
        script_or_style.decompose()
    for tag in soup.find_all():
        if not tag.text.strip():
            tag.decompose()
        else:
            tag.attrs = {}
    return str(soup)

def build_block_tree(html_content, max_words=50):
    soup = BeautifulSoup(html_content, 'html.parser')
    blocks = []

    def traverse(tag):
        text = tag.get_text(separator=" ", strip=True)
        if len(text.split()) > max_words:
            for child in tag.children:
                if child.name:
                    traverse(child)
        else:
            blocks.append((tag.name, text.strip()))

    traverse(soup.body)
    return blocks

emb_model = SentenceTransformer('intfloat/multilingual-e5-large')

def prune_blocks(blocks, query, max_length=300):
    query_embedding = emb_model.encode(query, convert_to_tensor=True)
    pruned_blocks = []

    for tag, text in blocks:
        block_embedding = emb_model.encode(text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(query_embedding, block_embedding).item()
        if score > 0.5:
            pruned_blocks.append((tag, text))

    total_length = 0
    result = []
    for tag, text in pruned_blocks:
        if total_length + len(text) <= max_length:
            result.append((tag, text))
            total_length += len(text)
        else:
            break
    return result

# 環境変数の設定

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

llm_model_name = "elyza/ELYZA-japanese-Llama-2-13b-instruct"
llm_model = AutoModelForCausalLM.from_pretrained(llm_model_name)
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)

def generate_answer(pruned_blocks, query):
    context = " ".join([text for _, text in pruned_blocks])
    llm = pipeline("text-generation", model=llm_model, tokenizer=tokenizer, device=-1)  # GPUを使用するためにdevice=0を設定
    prompt = f"質問: {query}\n\n文脈: {context}\n\n答え:"
    return llm(prompt, max_new_tokens=100)[0]['generated_text']

# HTMLファイルが格納されているディレクトリ
html_dir = "campusguide_html"

# ディレクトリ内のすべてのHTMLファイルを取得
html_files = [os.path.join(html_dir, file) for file in os.listdir(html_dir) if file.endswith(".html")]

# with open("combined_html_files.html", "w", encoding="utf-8") as output_file:
#     for html_file in html_files:
#         # HTMLファイルを読み込む
#         with open(html_file, "r", encoding="utf-8") as f:
#             html_content = f.read()
#             # HTMLファイルの内容を出力ファイルに書き込む
#             output_file.write(html_content)
#             output_file.write("\n")  # ファイル間に改行を追加

# HTMLファイルを読み込む
with open("combined_html_files.html", "r", encoding="utf-8") as f:
    html = f.read()

simplified_html = clean_html(html)
# # simplified_htmlをテキストファイルに保存
# with open("simplified_miyazaki_university.html", "w", encoding="utf-8") as f:
#     f.write(simplified_html)

blocks = build_block_tree(simplified_html)
# blocksをテキストファイルに保存
# with open("blocks_miyazaki_university.txt", "w", encoding="utf-8") as f:
#     for tag, text in blocks:
#         f.write(f"{tag}: {text}\n")

file_path = "/home/stm7101@csvm.local/HtmlRAG/qestion.txt"
output_rag_path = "/home/stm7101@csvm.local/HtmlRAG/htmlrag-elyza-ELYZA-japanese-Llama-2-13b-instruct-v2.txt"

i = 1
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        question = line.strip()
        # 質問に対して、データベース中の類似度上位を抽出。
        pruned_blocks = prune_blocks(blocks, question)

        ###################################################
        # 生成
        with open(output_rag_path, "a", encoding="utf-8") as f:
            f.write("\n" + str(i) + "\n")
        
        # RAG ありの場合
        answer = generate_answer(pruned_blocks, question)

        # answer.txt に出力
        with open(output_rag_path, "a", encoding="utf-8") as f:
            f.write(answer + "\n")
        
        # break
        i += 1
        if(i%10 == 0):
            print(i)


# メモリの解放
del emb_model, llm_model, simplified_html, blocks, pruned_blocks, answer
torch.cuda.empty_cache()