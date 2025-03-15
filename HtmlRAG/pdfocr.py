# これを実行したら、Yomitokuパッケージをターミナルで実行してHTMLに変換してください。
# yomitoku                  0.6.0

import os
# pdf2image                 1.17.0
from pdf2image import convert_from_path

# PDFファイルのパス
pdf_files = ["../data/20210427kyoutuu.pdf", "../data/20210427gen.pdf", "../data/20210427med.pdf", "../data/20210427edu.pdf", "../data/20210427eng.pdf", "../data/20210427agr.pdf", "../data/20210427reg.pdf"]
# 画像ファイルの出力ディレクトリ
output_dir = "campusguide_images"

def convert_pdf_to_image(pdf_path, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)

    images = convert_from_path(pdf_path)
    pdf_name = os.path.basename(pdf_path).replace(".pdf", "")
    for i, image in enumerate(images):
        output_path = f"{output_dir_path}/{pdf_name}_page_{i + 1}.png"
        image.save(output_path, "PNG")
        print(f"Saved: {output_path}")

for pdf in pdf_files:
    convert_pdf_to_image(pdf, output_dir)