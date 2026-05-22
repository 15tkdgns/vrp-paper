import zipfile
import xml.etree.ElementTree as ET
import os

def convert_docx_to_md(docx_path, md_path):
    if not os.path.exists(docx_path):
        print(f"Error: {docx_path} not found.")
        return

    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    try:
        with zipfile.ZipFile(docx_path, 'r') as z:
            with z.open('word/document.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                
                paragraphs = []
                for p in root.findall('.//w:p', ns):
                    texts = p.findall('.//w:r/w:t', ns)
                    paragraph_text = "".join(t.text for t in texts if t.text)
                    if paragraph_text.strip():
                        paragraphs.append(paragraph_text)
                
                with open(md_path, 'w', encoding='utf-8') as md_file:
                    for p in paragraphs:
                        md_file.write(p + "\n\n")
                
                print(f"Successfully converted {docx_path} to {md_path}")
                
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = r"v0_Weighted Ensemble을 활용한다중 예측 기간 Cross-Asset 분산 리스크 프리미엄 예측_피드백_2026.03.12..docx"
    output_file = "피드백_2026.03.12.md"
    convert_docx_to_md(input_file, output_file)
