import zipfile
import xml.etree.ElementTree as ET

docx_path = 'v23.docx'
out_path = 'docx_output_utf8.txt'

with open(out_path, 'w', encoding='utf-8') as out_f:
    with zipfile.ZipFile(docx_path, 'r') as z:
        with z.open('word/document.xml') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
            paragraphs = root.findall('.//w:p', ns)
            out_f.write(f"Total paragraphs: {len(paragraphs)}\n")
            out_f.write("="*80 + "\n")
            for i, p in enumerate(paragraphs):
                texts = p.findall('.//w:t', ns)
                line = ''.join(t.text for t in texts if t.text)
                if line.strip():
                    out_f.write(f"{i}: {line}\n")
