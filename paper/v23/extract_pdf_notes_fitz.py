import fitz  # PyMuPDF
import sys

try:
    doc = fitz.open('v23.pdf')
    annotations = []
    
    for i in range(len(doc)):
        page = doc[i]
        for annot in page.annots():
            # Get annotation details
            info = annot.info
            text = info.get('content', '')
            if text:
                annotations.append(f"Page {i+1} [{annot.type[1]}]: {text}")
                
    if annotations:
        with open('pdf_annotations_fitz.txt', 'w', encoding='utf-8') as f:
            f.write(f"Found {len(annotations)} annotations:\n")
            for ann in annotations:
                f.write(ann + "\n")
        print(f"Extraction successful: {len(annotations)} annotations found.")
    else:
        with open('pdf_annotations_fitz.txt', 'w', encoding='utf-8') as f:
            f.write("No annotations found.\n")
        print("No annotations found.")
        
except Exception as e:
    print(f"Error: {e}")
