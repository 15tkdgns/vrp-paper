import PyPDF2
import sys

try:
    reader = PyPDF2.PdfReader('v23.pdf')
    annotations = []
    for i, page in enumerate(reader.pages):
        if '/Annots' in page:
            for annot in page['/Annots']:
                obj = annot.get_object()
                if '/Contents' in obj:
                    annotations.append(f"Page {i+1}: {obj['/Contents']}")
    
    if annotations:
        print(f"Found {len(annotations)} annotations:")
        for ann in annotations:
            print(ann)
    else:
        print("No annotations found.")
except Exception as e:
    print(f"Error: {e}")
