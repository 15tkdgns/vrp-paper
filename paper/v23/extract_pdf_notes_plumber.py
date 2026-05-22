import pdfplumber
import json
import sys

try:
    annotations_data = []
    with pdfplumber.open("v23.pdf") as pdf:
        for i, page in enumerate(pdf.pages):
            if page.annots:
                for annot in page.annots:
                    # annots is a dict representing the PDF annotation dictionary
                    annot_type = annot.get("Subtype")
                    if isinstance(annot_type, bytes):
                        annot_type = annot_type.decode("utf-8", errors="ignore")
                    elif hasattr(annot_type, "name"):
                         annot_type = annot_type.name
                    
                    contents = annot.get("Contents")
                    if isinstance(contents, bytes):
                        try:
                            # PDF strings might be UTF-16 BE or PDFDocEncoding
                            if contents.startswith(b'\xfe\xff'):
                                contents = contents.decode('utf-16-be', errors='replace')
                            else:
                                contents = contents.decode('utf-8', errors='replace')
                        except:
                            contents = repr(contents)
                    
                    if contents:
                         annotations_data.append({
                             "page": i + 1,
                             "type": str(annot_type),
                             "contents": contents
                         })

    if annotations_data:
        print(f"Found {len(annotations_data)} annotations:")
        for ann in annotations_data:
            print(f"Page {ann['page']} [{ann['type']}]: {ann['contents']}")
        with open("pdf_annotations_plumber.txt", "w", encoding="utf-8") as f:
            for ann in annotations_data:
                f.write(f"Page {ann['page']} [{ann['type']}]: {ann['contents']}\n")
    else:
        print("No annotations with contents found.")

except Exception as e:
    print(f"Error extracting annotations: {e}")
