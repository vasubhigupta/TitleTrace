from pathlib import Path
import json
from predict_from_pdf import predict_from_pdf
from csv_to_json import extract_headings_from_df  

def process_pdfs():
    input_dir = Path(".\input")
    output_dir = Path(".\output")
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_dir.glob("*.pdf"))

    for pdf_file in pdf_files:
        try:
            print(f"\nProcessing: {pdf_file.name}")

            df = predict_from_pdf(str(pdf_file))

            json_data = extract_headings_from_df(df)

            json_output_path = output_dir / f"{pdf_file.stem}.json"
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4, ensure_ascii=False)

            print(f" Done: {pdf_file.name} -> {json_output_path.name} ({len(json_data)} entries)")

        except Exception as e:
            print(f"Failed to process {pdf_file.name}: {e}")

if __name__ == "__main__":
    print(" Starting processing pipeline")
    process_pdfs()
    print(" All PDFs processed")
