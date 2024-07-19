from llms.vision_llm import VisionLLM
from llms import model_types
import argparse
import os
from pathlib import Path
import pandas as pd
from PIL import Image
from tqdm import tqdm
import json
import re


def clean_json_string(json_str):
    # Remove special characters
    json_str = json_str.strip("```").strip("json").strip()

    # Remove escaped newlines and replace with actual newlines
    json_str = json_str.replace("\\n", "\n")

    # Remove any trailing commas in arrays or objects
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    return json_str


def parse_json_safely(json_str):
    try:
        # Clean the JSON string
        cleaned_json_str = clean_json_string(json_str)

        # Parse the cleaned JSON string
        data = json.loads(cleaned_json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return {}


def process_image(image_path, vision_lm):
    # Open the image file
    output_schema = {
        "company_name": "Name of the company issuing the invoice",
        "buyer_name": "Buyer Name",
        "invoice_date": "Invoice date",
        "invoice_number": "Invoice number",
        "gst_number": "Company's GST [or similar number]. Leave blank if there is no equivalent number",
        "product_description": "Product/Service purchased",
        "total_amount": "Total Amount",
        "other": "other details from the image in a json format",
    }

    user_prompt = f"Extract text data from this image in a json format. Use the schema below to define the output format \n{output_schema}"

    response = {}
    with Image.open(image_path) as img:
        response = vision_lm.chat(
            user_prompt,
            [img],
        )
        if isinstance(response, str):
            response = parse_json_safely(response)

    return response


def main():
    parser = argparse.ArgumentParser(
        description="Process images in a folder and save results to a file."
    )
    parser.add_argument(
        "--input_folder",
        help="Path to the folder containing image files",
        type=str,
    )
    parser.add_argument(
        "--model_type",
        help="Type of model service to use",
        type=str,
        choices=[
            model_types.ModelServiceType.OPENAI.value,
            model_types.ModelServiceType.CLAUDE.value,
        ],
        default=model_types.ModelServiceType.OPENAI.value,
    )
    parser.add_argument(
        "--model_name",
        help="Name of model to use",
        type=str,
        choices=[
            model_types.OpenAIModelType.GPT4_O.value,
            model_types.OpenAIModelType.GPT4_O_MINI.value,
            model_types.OpenAIModelType.GPT4_TURBO.value,
            model_types.ClaudeModelType.CLAUDE35_SONNET.value,
            model_types.ClaudeModelType.CLAUDE3_OPUS.value,
        ],
        default=model_types.OpenAIModelType.GPT4_O.value,
    )
    parser.add_argument(
        "--output_file",
        help="Path to the output file (CSV/TSV/JSON/XLSX)",
        type=str,
    )
    args = parser.parse_args()

    # Validate input folder
    if not os.path.isdir(args.input_folder):
        print(f"Error: {args.input_folder} is not a valid directory")
        return

    # Validate output file extension
    output_ext = Path(args.output_file).suffix.lower()
    if output_ext not in [".csv", ".tsv", ".json", ".xlsx"]:
        print(f"Error: Output file must have .csv, .tsv, .json, or .xlsx extension")
        return

    vlm = VisionLLM(model_service_type=args.model_type)
    # Get list of image files
    image_files = [
        f
        for f in os.listdir(args.input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp"))
    ]

    # Process images with progress bar
    results = []
    with tqdm(total=len(image_files), unit="file") as pbar:
        for filename in image_files:
            pbar.set_description(f"Processing {filename}")
            image_path = os.path.join(args.input_folder, filename)
            result = process_image(image_path, vlm)
            if result:
                result["filename"] = filename
                results.append(result)
            pbar.update(1)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Save to file
    if output_ext == ".csv":
        df.to_csv(args.output_file, index=False)
    elif output_ext == ".tsv":
        df.to_csv(args.output_file, sep="\t", index=False)
    elif output_ext == ".json":
        df.to_json(args.output_file, orient="records")
    elif output_ext == ".xlsx":
        df.to_excel(args.output_file, index=False)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
