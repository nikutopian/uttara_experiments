import argparse
import os
import zipfile
from typing import List, Dict
import ollama
from file_parser import FileParser

# Function to process input path (zip file or folder) and return a mapping of file paths to their content
def process_input(input_path: str) -> Dict[str, str]:
    file_parser = FileParser()
    file_mapping = {}

    folder_path = input_path

    # Check if the input is a zip file
    if os.path.isfile(input_path) and input_path.endswith('.zip'):
        folder_path = input_path + "_extracted"
        os.makedirs(folder_path, exist_ok=True)
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                zip_ref.extract(file_name, folder_path)
    
    # Process the folder to parse files
    if os.path.isdir(folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    content = file_parser.parse_file(file_path)
                    file_mapping[file_path] = content
                except Exception:
                    pass
    else:
        raise ValueError("Input must be a zip file or a folder")

    return file_mapping

# Function to analyze the content of files against the expense policy
def analyze_content(content_mapping: Dict[str, str], expense_policy: str) -> str:
    sys_prompt = "You are a finance controller in a company who is tasked with making sure all expenses submitted by employees conform to the company's expense policy."
    client = ollama.Client()

    full_output = ''

    for file_path, content in content_mapping.items():
        # Prepare output for each file
        output += "-"*100
        output = f"File: {file_path}\n"
        output += "-"*100
        output += f"Content: {content}\n"
        output += "-"*100
        output += "\n\n"

        full_output += output
        print(output)

        # Prepare user prompt for the LLM
        user_prompt = f"Given the company's expense policy : {expense_policy}. Can you now validate if the expense document conforms with the expense policy. Document: {content}"
        response = client.chat(model="mistral", messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ])
        assistant_output = f"Assistant: {response['message']['content']}\n"
        full_output += assistant_output
        print(assistant_output)

    return full_output

# Main function to handle argument parsing and orchestrate the processing and analysis
def main():
    parser = argparse.ArgumentParser(description="Analyze files using LLM")
    parser.add_argument("-i", "--input", required=True, help="Path to zip file or folder")
    parser.add_argument("-p", "--policy", required=True, help="Path to expense policy document")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    args = parser.parse_args()

    try:
        # Process input files
        content_list = process_input(args.input)
        file_parser = FileParser()
        # Parse the expense policy document
        expense_policy = file_parser.parse_file(args.policy)
        # Analyze the content of the files
        analysis_result = analyze_content(content_list, expense_policy)
        # Write the analysis result to the output file
        with open(args.output, "w") as f:
            f.write(analysis_result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()