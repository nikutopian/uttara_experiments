import argparse
import os
import zipfile
from typing import List, Dict
import ollama
from file_parser import FileParser


def process_input(input_path: str) -> Dict[str, str]:
    file_parser = FileParser()
    file_mapping = {}

    folder_path = input_path

    if os.path.isfile(input_path) and input_path.endswith('.zip'):
        folder_path = input_path + "_extracted"
        os.makedirs(folder_path, exist_ok=True)
        with zipfile.ZipFile(input_path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                zip_ref.extract(file_name, folder_path)
                # with zip_ref.open(file_name) as file:
                #     content = file_parser.parse_file(file.read())
                #     content_list.append(content)
    

    if os.path.isdir(folder_path):
        for root, _, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    content = file_parser.parse_file(file_path)
                    file_mapping[file_path] = content
                except Exception:
                    pass
                # content_list.append(content)
    else:
        raise ValueError("Input must be a zip file or a folder")

    return file_mapping

def analyze_content(content_mapping: Dict[str, str]) -> str:
    # combined_content = "\n\n".join(content_mapping)
    combined_content = str(content_mapping)
    sys_prompt = "You are a finance controller in a company who is tasked with making sure all expenses submitted by employees conform to the company's expense policy."
    expense_policy_file_name = 'Expense_policy.docx'
    expense_policy = ''
    client = ollama.Client()

    for file_path, content in content_mapping.items():
        if file_path.endswith(expense_policy_file_name):
            expense_policy = content

    for file_path, content in content_mapping.items():
        if not file_path.endswith(expense_policy_file_name):
            print()
            print("-"*100)
            print(file_path)
            print("-"*100)
            user_prompt = f"Given the company's expense policy : {expense_policy}. Can you now validate if the expense document conforms with the expense policy. Document: {content}"
            response = client.chat(model="mistral", messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ])
            print(response["message"]["content"])
            print()


    return response['message']["content"]

def main():
    parser = argparse.ArgumentParser(description="Analyze files using LLM")
    parser.add_argument("-i", "--input", required=True, help="Path to zip file or folder")
    args = parser.parse_args()

    try:
        content_list = process_input(args.input)
        analysis_result = analyze_content(content_list)
        print("Analysis Result:")
        print(analysis_result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()