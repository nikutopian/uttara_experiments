import argparse
import os
import zipfile
from typing import List, Dict
import ollama
from file_parser import FileParser
import sys

sys.stdout.reconfigure(encoding='utf-8')

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
def analyze_content(content_mapping: Dict[str, str], expense_policy: str, model_name: str) -> str:
    sys_prompt = "You are a finance controller at a company, responsible for ensuring that all submitted expense documents adhere to the company's expense policy. Your task is to carefully review each expense document and determine if it meets the policy requirements. Provide clear and concise feedback based on your assessment."
    client = ollama.Client()

    full_output = ''

    for file_path, content in content_mapping.items():
        # Prepare output for each file
        output = "-"*10
        output += f"File: {file_path}"
        output += "-"*10
        # output += f"Content: {content}\n"
        # output += "-"*100
        output += "\n"

        full_output += output
        print(output)

        # Prepare user prompt for the LLM
        user_prompt = f"""
            Expense Document: {content}\n\n

            Expense Policy: {expense_policy}\n\n

            Review the contents of Expense Document and compare it with the Expense Policy. Follow the instructions carefully and do not provide open ended strings in your response.

            Instructions:
                1. Response should be exactly one of the three string - "Conforms", "Does not Conform" or "Not an Expense Document".
                2. If the Expense Document conforms to the policy, respond with "Conforms"
                3. If the Expense Document does not conform to the policy, respond with "Does not Conform".
                4. If the Expense Document does not represent an expense report, respond with "Not an Expense Document".
            """
        response = client.chat(model=model_name, messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])
        audit_decision = response['message']['content']
        full_output += f"Audit Decision: {audit_decision}\n"

        user_prompt = f"""

            Expense Document: {content}\n\n

            Expense Policy: {expense_policy}\n\n

            Audit Decision: {audit_decision}\n\n

            Review the contents of Expense Document and compare it with the Expense Policy. Follow the instructions carefully and do not provide open ended strings in your response.

            Instructions:
                1. Response should be exactly one of the two string - "N/A" or a string explaining why the Expense Document does not conform with Expense Policy.
                2. If the Audit Decision is "Conforms", respond with "N/A".
                3. If the Audit Decision is "Does not Conform" figure out the primary reason for non-conformance and respond back with the reason.
                4. If the Audit Decision is "Not an Expense Document", respond with "N/A".
            """
        response = client.chat(model=model_name, messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ])
        decision_reason = response['message']['content']
        full_output += f"Reason: {decision_reason}\n"

        user_prompt = f"""
            Expense Document: {content}\n\n

            Expense Policy: {expense_policy}\n\n

            Audit Decision: {audit_decision}\n\n

            Reason: {decision_reason}\n\n

            Please figure out the confidence level in the Audit Decision and Reason for the given Expense Document by comparing it with Expense Policy. Respond back with one of the three strings - "Low", "Medium", or "High" depending on the level of confidence. Do not respond with anything else.
            """
        response = client.chat(model=model_name, messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ])
        audit_confidence = response['message']['content']
        full_output += f"Audit Confidence: {audit_confidence}\n"

        print(f"Audit Decision: {audit_decision}\n")
        print(f"Reason: {decision_reason}\n")
        print(f"Audit Confidence: {audit_confidence}\n")
        print("*"*100)

    return full_output

# Main function to handle argument parsing and orchestrate the processing and analysis
def main():
    parser = argparse.ArgumentParser(description="Analyze files using LLM")
    parser.add_argument("-i", "--input", required=True, help="Path to zip file or folder")
    parser.add_argument("-p", "--policy", required=True, help="Path to expense policy document")
    parser.add_argument("-o", "--output", required=True, help="Path to output file")
    parser.add_argument("-m", "--model", required=True, help="Ollama model identifier")
    args = parser.parse_args()

    try:
        # Process input files
        content_list = process_input(args.input)
        file_parser = FileParser()
        # Parse the expense policy document
        expense_policy = file_parser.parse_file(args.policy)
        # Analyze the content of the files
        analysis_result = analyze_content(content_list, expense_policy, args.model)
        # Write the analysis result to the output file
        with open(args.output, "w") as f:
            f.write(analysis_result)
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()