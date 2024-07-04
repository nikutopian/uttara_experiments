import argparse
import os
import zipfile
from typing import List, Dict
import ollama
from file_parser import FileParser
import sys
import glob
from api_llm import (
    OpenAILLM,
    OpenAIModelType,
)

sys.stdout.reconfigure(encoding='utf-8')

# Function to process input path (zip file or folder) and return a mapping of file paths to their content
def process_file(filename: str) -> str:
    file_parser = FileParser()
    file_mapping = {}

    # Check if the input is a zip file
    if os.path.isfile(filename):
        content = file_parser.parse_file(filename)
    else:
        raise ValueError("Input must be a file")

    return content

# Function to analyze the content of files against the expense policy
def analyze_file(filename: str, query: str) -> str:
    sys_prompt = "You are a business analyst at a large bank responsible for extracting accurate information about a potential investment from the provided document."
    
    output = "-"*10
    output += f"Input File: {filename}"
    output += "-"*10
    print(output)

    content = process_file(filename)

    llm = OpenAILLM(OpenAIModelType.GPT4_O)
    
    # Prepare user prompt for the LLM
    user_prompt = f"""
        Extract an answer for the provided query from the document contents listed below. Accuracy is the highest priority so, please make sure your answer is grounded in the contents. If it is not possible to answer the input query based on the contents of the document, please repond back with just "Cannot Find The Answer". Nothing else should be output. 
        
        File Content: {content}\n\n

        Input Query: {query}\n\n
        """
    
    response = llm.chat(
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
        )
    
    answer = response.choices[0].message.content
    
    print(f"Answer: {answer}\n")
    print("*"*100)

# Main function to handle argument parsing and orchestrate the processing and analysis
def main():
    parser = argparse.ArgumentParser(description="Analyze files using LLM")
    parser.add_argument("-i", "--input", required=True, help="Path to folder containing investment documents")
    args = parser.parse_args()

    query = "What is the profit for Indifi for FY22? Do not calculate. Extract this information from the document."

    print(f"\n\nInput Query: {query}\n\n")

    for filename in glob.iglob(args.input + '**/**', recursive=True):
        if os.path.isfile(filename):    
            analyze_file(filename, query)

if __name__ == "__main__":
    main()