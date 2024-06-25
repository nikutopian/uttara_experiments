# Uttara Experiments

## Expense Policy Validation

### Environment Setup 

Install Conda from https://docs.anaconda.com/miniconda/

Install OLLAMA from https://ollama.com/download

```bash
conda create env -f environment.yml
ollama pull mistral
```

### Run the program

```bash
python expense_policy_report.py \
    -p <path_to_expense_policy> \
    -i <path_to_zip_file_or_folder> \
    -o <path_to_output_file>
```
