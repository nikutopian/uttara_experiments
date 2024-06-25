# Uttara Experiments

## Expense Policy Validation

### Environment Setup 

Install Conda from https://docs.anaconda.com/miniconda/

```bash
conda create env -f environment.yml
conda activate utenv
```

Alternatively, use python Virtul Environment

```bash
python -m venv utvenv

utvenv\Scripts\activate [In Windows]
source utvenv/bin/activate [In Mac]

pip install -r requirements.txt
```

Install OLLAMA from https://ollama.com/download

```bash
ollama pull mistral
```

### Run the program

```bash
python expense_policy_report.py \
    -p <path_to_expense_policy> \
    -i <path_to_zip_file_or_folder> \
    -o <path_to_output_file>
```
