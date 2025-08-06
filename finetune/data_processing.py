import pandas as pd
from datasets import Dataset
from config import Config

def load_data(file_path):
    """
    Load CSV data into a pandas DataFrame.

    This function reads a CSV file, drops rows with missing 'question' or 
    'answer' fields, and prints the number of valid samples loaded.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame containing 'question' and 'answer' columns.
    """
    """Load CSV data"""
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['question', 'answer'])
    print(f"Loaded {len(df)} samples from {file_path}")
    return df

def format_data(examples):
    """
    Format raw QA pairs into language model-compatible text prompts.

    For each question-answer pair, this function generates a formatted string 
    with special tokens indicating user and assistant turns, compatible with 
    chat-style language model training.

    Args:
        examples (dict): Dictionary with keys 'question' and 'answer'.

    Returns:
        dict: Dictionary with a single key 'text' containing formatted strings.
    """
    """Format data for training"""
    texts = []
    for i in range(len(examples["question"])):
        text = f"""<|user|>
{examples["question"][i]}
<|assistant|>
{examples["answer"][i]}<|end_of_text|>"""
        texts.append(text)
    return {"text": texts}

def prepare_dataset(file_path):
    """
    Load and format data into a Hugging Face Dataset.

    This function loads a CSV using `load_data`, then formats the data using 
    `format_data`, and returns a Hugging Face `Dataset` object suitable for 
    tokenization and training.

    Args:
        file_path (str): Path to the CSV file containing QA pairs.

    Returns:
        datasets.Dataset: Hugging Face dataset object with formatted text entries.
    """
    """Create dataset from CSV"""
    df = load_data(file_path)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(format_data, batched=True)
    return dataset