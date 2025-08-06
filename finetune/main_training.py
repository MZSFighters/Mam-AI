"""
Main script to fine-tune a language model on a medical question-answering dataset.

This script sets up the model, loads and formats the training/validation data,
trains the model using LoRA fine-tuning, and saves the trained model and tokenizer
to disk. It uses components from the following modules:

"""
from config import Config
from model_setup import setup_model
from data_processing import prepare_dataset
from training import train_model
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64 

def main():
    """
    Main execution pipeline for fine-tuning the medical QA model.

    Steps:
        1. Load model and tokenizer using LoRA-compatible setup.
        2. Load and preprocess training and validation datasets.
        3. Train the model using Hugging Face Trainer.
        4. Save the fine-tuned model and tokenizer to disk.
    """
    print("=== Medical Q&A Fine-tuning ===")
    
    # Setup model
    model, tokenizer = setup_model()
    
    # Load data
    train_dataset = prepare_dataset(Config.TRAIN_DATA)
    val_dataset = prepare_dataset(Config.VAL_DATA)
    
    # Train
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # Save model
    print(f"Saving model to {Config.OUTPUT_DIR}")
    model.save_pretrained(Config.OUTPUT_DIR)
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    # print("Done! Run 'python test_model.py' to test the model")

if __name__ == "__main__":
    main()