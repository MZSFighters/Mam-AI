class Config:
    """
        Configuration class for setting up model training parameters and paths.

        This class centralizes all configurable settings used for training a 
        medical question-answering model. It includes model architecture choices, 
        LoRA fine-tuning parameters, training hyperparameters, dataset file paths, 
        and the directory for saving the trained model.

        Attributes:
            MODEL_NAME (str): Name or path of the pre-trained model to be used.
            MAX_SEQ_LENGTH (int): Maximum sequence length for tokenized inputs.
            
            LORA_R (int): Rank value for LoRA (Low-Rank Adaptation) layers.
            LORA_ALPHA (int): Scaling factor for LoRA layers.
            
            BATCH_SIZE (int): Number of samples per training batch.
            EPOCHS (int): Number of training epochs.
            LEARNING_RATE (float): Learning rate for the optimizer.
            
            TRAIN_DATA (str): Path to the training dataset CSV file.
            VAL_DATA (str): Path to the validation dataset CSV file.
            TEST_DATA (str): Path to the test dataset CSV file.
            
            OUTPUT_DIR (str): Directory where the trained model and tokenizer will be saved.
        """
    # Model settings
    MODEL_NAME = "unsloth/gemma-3n-E4B-it" #unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit
    MAX_SEQ_LENGTH = 1024
    
    # LoRA settings
    LORA_R = 16
    LORA_ALPHA = 16
    
    # Training settings
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-4
    
    # Data paths
    TRAIN_DATA = "data/train_medical_qa.csv"
    VAL_DATA = "data/val_medical_qa.csv"
    TEST_DATA = "data/test_medical_qa.csv"
    
    # Output
    OUTPUT_DIR = "medical_qa_model"