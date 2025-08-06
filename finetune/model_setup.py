from unsloth import FastLanguageModel
from config import Config


# target modules for Llama/Gemma models
target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

def setup_model():

    """
    Load and configure the base model with LoRA using Unsloth.

    This function:
    - Loads the pre-trained model and tokenizer using the FastLanguageModel interface.
    - Applies LoRA (Low-Rank Adaptation) to specified target modules.
    - Enables gradient checkpointing and memory optimizations.
    
    Returns:
        model (transformers.PreTrainedModel): The LoRA-adapted model ready for training.
        tokenizer (transformers.PreTrainedTokenizer): The corresponding tokenizer.
    """
    """Load and configure model"""
    print(f"Loading model: {Config.MODEL_NAME}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=Config.MODEL_NAME,
        max_seq_length=Config.MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        target_modules=target_modules,
        modules_to_save=["lm_head", "embed_tokens"],
        use_gradient_checkpointing="unsloth",  
        random_state=3407,                     
        use_rslora=False,                      
        loftq_config=None,                     
    )


    print("Model setup complete")
    return model, tokenizer