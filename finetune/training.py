from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
from config import Config


def train_model(model, tokenizer, train_dataset, val_dataset=None):
    """
    Fine-tune a language model using supervised instruction tuning.

    This function sets up and runs training using the `SFTTrainer` from TRL with 
    Hugging Face's `TrainingArguments`. It supports optional evaluation with a 
    validation dataset and automatically handles bf16/fp16 precision.

    Args:
        model (PreTrainedModel): The base or LoRA-adapted model to fine-tune.
        tokenizer (PreTrainedTokenizer): The tokenizer corresponding to the model.
        train_dataset (datasets.Dataset): Hugging Face dataset with formatted training text.
        val_dataset (datasets.Dataset, optional): Dataset for evaluation during training.

    Returns:
        SFTTrainer: The trainer instance after training is complete.
    """

    """Train the model"""
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=Config.MAX_SEQ_LENGTH,
        args=TrainingArguments(
            output_dir=Config.OUTPUT_DIR,
            per_device_train_batch_size=Config.BATCH_SIZE,
            num_train_epochs=Config.EPOCHS,
            learning_rate=Config.LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            save_strategy="epoch",
            report_to=None,
        ),
    )
    
    print("Starting training...")
    trainer.train()
    print("Training complete!")
    
    return trainer