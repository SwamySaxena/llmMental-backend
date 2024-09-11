# train_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import torch.nn as nn

# Load the smaller base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained('distilgpt2')
tokenizer = AutoTokenizer.from_pretrained('distilgpt2')

# Set the padding token to the end-of-sequence (EOS) token
tokenizer.pad_token = tokenizer.eos_token

# Define a custom LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, input_dim, output_dim, rank=8, alpha=32):
        super(LoRALayer, self).__init__()
        self.rank = rank
        self.alpha = alpha

        # Define low-rank matrices
        self.A = nn.Parameter(torch.randn(input_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.randn(rank, output_dim) * 0.01)

    def forward(self, x):
        return x + self.alpha * (x @ self.A @ self.B)

# Function to selectively apply LoRA to specific layers
def apply_lora_to_model(model, rank=8, alpha=32):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.size(0) == 768 and module.weight.size(1) == 768:
            input_dim, output_dim = module.weight.size()
            lora_layer = LoRALayer(input_dim, output_dim, rank=rank, alpha=alpha)
            model._modules[name] = nn.Sequential(module, lora_layer)
    return model

# Apply LoRA to the model
model = apply_lora_to_model(base_model)

# Fine-tune the model
def fine_tune_model():
    # Load the EmpatheticDialogues dataset directly from Hugging Face
    dataset = load_dataset('empathetic_dialogues', cache_dir='./dataset_cache')

    # Tokenize dataset
    def tokenize_function(examples):
        tokenized = tokenizer(examples['utterance'], padding="max_length", truncation=True)
        tokenized['labels'] = tokenized['input_ids'].copy()
        return tokenized

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
    )

    # Start fine-tuning
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained('./fine_tuned_model')
    tokenizer.save_pretrained('./fine_tuned_model')

if __name__ == "__main__":
    fine_tune_model()
