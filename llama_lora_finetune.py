from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    logging,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, PeftModel, get_peft_model
import accelerate
from datasets import load_dataset
import torch

#Defining Important parameters
use_4bit = True
use_nested_quant = False
device_map = {"": 0}
bnb_4bit_compute_dtype="float16"
bnb_4bit_quant_type = "nf4"
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

# Load model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B", quantization_config=bnb_config, device_map=device_map)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    label_names=["labels"],
    learning_rate=2e-5,
    fp16=True,  # Use mixed precision if GPU supports it
    save_steps=500,
)

# Prepare dataset
dataset_ACI = load_dataset("allenai/scitldr", "AIC")

def preprocess_function(examples):

    # Flatten the source and summary text lists
    source_texts = [' '.join(source) for source in examples['source']]  # Flatten each source text
    summary_texts = [' '.join(summary) for summary in examples['target']]  # Flatten each summary


    # print(source_texts[:2])
    # print(summary_texts[:2])

    # Tokenize the source text (abstract, introduction, conclusion)
    inputs = tokenizer(source_texts, padding="max_length", truncation=True, max_length=1024, return_tensors='pt')
    # Tokenize the summary (labels)
    labels = tokenizer(summary_texts, padding="max_length", truncation=True, max_length=150, return_tensors = 'pt')

    # Return tokenized inputs and labels for fine-tuning
    inputs['labels'] = labels['input_ids']
    return inputs

# Apply preprocessing to the dataset
tokenized_data = dataset_ACI.map(preprocess_function, batched=True)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
)

# Start training
trainer.train()
