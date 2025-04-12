from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("pubmed_qa", "pqa_labeled")
train_dataset = dataset["train"].select(range(800))
eval_dataset = dataset["train"].select(range(800, 1000))


def format_pubmedqa(examples):
    return {
        "question": examples["question"],
        "context": examples["context"],
        "answer": examples["long_answer"]
    }

dataset = train_dataset.map(format_pubmedqa, remove_columns=train_dataset.column_names)


model_id = "microsoft/Phi-3-mini-4k-instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token


peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)

training_args = TrainingArguments(
    output_dir="./phi3-pubmedqa-finetuned",
    per_device_train_batch_size=4, 
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    learning_rate=1e-4,  
    optim="adamw_torch",
    evaluation_strategy="steps",
    eval_steps=50
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    dataset_text_field="question",
    max_seq_length=512,
    args=training_args,
    formatting_func=lambda x: f"<|user|>\nQUESTION: {x['question']}\nCONTEXT: {x['context']}\n<|assistant|>\nANSWER: {x['answer']} EOS"
)

trainer.train()
trainer.save_model("./phi3-pubmedqa-finetuned")