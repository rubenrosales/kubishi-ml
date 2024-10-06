# Import necessary libraries
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np

# Set model checkpoints and project name
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-en-mul"
PROJECT_NAME = f"{MODEL_CHECKPOINT}_train-base-model-80_train"

# Load the tokenizer
def load_tokenizer(checkpoint):
    return AutoTokenizer.from_pretrained(checkpoint, return_tensors="pt")

# Load dataset from CSV
def load_dataset(filename):
    return pd.read_csv(filename)

# Preprocess data for model input
def preprocess_data(tokenizer, data, src, tgt, max_length):
    model_inputs = tokenizer(
        data[src], text_target=data[tgt], max_length=max_length, truncation=True
    )
    return model_inputs

# Tokenize the dataset
def tokenize_dataset(tokenizer, dataframe, src, tgt, max_length):
    data = dataframe[[tgt, src]]
    dataset = Dataset.from_pandas(data)
    tokenized_dataset = dataset.map(
        lambda x: preprocess_data(tokenizer, x, src, tgt, max_length),
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized_dataset

# Compute metrics
def compute_metrics(eval_preds, tokenizer, metric):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# Main function to run the training
def main():
    # Load tokenizer
    tokenizer = load_tokenizer(MODEL_CHECKPOINT)

    # Load dataset
    filename = "../data/random_good_translations.csv"
    df = load_dataset(filename)[:80]

    # Set parameters
    src = "eng"
    tgt = "ovp"
    max_length = 512

    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(tokenizer, df, src, tgt, max_length)

    # Set up training arguments
    training_args = Seq2SeqTrainingArguments(
        PROJECT_NAME,
        evaluation_strategy="no",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=128,
        per_device_eval_batch_size=4,
        weight_decay=0.001,
        save_total_limit=3,
        num_train_epochs=500,
        predict_with_generate=True,
        fp16=True,
    )

    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    metric = evaluate.load("sacrebleu")

    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer, metric)
    )

    # Start training
    trainer.train()

# Execute the main function
if __name__ == "__main__":
    main()
