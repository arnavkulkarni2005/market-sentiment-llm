from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from datasets import Dataset
import torch
import os
import pandas as pd
from utils.data_loader import load_sentences

MODEL_NAME = "ProsusAI/finbert"

def preprocess_data(df):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["headline"], padding="max_length", truncation=True)

    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset, tokenizer

def main():
    df = load_sentences("data\Sentiment_Stock_data.csv")
    tokenized_dataset, tokenizer = preprocess_data(df)
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.2)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    training_args = TrainingArguments(
        output_dir="models/",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="logs/"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"]
    )

    trainer.train()
    model.save_pretrained("models/")
    tokenizer.save_pretrained("models/")

if __name__ == "__main__":
    main()
