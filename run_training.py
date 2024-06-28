# %%


# %%
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModelForSeq2SeqLM
# tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
# model = BertModel.from_pretrained('bert-base-multilingual-cased')
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
model_checkpoint="Helsinki-NLP/opus-mt-en-mul"
#model_checkpoint="google-t5/t5-large"
PROJECT_NAME=f"{model_checkpoint}_train-base-model-fixed-source"

# model_checkpoint = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

# %%


# %%
import pandas as pd


df = pd.read_csv('trans-sentences-long.csv')[:2000]

# %%
training_corpus = [df["sentence"][i: i + 10].values for i in range(0, len(df["sentence"]), 10)]
# %%
new_tokenizer = tokenizer#tokenizer.train_new_from_iterator(training_corpus, 129547)

# %%
max_length = 512
inputs = df['translation'].values.tolist()
targets = df['sentence'].values.tolist()

# %%
def preprocess_data(data):

    model_inputs = new_tokenizer(
    data['translation'], text_target=data['sentence'], max_length=max_length, truncation=True
    )

    return model_inputs

# %%
from datasets import Dataset
data = df[['sentence', 'translation']]
_dataset = Dataset.from_pandas(data)


# %%
tokenized_dataset = _dataset.map(    preprocess_data,
    batched=True, 
     remove_columns=_dataset.column_names,
)

# %%
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# %%
import evaluate

metric = evaluate.load("sacrebleu")

# %%
import numpy as np


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the prediction logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = new_tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, new_tokenizer.pad_token_id)
    decoded_labels = new_tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# %%
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# %%
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(new_tokenizer, model=model)

# %%
tokenized_dataset

# %%

from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
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
trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=new_tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics
)


trainer.train()