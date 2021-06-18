import os
from dataset import MPedDataset, MPedFromExample
import torch
from torch.utils.data import random_split, ConcatDataset, DataLoader
from transformers import BertTokenizerFast, DataCollatorForLanguageModeling, Trainer, TrainingArguments, EvalPrediction,\
    BertConfig
from model import MyBertForSequenceClassification, BertForSequenceClassification
import numpy as np
import random
from sklearn import metrics
from sampler import ImbalancedDatasetSampler
import sys
import pandas as pd

context = 10
print(context, 10)

category = 'strategy'
num_labels = 13
epochs_range = range(5, 6)


tokenizerbase = "bert-base-uncased"
batch_size = 32
basemodel = "./models/MPedBERT"
outdir = f"./models/MPedBERTBase-{category}"


def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(p: EvalPrediction):
    THRESHOLD = 0.5
    predictions = 1 / (1 + np.exp(-p.predictions))  # sigmoid

    print("p.predictions (sigmoid)")
    print(*predictions, sep=", ")

    print("original labels")
    print(*p.label_ids, sep=", ")
    
    label_results = []
    for item in p.label_ids:
        result = set()
        for index, value in enumerate(item):
            if value == 1:
                result.add(index)
        if len(result) == 0:
            result.add(0)
        label_results.append(result)
    print("labels_results", label_results)


    # predict the one with highest confidence
    pred_results = np.argmax(p.predictions, axis=1)
    print("pred_results", pred_results)


    # If the predicted label match one of the annotated label, use the matched label as the actual label.
    modified_labels = []
    modified_preds = []

    for index, value in enumerate(pred_results):
        if pred_results[index] in label_results[index]:
            intersected_label = pred_results[index]
            modified_labels.append(intersected_label)
            modified_preds.append(intersected_label)
        else:
            modified_labels.append(label_results[index].pop())
            modified_preds.append(pred_results[index])

    print("modified_labels", modified_labels)
    print("modified_preds", modified_preds)


    classification_report = metrics.classification_report(modified_labels, modified_preds, digits=3, output_dict=True)

    return {"macro-avg-f1-score": classification_report['macro avg']['f1-score'], "labels": modified_labels, "preds":  modified_preds}



class MyTrainer(Trainer):

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.
        """

        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_sampler = ImbalancedDatasetSampler(self.train_dataset, num_samples=len(self.train_dataset) * 2, callback_get_label=self._get_label)

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
        )

        return data_loader

    def _get_label(self, dataset, index):

        top_k_value, top_k_ind = torch.topk(dataset[index]["labels"], 2)
        top_k_ind = top_k_ind.numpy()
        top_k_value = top_k_value.numpy()
        if top_k_value[1] == 0:  # if no second label
            return top_k_ind[:1]
        else:
            return top_k_ind


tokenizer = BertTokenizerFast.from_pretrained(tokenizerbase)
full_dataset = MPedDataset(tokenizer, annotated=True, category=category, context_width=context, block_size=256)
print("get full_dataset")

datasets = []
documents = MPedDataset.documents(MPedDataset.RATERS, MPedDataset.MPED)
print("documents")

for document in documents:
    dataset = MPedFromExample(full_dataset.get_samples([document]))
    datasets.append(dataset)
# print("datasets", datasets)


splits = []
for i in range(len(datasets)):
    evaldoc = datasets[i]
    # evaldoc = datasets[i + 1 if i + 1 < len(datasets) else 0]
    traindocs = [x for x in datasets if x != evaldoc]
    splits.append([traindocs, evaldoc])


last_f1 = 0
for epochs in epochs_range:

    eval_split_metric = []
    test_split_metric = []

    df_total = pd.DataFrame(columns=['split', 'label', 'pred'])

    for split_idx, split in enumerate(splits):
        model = BertForSequenceClassification.from_pretrained(
            basemodel,
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )

        print(f"Starting split {split_idx + 1}/{len(splits)}", flush=True)

        traindocs, evaldoc = split

        trainset = ConcatDataset(traindocs)
        evalset = evaldoc

        training_args = TrainingArguments(
            output_dir=f"{outdir}-split-{split_idx}",
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            save_total_limit=1,
            evaluate_during_training=False,
        )

        trainer = MyTrainer(
            model=model,
            args=training_args,
            train_dataset=trainset,
            eval_dataset=evalset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics_ = trainer.evaluate()

        eval_macro_avg_f1 = metrics_["eval_macro-avg-f1-score"]
        eval_split_metric.append(eval_macro_avg_f1)
        mean_eval_macro_avg_f1 = np.mean(eval_macro_avg_f1)

        labels = metrics_["eval_preds"]
        predictions = metrics_["eval_labels"]
        print("labels", labels)
        print("predictions", predictions)

        df = pd.DataFrame(columns=['split', 'label', 'pred'])

        df['split'] = split_idx
        df['label'] = labels
        df['pred'] = predictions

        df_total = df_total.append(df)
        df_total.to_csv("./confusion_matrix.csv")

        print(f"Category: ${category} - Context: ${context} - Epochs: ${epochs} - f1: ${mean_eval_macro_avg_f1}", flush=True)
