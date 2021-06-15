from pathlib import Path
from typing import List, TextIO

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

OFF_TASKS = "Off-task"

CATEGORIES = {
    "domain": [
        "Introduction",
        "Guideline",
        "Benefit",
        "Goal",
        "Monitoring",
        "Support",
        "Self-efficacy",
        "Motivation",
        "Barrier",
        "Relapse",
        "Safety",
        "Diet",
        "Weight",
        "Off-task",
    ],
    "strategy": [
        "None",
        "Guideline",
        "Benefit",
        "Goal",
        "Monitoring",
        "Support",
        "Self-efficacy",
        "Motivation",
        "Barrier",
        "Relapse",
        "Safety",
        "Diet",
        "Weight",
    ],
    "strategy1": [
        "None",
        "Guideline",
        "Benefit",
        "Goal",
        "Monitoring",
        "Support",
        "Self-efficacy",
        "Motivation",
        "Barrier",
        "Relapse",
        "Safety",
        "Diet",
        "Weight",
    ],
    "strategy2": [
        "None",
        "Guideline",
        "Benefit",
        "Goal",
        "Monitoring",
        "Support",
        "Self-efficacy",
        "Motivation",
        "Barrier",
        "Relapse",
        "Safety",
        "Diet",
        "Weight",
    ],
    "social_exchange": [
        "None",
        "Agree",
        "Incomplete",
        "Approve/Encourage",
        "Disapprove/Discourage",
        "Greeting",
        "Goodbye",
    ],
    "task_focused": [
        "None",
        "Orient",
        "Ask-GenInfo",
        "Give-GenInfo",
        "Ask-PerInfo",
        "Give-PerInfo",
        "Check-Understanding",
        "Ask-Repeat",
        "Ask-Opinion",
        "Give-Opinion",
    ],
}


class MPedFromExample(Dataset):
    def __init__(self, examples, withPos=False):
        self.examples = examples
        self.withPos = withPos

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.withPos:
            return {
                "input_ids": torch.tensor(self.examples[i][0], dtype=torch.long),
                "attention_mask": torch.tensor(self.examples[i][1], dtype=torch.long),
                "labels": torch.tensor(self.examples[i][2], dtype=torch.long),
                "pos_in_doc": torch.tensor(self.examples[i][3], dtype=torch.float32)
            }

        return {
            "input_ids": torch.tensor(self.examples[i][0], dtype=torch.long),
            "attention_mask": torch.tensor(self.examples[i][1], dtype=torch.long),
            "labels": torch.tensor(self.examples[i][2], dtype=torch.long)
        }


class MPedDataset(Dataset):
    PREPROCESS_OUTDIR_PATH = "preprocessed"
    BASE_DATA_DIR = Path("./data")

    # RATERS = ["Erika", "Kiley"]
    RATERS = ["annotated_selected"]
    UNANNOTATED = ["unannotated"]
    PRETRAINING = ["annotated_selected", "unannotated"]
    ALL = ["annotated_selected", "unannotated"]

    ROUNDS = ["First", "Second", "Third", "Fourth"]
    MANUAL = 'Finalized'
    MPED = 'MPed'

    def __init__(self, tokenizer, annotated=True, category='pretraining', context_width=0, block_size=512,
                 documents=None, raters=["annotated_selected"], dataset="MPed", withPos=False):
        self.withPos = withPos
        self.annotated = annotated
        self.context_width = context_width
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.for_pretraining = category == 'pretraining' if annotated else True
        self.category = 'pretraining' if self.for_pretraining else category

        if self.for_pretraining:
            self.data = self._add_context(self._load_clean_data(self.PRETRAINING, dataset), context_width)
        elif annotated:
            for_analysis = self._process_annotated_data_for_analysis(self._load_clean_data(raters, dataset))
            self.data = self._add_context(self._annotations_to_one_hot(for_analysis), context_width)
        else:
            self.data = self._add_context(self._load_clean_data(self.UNANNOTATED, dataset), context_width)
            # print("self.data", self.data)

        if documents is not None:
            self.data = self._filter_documents(self.data, documents)

        if not self.for_pretraining:
            mask = (self.data[self.category].isnull()) | (self.data[self.category].isna()) | (
                        self.data[self.category] == None)
            self.data = self.data[~mask]

        self.examples = []
        if self.for_pretraining:
            batch_encoding = tokenizer.batch_encode_plus(
                [x for x in self.data.utterance.values],
                add_special_tokens=True,
                max_length=block_size,
                truncation=True,
                pad_to_multiple_of=32
            )
            self.examples = batch_encoding["input_ids"]
        else:
            self.data = self.data[~self.data.utterance.isna()]
            self.data = self.data[~self.data.context.isna()]
            self.data = self.data[~self.data[self.category].isnull()]
            utterances = [x for x in self.data.utterance.values]
            context = [' '.join(x) for x in self.data.context.values]

            sentence_pairs = list(zip(context, utterances))
            labels = [x for x in self.data[self.category].values]
            pos_in_doc = [x for x in list(zip(self.data.pos_in_doc.values, self.data.speaker.values))]

            batch_encoding = tokenizer.batch_encode_plus(
                sentence_pairs,
                add_special_tokens=True,
                max_length=block_size,
                truncation=True,
                truncation_strategy="longest_first",
                pad_to_max_length=True,
                return_attention_mask=True
            )
            self.examples = list(zip(batch_encoding["input_ids"], batch_encoding["attention_mask"], labels, pos_in_doc))

    def get_data(self, documents):
        if documents is not None:
            data = self._filter_documents(self.data, documents)
            data = data[~data.utterance.isna()]
            data = data[~data.context.isna()]
            if self.annotated:
                data = data[~data[self.category].isnull()]
            return data

    def get_samples(self, documents):
        data = self.get_data(documents)
        # print("data",data)
        utterances = [x for x in data.utterance.values]
        context = [' '.join(x) for x in data.context.values]
        sentence_pairs = list(zip(context, utterances))
        # print("sentence_pairs", sentence_pairs)
        if self.annotated:
            labels = [x for x in data[self.category].values]
        else:
            labels = [np.zeros(13) for x in data.utterance.values]  # TODO: for unannotated data
        pos_in_doc = [[x] for x in data.pos_in_doc.values]
        batch_encoding = self.tokenizer.batch_encode_plus(
            sentence_pairs,
            add_special_tokens=True,
            max_length=self.block_size,
            truncation=True,
            truncation_strategy="longest_first",
            pad_to_max_length=True,
            return_attention_mask=True
        )
        return list(zip(batch_encoding["input_ids"], batch_encoding["attention_mask"], labels, pos_in_doc))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.for_pretraining:
            return torch.tensor(self.examples[i], dtype=torch.long)
        else:
            if self.withPos:
                return {
                    "input_ids": torch.tensor(self.examples[i][0], dtype=torch.long),
                    "attention_mask": torch.tensor(self.examples[i][1], dtype=torch.long),
                    "labels": torch.tensor(self.examples[i][2], dtype=torch.long),
                    "pos_in_doc": torch.tensor(self.examples[i][3], dtype=torch.float32)
                }

            return {
                "input_ids": torch.tensor(self.examples[i][0], dtype=torch.long),
                "attention_mask": torch.tensor(self.examples[i][1], dtype=torch.long),
                "labels": torch.tensor(self.examples[i][2], dtype=torch.long)
            }

    ### Preprocessing

    def _add_context(self, data: pd.DataFrame, context_width):
        def is_same_file(data, i, j):
            doceq = data.loc[i, 'document'] == data.loc[j, 'document']
            return doceq

        contexts = []
        for line_idx in range(0, len(data)):
            context = []
            for step_size in range(1, context_width + 1):
                if line_idx - step_size >= 0 and is_same_file(data, line_idx, line_idx - step_size) and isinstance(
                        data.loc[line_idx - step_size, 'utterance'], str):
                    context.append(data.loc[line_idx - step_size, 'utterance'])
                else:
                    context.append("")
            contexts.append(context)

        data["context"] = contexts

        return data

    def _process_annotated_data_for_analysis(self, annotated_raw: pd.DataFrame) -> pd.DataFrame:
        annotated_raw = self._auto_annotate(annotated_raw)
        annotated_raw = self._annotations_to_categorical(annotated_raw)
        return annotated_raw

    # Retains only those documents that are given
    def _filter_documents(self, data: pd.DataFrame, documents: list) -> pd.DataFrame:
        mask = data.document.isin(documents)
        return data[mask].reset_index(drop=True)

    def _load_clean_data(self, raters, dataset) -> pd.DataFrame:
        df = self._load_raw_data(raters, dataset)
        df = self._remove_transcript_annotations(df)
        df = self._add_pos_in_doc(df)

        return df

    def _add_pos_in_doc(self, data) -> pd.DataFrame:
        def add_pos_in_doc(group):
            max_row = max(group.row.values)
            group['pos_in_doc'] = group.row.divide(max_row)
            return group

        return data.groupby(['document']).apply(add_pos_in_doc)

    @classmethod
    def documents(cls, raters, dataset) -> list:
        data = cls._load_raw_data(raters, dataset)
        return data.document.unique()

    # Loads the raw data from the given raters
    @classmethod
    def _load_raw_data(cls, raters, dataset) -> pd.DataFrame:
        def _rename_column(column_name: str) -> str:
            return column_name.lower().replace("-", "_").replace(" ", "_")

        data = None

        for rater in raters:
            # print("rater: ", rater)

            data_dir = cls.BASE_DATA_DIR / dataset / rater
            # print("data_dir: ", data_dir)

            if not data_dir.exists():
                print(f"skipping {data_dir}: does not exist")
                continue

            files = list(data_dir.glob("*.xlsx"))
            for file in files:
                print("file", file)
                df = cls.get_dataframe(_rename_column, file, rater)

                if data is None:
                    data = df
                else:
                    data = data.append(df, ignore_index=True)

        return data



    @classmethod
    def get_dataframe(cls, _rename_column, file, rater=""):
        excel_columns = [0, 2, 3, 4, 5, 6, 7]
        df = pd.read_excel(file, usecols=excel_columns)
        df = df.rename(_rename_column, axis="columns")
        df.insert(0, "rater", rater)
        df.insert(1, "dataset", "MPed")
        df.insert(2, "document", file.stem)
        df.insert(3, "row", df.apply(lambda x: x.name, axis=1))
        return df

    # Removes annotations coming from transcript such as [LAUGHTER]
    def _remove_transcript_annotations(self, data: pd.DataFrame) -> pd.DataFrame:
        mask1 = (data.utterance.str.startswith("(")) & (
            data.utterance.str.endswith(")")
        )
        mask2 = (data.utterance.str.startswith("[")) & (
            data.utterance.str.endswith("]")
        )

        mask = mask1 | mask2

        ret = data[~mask]

        def remove_annotations(sentence):
            words = str(sentence).split(' ')

            def is_annotation(word):
                return word.startswith("[") or word.endswith("]") or word.startswith("(") or word.endswith(")") or (
                            ":" in word)

            new_words = [w for w in words if not is_annotation(w)]
            return ' '.join(new_words)

        ret['utterance'] = data.utterance.apply(lambda x: remove_annotations(x))
        return ret.reset_index(drop=True)

    # Automatically produces annotations based on agreed upon rules
    def _auto_annotate(self, data: pd.DataFrame) -> pd.DataFrame:
        # Automatically annotate Off-task utterances as 'None'
        columns = ["strategy1", "strategy2", "social_exchange", "task_focused"]
        mask = data.domain == OFF_TASKS
        for column in columns:
            data.loc[mask, column] = "None"

        # Fix incorrect, "None" annotations
        mask = data.domain == "None"
        data.loc[mask, "domain"] = OFF_TASKS

        # Fix incorrect "Off-task" annotations
        mask = data.social_exchange == OFF_TASKS
        data.loc[mask, "social_exchange"] = "None"

        return data

    # Converts annotation categories into catagorical data
    def _annotations_to_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        categories = [
            "domain",
            "strategy1",
            "strategy2",
            "social_exchange",
            "task_focused",
        ]

        for category in categories:
            data[category] = pd.Categorical(data[category], categories=CATEGORIES[category])

        return data

    # Returns DataFrame, each line missing some annotation
    def _missing_annotations(self, data: pd.DataFrame) -> pd.DataFrame:
        columns = [
            "domain",
            "strategy1",
            "strategy2",
            "social_exchange",
            "task_focused",
        ]
        mask = [False] * len(data)
        for column in columns:
            mask |= data[column].isna()

        return data[mask]

    # Turns annotations into one_hot style encodings, weighted based on annotators
    # multiple annotators are combined into one label vector
    def _annotations_to_one_hot(self, data: pd.DataFrame) -> pd.DataFrame:
        def one_hot(num_cats, weight):
            def _one_hot(code):
                vec = np.zeros(num_cats)
                if code != -1:
                    vec[code] = weight
                return vec

            return _one_hot

        def combine(group: pd.DataFrame):
            columns = ["domain", "strategy1", "strategy2", "social_exchange",  "task_focused"]

            df = group[columns]

            ret = group.iloc[[0]][["dataset", "document", "row", "speaker", "utterance", "pos_in_doc"]]
            for column in columns:
                num_cats = len(df[column].cat.categories)
                non_nan_in_group = len(df[column]) - df[column].isna().sum()

                if non_nan_in_group > 0:
                    weight = 1.0 / non_nan_in_group
                    labels = df[column].cat.codes.apply(one_hot(num_cats, weight))
                else:
                    labels = None

                if labels is not None:
                    label = np.sum(labels)
                    total_value = np.sum(label)

                    assert total_value == 1.0
                else:
                    label = labels

                ret[column] = [label]

            return ret

        def strategy(row):

            if row["strategy1"] is None:
                s1 = np.zeros(13)
            else:
                s1 = row["strategy1"]

            if row["strategy2"] is None:
                s2 = np.zeros(13)
            else:
                s2 = row["strategy2"]

            strategy = np.add(s1, s2)
            if strategy[0] == 2:  # both strategy 1 and 2 are none
                # print("strategy[0] == 2")
                strategy[0] = 1
            elif strategy[0] == 1 and np.sum(strategy) == 2: # only strategy 2 is none
                # print("strategy[0] == 1")
                strategy[0] = 0
            elif np.sum(strategy) == 0:
                strategy[0] = 1
            # if strategy[0] == 0:
                # print("strategy", strategy)

            return strategy

        data = data.groupby(["document", "row"], as_index=False).apply(combine)
        data["strategy"] = data.apply(strategy, axis=1)

        return data.reset_index(drop=True)

