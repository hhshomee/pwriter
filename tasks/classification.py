
import os
import os
os.environ["WANDB_DISABLED"] = "true"
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import torch

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"
CSV = "results/GPT4/A61_GPT4_abs.csv"
LOG_PATH = "results/classification/A61_GPT4.txt"
SPLIT_ID_PATH = "tasks/ids/A61_ids.npz"

TEST_SIZE = 0.2
RANDOM_STATE = 29


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["abstract", "generated_abstract", "cpc_subclass"])
    df = df.reset_index(drop=True)
    return df


def get_ids(df):
    
    split = np.load(SPLIT_ID_PATH, allow_pickle=True)
    train_ids = split["train_ids"].astype(int)
    test_ids = split["test_ids"].astype(int)


    df_train = df[df["patent_id"].isin(train_ids)].reset_index(drop=True)
    df_test = df[df["patent_id"].isin(test_ids)].reset_index(drop=True)

    return df_train, df_test


def prepare_dataset(df, label_encoder, text_column):
    labels = label_encoder.transform(df["cpc_subclass"])
    dataset = Dataset.from_dict({
        "text": df[text_column].tolist(),
        "label": labels
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset = dataset.map(lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=256), batched=True)
    return dataset


def classification(csv_path, model_name):
    print(f"Loading tokenizer and model: {model_name}")
    
    df = load_data(csv_path)

    df_train, df_test = get_ids(df)


    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["cpc_subclass"])

    for col in ["patent_abstract", "generated_abstract"]:
        tag = "Abstract" if col == "patent_abstract" else "GenAbs"
        print(f"\n--- Finetuning on {tag} ---")

        dataset_train = prepare_dataset(df_train, label_encoder, col)
        dataset_test = prepare_dataset(df_test, label_encoder, col)


        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

        args = TrainingArguments(
            output_dir=f"results/checkpoints/{tag.lower()}",
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            learning_rate=2e-5,
            num_train_epochs=4,
            evaluation_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            disable_tqdm=True
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            tokenizer=AutoTokenizer.from_pretrained(model_name),
        )

        trainer.train()

        
        predictions = trainer.predict(dataset_test)
        logits = predictions.predictions
        y_true = predictions.label_ids
        y_pred = np.argmax(logits, axis=-1)

        acc = accuracy_score(y_true, y_pred)
        report_dict = classification_report(y_true, y_pred, target_names=label_encoder.classes_, output_dict=True)

        
        header = f"\n====== Fine-tuned Classification Report for {tag} ======\n"
        header += f"Accuracy: {acc:.4f}\n\n"

        class_metrics = "Per class metrics:\n"
        for cls in label_encoder.classes_:
            m = report_dict[cls]
            class_metrics += f"{cls:<10s}  Precision: {m['precision']:.4f}  Recall: {m['recall']:.4f}  F1-score: {m['f1-score']:.4f}  Support: {int(m['support'])}\n"

        macro = report_dict["macro avg"]
        weighted = report_dict["weighted avg"]
        overall_metrics = "\nTotal (averaged) metrics:\n"
        overall_metrics += f"Macro Avg     Precision: {macro['precision']:.4f}  Recall: {macro['recall']:.4f}  F1-score: {macro['f1-score']:.4f}\n"
        overall_metrics += f"Weighted Avg  Precision: {weighted['precision']:.4f}  Recall: {weighted['recall']:.4f}  F1-score: {weighted['f1-score']:.4f}\n"
        overall_metrics += f"Micro Accuracy: {acc:.4f}\n"

        log_text = header + class_metrics + overall_metrics

        print(log_text)
        with open(LOG_PATH, "a") as f:
            f.write(log_text + "\n")
        


if __name__ == "__main__":
    classification(CSV, MODEL_NAME)





