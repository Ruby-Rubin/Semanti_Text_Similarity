import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from datasets import Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sentence_transformers import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments
)

from sentence_transformers.cross_encoder import losses
from sentence_transformers.cross_encoder.evaluation import (
    CrossEncoderCorrelationEvaluator
)


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

DATASET_PATH = "dataset/STS-B"

# NLI-pretrained model
MODEL_NAME = "cross-encoder/nli-distilroberta-base"

BATCH_SIZE = 8
EPOCHS = 2
MAX_LENGTH = 128
LEARNING_RATE = 2e-5

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 2. LOAD STS-B DATASET
# ============================================================

print("\n========== LOADING DATASET ==========")

train = pd.read_csv(
    os.path.join(DATASET_PATH, "train.tsv"),
    sep="\t",
    engine="python",
    on_bad_lines="skip"
)

validation = pd.read_csv(
    os.path.join(DATASET_PATH, "dev.tsv"),
    sep="\t",
    engine="python",
    on_bad_lines="skip"
)

# Keep only required columns
train = train[
    ["sentence1", "sentence2", "score"]
].copy()

validation = validation[
    ["sentence1", "sentence2", "score"]
].copy()


# ============================================================
# 3. DATA CLEANING
# ============================================================

train["sentence1"] = train["sentence1"].fillna("")
train["sentence2"] = train["sentence2"].fillna("")

validation["sentence1"] = validation["sentence1"].fillna("")
validation["sentence2"] = validation["sentence2"].fillna("")

# Convert scores to numeric
train["score"] = pd.to_numeric(
    train["score"],
    errors="coerce"
)

validation["score"] = pd.to_numeric(
    validation["score"],
    errors="coerce"
)

# Remove invalid scores
train = train.dropna(
    subset=["score"]
).reset_index(drop=True)

validation = validation.dropna(
    subset=["score"]
).reset_index(drop=True)

print("Training samples   :", len(train))
print("Validation samples :", len(validation))


# ============================================================
# 4. PREPROCESSING
# ============================================================

print("\n========== PREPROCESSING ==========")

print("\nOriginal sentence:")
print(train.iloc[0]["sentence1"])

print("\nSecond sentence:")
print(train.iloc[0]["sentence2"])

print("\nOriginal similarity score:")
print(train.iloc[0]["score"])

# STS-B scores range from 0 to 5.
# BinaryCrossEntropyLoss uses a 0 to 1 target.
train["label"] = (
    train["score"].astype(float) / 5.0
)

validation["label"] = (
    validation["score"].astype(float) / 5.0
)

print("\nNormalized similarity score:")
print(train.iloc[0]["label"])


# ============================================================
# 5. CREATE HUGGING FACE DATASET
# ============================================================

print("\n========== CREATING TRAINING DATA ==========")

train_dataset = Dataset.from_pandas(
    train[
        ["sentence1", "sentence2", "label"]
    ],
    preserve_index=False
)

print("Training pairs:", len(train_dataset))


# ============================================================
# 6. LOAD NLI-PRETRAINED TRANSFORMER
# ============================================================

print("\n========== LOADING TRANSFORMER ==========")

model = CrossEncoder(
    MODEL_NAME,

    # We need one output score for STS.
    num_labels=1,

    max_length=MAX_LENGTH,

    # STS target is continuous after normalization.
    activation_fn=torch.nn.Sigmoid(),

    # NLI checkpoint originally has a 3-class head.
    # Replace that head with a single-output head.
    model_kwargs={
        "ignore_mismatched_sizes": True
    },

    device="cpu"
)

print("Model:", MODEL_NAME)
print("Task : Semantic Text Similarity")
print("Base : Natural Language Inference")


# ============================================================
# 7. LOSS FUNCTION
# ============================================================

loss = losses.BinaryCrossEntropyLoss(model)


# ============================================================
# 8. VALIDATION EVALUATOR
# ============================================================

validation_pairs = list(
    zip(
        validation["sentence1"].tolist(),
        validation["sentence2"].tolist()
    )
)

validation_scores = (
    validation["label"]
    .astype(float)
    .tolist()
)

evaluator = CrossEncoderCorrelationEvaluator(
    sentence_pairs=validation_pairs,
    scores=validation_scores,
    name="sts-validation",
    batch_size=BATCH_SIZE,
    write_csv=True
)


# ============================================================
# 9. TRAINING ARGUMENTS
# ============================================================

training_args = CrossEncoderTrainingArguments(
    output_dir="semantic_similarity_nli_transformer",

    num_train_epochs=EPOCHS,

    per_device_train_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    warmup_ratio=0.1,

    weight_decay=0.01,

    logging_strategy="steps",

    logging_steps=100,

    eval_strategy="no",

    save_strategy="no",

    report_to="none",

    use_cpu=True,

    seed=SEED
)


# ============================================================
# 10. TRAINER
# ============================================================

trainer = CrossEncoderTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,

    loss=loss
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n========== TRAINING ==========")

trainer.train()

print("\nTraining completed.")


# ============================================================
# 12. SAVE MODEL
# ============================================================

MODEL_OUTPUT = (
    "semantic_similarity_nli_transformer"
)

model.save_pretrained(
    MODEL_OUTPUT
)

print("\nModel saved to:")
print(MODEL_OUTPUT)


# ============================================================
# 13. VALIDATION CORRELATION
# ============================================================

print("\n========== VALIDATION ==========")

results = evaluator(model)

print("Evaluator results:")

for key, value in results.items():
    print(
        f"{key}: {value:.4f}"
    )


# ============================================================
# 14. VALIDATION PREDICTIONS
# ============================================================

print("\n========== VALIDATION PREDICTIONS ==========")

predicted_normalized = model.predict(
    validation_pairs,

    batch_size=BATCH_SIZE,

    show_progress_bar=True,

    activation_fn=torch.nn.Sigmoid()
)

predicted_normalized = np.asarray(
    predicted_normalized,
    dtype=np.float32
)

# Convert 0-1 back to original 0-5 scale
predicted_scores = (
    predicted_normalized * 5.0
)

# Keep predictions within valid STS-B range
predicted_scores = np.clip(
    predicted_scores,
    0.0,
    5.0
)

actual_scores = (
    validation["score"]
    .astype(float)
    .values
)


# ============================================================
# 15. SAMPLE PREDICTIONS
# ============================================================

print("\n========== SAMPLE PREDICTIONS ==========")

for i in range(10):
    print(
        f"Actual: {actual_scores[i]:.2f} | "
        f"Predicted: {predicted_scores[i]:.2f}"
    )


# ============================================================
# 16. MODEL PERFORMANCE
# ============================================================

mae = mean_absolute_error(
    actual_scores,
    predicted_scores
)

mse = mean_squared_error(
    actual_scores,
    predicted_scores
)

pearson = np.corrcoef(
    actual_scores,
    predicted_scores
)[0, 1]

spearman = pd.Series(
    actual_scores
).corr(
    pd.Series(predicted_scores),
    method="spearman"
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"MAE      : {mae:.4f}")
print(f"MSE      : {mse:.4f}")
print(f"Pearson  : {pearson:.4f}")
print(f"Spearman : {spearman:.4f}")


# ============================================================
# 17. ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(
    figsize=(7, 6)
)

plt.scatter(
    actual_scores,
    predicted_scores,
    alpha=0.5
)

plt.plot(
    [0, 5],
    [0, 5],
    linestyle="--"
)

plt.xlabel(
    "Actual Similarity Score"
)

plt.ylabel(
    "Predicted Similarity Score"
)

plt.title(
    "Actual vs Predicted Semantic Similarity"
)

plt.xlim(0, 5)

plt.ylim(0, 5)

plt.grid()

plt.tight_layout()

plt.savefig(
    "nli_transformer_actual_vs_predicted.png",
    dpi=300
)

plt.show()


# ============================================================
# 18. CUSTOM SIMILARITY FUNCTION
# ============================================================

def predict_similarity(
    sentence1,
    sentence2
):

    pair = [
        (sentence1, sentence2)
    ]

    normalized_score = model.predict(
        pair,

        show_progress_bar=False,

        activation_fn=torch.nn.Sigmoid()
    )[0]

    score = float(
        normalized_score * 5.0
    )

    score = float(
        np.clip(
            score,
            0.0,
            5.0
        )
    )

    if score >= 4.0:

        interpretation = "Highly Similar"

    elif score >= 2.5:

        interpretation = "Moderately Similar"

    elif score >= 1.0:

        interpretation = "Slightly Similar"

    else:

        interpretation = "Low Similarity"


    print(
        "\n========== CUSTOM TEST =========="
    )

    print(
        "Sentence 1:",
        sentence1
    )

    print(
        "Sentence 2:",
        sentence2
    )

    print(
        f"Similarity Score: {score:.2f} / 5"
    )

    print(
        "Interpretation:",
        interpretation
    )

    return score


# ============================================================
# 19. CUSTOM TESTS
# ============================================================

predict_similarity(
    "How can I reset my password?",
    "I forgot my password. How can I change it?"
)

predict_similarity(
    "A man is playing a guitar.",
    "A person is playing an instrument."
)

predict_similarity(
    "The weather is beautiful today.",
    "The sky is clear and sunny."
)
