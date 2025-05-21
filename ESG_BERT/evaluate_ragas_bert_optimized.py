import os
import ast
import pandas as pd
import certifi
import ssl
from config_bert import OPENAI_API_KEY, COMPLETION_MODEL

# Fix Windows SSL
ssl._create_default_https_context = ssl._create_default_https_context

from ragas.evaluation import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from ragas.llms import LlamaIndexLLMWrapper
from ragas.dataset_schema import EvaluationDataset
from llama_index.llms.openai import OpenAI

# Set API key
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load evaluation dataset
df = pd.read_csv("rag_evaluation_dataset_optimized.csv", sep=";")
print("📄 Loaded columns:", df.columns.tolist())

# Parse retrieved_contexts list
df["retrieved_contexts"] = df["retrieved_contexts"].apply(ast.literal_eval)

# RAGAS expects specific column names
df = df.rename(columns={
    "user_input": "user_input",
    "response": "response",
    "retrieved_contexts": "retrieved_contexts",
    "reference": "reference"
})

# Build evaluation set
dataset = EvaluationDataset.from_pandas(df)

# Set up OpenAI LLM
llm = LlamaIndexLLMWrapper(OpenAI(model=COMPLETION_MODEL))

# Metrics
metrics = [
    Faithfulness(llm=llm),
    AnswerRelevancy(llm=llm),
    ContextPrecision(llm=llm),
    ContextRecall(llm=llm),
]

# Evaluate
print("📊 Evaluating RAG pipeline with RAGAS...")
results = evaluate(dataset=dataset, metrics=metrics)

# Convert to DataFrame
df_results = results.to_pandas()

# Compute global averages
averages = df_results[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]].mean()
averages.name = "Global Average"
df_results = pd.concat([df_results, averages.to_frame().T], ignore_index=True)

# Save to CSV
df_results.to_csv("ragas_results_bert_optimized.csv", sep=";", index=False)
print("✅ Evaluation results saved to ragas_results_bert_optimized.csv")

# Print global metrics
print("\n📈 Global Averages:")
print(averages.round(3))
