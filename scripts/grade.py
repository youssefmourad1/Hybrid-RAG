import pandas as pd
import os
import glob
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# NOTE: Code refactored based on : 
# https://github.com/vibrantlabsai/ragas/issues/2351 
# The error was that the rate limits for batch processing on openai requests is restricted. for our api usage
# WE HAD TO USE THE LLM_FACTORY PATTERN FROM RAGAS TO AVOID THE RATE LIMITS

def evaluate_ragas(results_file):
    df = pd.read_csv(results_file)
    # Sample 10% for faster grading (WE ARE BROKE KINDOF SLICING CODE :) KUDOS IF YOU READ THIS)
    df = df.sample(frac=0.1, random_state=42)
    
    # Ragas expects: question, answer (generated), contexts (list of strings), ground_truth
    # Rename columns to match Ragas requirement
    ragas_df = df.rename(columns={
        "generated_answer": "answer",
        "context": "contexts",
        "ground_truth": "ground_truth"
    })
    
    # Convert string representation of list back to list
    ragas_df['contexts'] = ragas_df['contexts'].apply(eval)
    ragas_df['ground_truth'] = ragas_df['ground_truth'].apply(lambda x: x if isinstance(x, str) else str(x))

    dataset = Dataset.from_pandas(ragas_df)
    
    print(f"Running Ragas evaluation on {results_file}...")
    
    # Ragas v0.2 Migration: Use llm_factory
    from ragas.llms import llm_factory
    from langchain_openai import OpenAIEmbeddings # Ensure we have embeddings

    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for grading.")

    # Initialize Client
    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Use factory pattern with client
    llm = llm_factory(
        provider="openai",
        model="gpt-4o-mini",
        client=client
    )
    # Reuse langchain embeddings or potentially ragas factory if available, sticking to langchain for now as user snippet didn't change embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Adjust strictness
    answer_relevancy.strictness = 1
    
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    return results

def main():
    results_files = glob.glob("outputs/*_results.csv")
    if not results_files:
        print("No result files found in outputs/")
        return

    all_scores = []

    for file in results_files:
        run_id = os.path.basename(file).replace("_results.csv", "")
        ragas_scores = evaluate_ragas(file)
        
        scores = {
            "Run ID": run_id,
            "Faithfulness": np.mean(ragas_scores["faithfulness"]),
            "Answer Relevancy": np.mean(ragas_scores["answer_relevancy"])
        }
        
        scores["Tokens/Sec"] = 15.0 # TODO: Add actual tokens/sec
        
        all_scores.append(scores)

    df_scores = pd.DataFrame(all_scores)
    print("\nFinal Leaderboard:")
    print(df_scores.to_markdown(index=False))
    
    # Save Report
    with open("outputs/report.md", "w") as f:
        f.write("# Benchmark Report\n\n")
        f.write(df_scores.to_markdown(index=False))

    # Plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_scores, x="Tokens/Sec", y="Answer Relevancy", hue="Run ID", s=100)
    plt.title("Accuracy vs Speed")
    plt.savefig("outputs/tradeoff_plot.png")
    print("Plot saved to outputs/tradeoff_plot.png")

if __name__ == "__main__":
    main()
