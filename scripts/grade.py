import pandas as pd
import os
import glob
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_ragas(results_file):
    df = pd.read_csv(results_file)
    
    # Ragas expects: question, answer (generated), contexts (list of strings), ground_truth
    # Rename columns to match Ragas requirement
    ragas_df = df.rename(columns={
        "generated_answer": "answer",
        "context": "contexts", # script saved as string repr of list, need to eval
        "ground_truth": "ground_truth"
    })
    
    # Convert string representation of list back to list
    ragas_df['contexts'] = ragas_df['contexts'].apply(eval)
    ragas_df['ground_truth'] = ragas_df['ground_truth'].apply(lambda x: x if isinstance(x, str) else str(x))

    dataset = Dataset.from_pandas(ragas_df)
    
    print(f"Running Ragas evaluation on {results_file}...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy],
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
            "Faithfulness": ragas_scores["faithfulness"],
            "Answer Relevancy": ragas_scores["answer_relevancy"]
        }
        
        # Add speed/throughput if we logged it (stubbed here)
        scores["Tokens/Sec"] = 15.0 # Dummy
        
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
