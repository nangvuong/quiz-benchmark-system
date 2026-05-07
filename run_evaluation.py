import os
import json
import matplotlib.pyplot as plt
import numpy as np
from src.evaluation.benchmark import MCQBenchmark

def run_evaluation_pipeline():
    print("Starting MCQ Evaluation Pipeline...")
    
    # 1. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    
    generated_file = os.path.join(data_dir, "generated_mcqs.json")
    
    # Find SQuAD reference
    reference_file = None
    for file in os.listdir(data_dir):
        if file.startswith("squad_validation_processed"):
            reference_file = os.path.join(data_dir, file)
            break
            
    if not reference_file or not os.path.exists(generated_file):
        print("Required data files not found. Ensure main.py has run successfully.")
        return
        
    # 2. Run Benchmark
    benchmark = MCQBenchmark(reference_file, generated_file)
    results = benchmark.run_evaluation()
    
    # 3. Save JSON Report
    report_file = os.path.join(outputs_dir, "evaluation_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print(f"JSON Report saved to {report_file}")
    
    # 4. Generate Visualizations
    models = list(results.keys())
    
    if not models:
        print("No models to evaluate.")
        return
        
    metrics_to_plot = {
        'BLEU': 'avg_bleu',
        'ROUGE-1': 'avg_rouge1',
        'ROUGE-L': 'avg_rougeL',
        'F1 Score (Answer)': 'avg_f1'
    }
    
    # 4a. Individual Charts (4 metrics + Runtime)
    all_metrics = metrics_to_plot.copy()
    all_metrics['Runtime (seconds)'] = 'total_runtime_seconds'
    
    for metric_name, metric_key in all_metrics.items():
        plt.figure(figsize=(8, 5))
        
        values = [results[m].get(metric_key, 0) for m in models]
        
        # Ensure we have enough colors regardless of model count
        palette = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860']
        colors = palette[:len(models)]
        
        model_names = [m.upper() for m in models]
        bars = plt.bar(model_names, values, color=colors)
        
        plt.ylabel('Score' if 'Runtime' not in metric_name else 'Seconds')
        plt.title(f'Model Comparison: {metric_name}')
        
        # Add value labels
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + (0.01 * max(values) if max(values) > 0 else 0.01), 
                     f'{yval:.4f}', ha='center', va='bottom')
            
        plt.tight_layout()
        filename = metric_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
        chart_file = os.path.join(outputs_dir, f"chart_{filename}.png")
        plt.savefig(chart_file)
        plt.close()
        print(f"Saved individual chart: {chart_file}")
    
    # 4b. Combined Chart (excluding Runtime due to scale differences)
    x = np.arange(len(models))
    num_metrics = len(metrics_to_plot)
    width = 0.8 / num_metrics
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (metric_name, metric_key) in enumerate(metrics_to_plot.items()):
        values = [results[m].get(metric_key, 0) for m in models]
        ax.bar(x + i*width, values, width, label=metric_name)
        
    ax.set_ylabel('Scores')
    ax.set_title('Combined Model Performance Comparison')
    ax.set_xticks(x + width * (num_metrics - 1) / 2)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend()
    
    plt.tight_layout()
    chart_file = os.path.join(outputs_dir, "benchmark_chart_combined.png")
    plt.savefig(chart_file)
    plt.close()
    print(f"Saved combined chart: {chart_file}")

if __name__ == "__main__":
    run_evaluation_pipeline()
