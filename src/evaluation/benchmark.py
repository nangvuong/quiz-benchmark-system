import os
import json
import numpy as np
from collections import defaultdict
from src.evaluation.bleu import BleuEvaluator
from src.evaluation.rouge import RougeEvaluator
from src.evaluation.runtime import RuntimeTracker
from src.evaluation.accuracy import AccuracyEvaluator

class MCQBenchmark:
    def __init__(self, reference_path: str, generated_path: str):
        self.reference_path = reference_path
        self.generated_path = generated_path
        
        self.bleu_eval = BleuEvaluator()
        self.rouge_eval = RougeEvaluator()
        self.accuracy_eval = AccuracyEvaluator()
        self.runtime_tracker = RuntimeTracker()
        
    def load_data(self):
        with open(self.reference_path, 'r', encoding='utf-8') as f:
            self.reference_data = json.load(f)
            
        with open(self.generated_path, 'r', encoding='utf-8') as f:
            self.generated_data = json.load(f)
            
        self.context_to_references = defaultdict(list)
        self.context_to_answers = defaultdict(list)
        for item in self.reference_data:
            ctx = item.get('context', '').strip()
            q = item.get('question', '').strip()
            ans = item.get('answers', [])
            if ctx and q:
                self.context_to_references[ctx].append(q)
            if ctx and ans:
                self.context_to_answers[ctx].extend(ans)

    def run_evaluation(self) -> dict:
        self.load_data()
        model_scores = defaultdict(lambda: defaultdict(list))
        
        # In a full benchmark, generation time would be tracked here. 
        # Since we are reading pre-generated JSON, runtime metrics might be mocked or we can evaluate the evaluation runtime.
        # Ideally, main.py should output runtime into the JSON. We'll track evaluation speed here just to use RuntimeTracker.
        
        for item in self.generated_data:
            context = item['context'].strip()
            reference_questions = self.context_to_references.get(context, [])
            reference_answers = self.context_to_answers.get(context, [])
            has_refs = len(reference_questions) > 0
            
            models_output = item.get('questions', {})
            for model_name, mcqs in models_output.items():
                if not mcqs:
                    continue
                    
                for mcq in mcqs:
                    gen_q = mcq.get('question', '')
                    options = mcq.get('options', [])
                    correct = mcq.get('correct_answer', '')
                    
                    if has_refs:
                        with self.runtime_tracker.measure(f"{model_name}_accuracy_eval"):
                            acc = self.accuracy_eval.calculate(correct, reference_answers)
                            model_scores[model_name]['exact_match'].append(acc['exact_match'])
                            model_scores[model_name]['f1'].append(acc['f1'])
                            
                        with self.runtime_tracker.measure(f"{model_name}_bleu_eval"):
                            bleu = self.bleu_eval.calculate(gen_q, reference_questions)
                            model_scores[model_name]['bleu'].append(bleu)
                            
                        with self.runtime_tracker.measure(f"{model_name}_rouge_eval"):
                            rouge = self.rouge_eval.calculate(gen_q, reference_questions)
                            model_scores[model_name]['rouge1'].append(rouge['rouge1'])
                            model_scores[model_name]['rouge2'].append(rouge['rouge2'])
                            model_scores[model_name]['rougeL'].append(rouge['rougeL'])

        # Aggregate averages
        aggregated_results = {}
        for model_name, metrics in model_scores.items():
            total_time = 0.0
            for suffix in ["_bleu_eval", "_rouge_eval", "_accuracy_eval"]:
                total_time += sum(self.runtime_tracker.records.get(f"{model_name}{suffix}", []))
                
            aggregated_results[model_name] = {
                'avg_bleu': float(np.mean(metrics['bleu'])) if metrics['bleu'] else 0.0,
                'avg_rouge1': float(np.mean(metrics['rouge1'])) if metrics['rouge1'] else 0.0,
                'avg_rouge2': float(np.mean(metrics['rouge2'])) if metrics['rouge2'] else 0.0,
                'avg_rougeL': float(np.mean(metrics['rougeL'])) if metrics['rougeL'] else 0.0,
                'avg_exact_match': float(np.mean(metrics['exact_match'])) if metrics['exact_match'] else 0.0,
                'avg_f1': float(np.mean(metrics['f1'])) if metrics['f1'] else 0.0,
                'total_runtime_seconds': float(total_time),
                'total_evaluated': len(metrics['bleu']) if metrics['bleu'] else 0
            }
            
        return aggregated_results
