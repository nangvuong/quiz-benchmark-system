import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

class MCQMetrics:
    """Class to calculate various evaluation metrics for Multiple Choice Questions."""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
    def calculate_bleu(self, generated_question: str, reference_questions: list[str]) -> float:
        """
        Calculate BLEU score for a generated question against reference questions.
        """
        if not generated_question or not reference_questions:
            return 0.0
            
        gen_tokens = nltk.word_tokenize(generated_question.lower())
        ref_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in reference_questions]
        
        try:
            score = sentence_bleu(ref_tokens_list, gen_tokens, smoothing_function=self.smoothing)
            return score
        except Exception:
            return 0.0
            
    def calculate_rouge(self, generated_question: str, reference_questions: list[str]) -> dict:
        """
        Calculate ROUGE scores, returning the max score across all references.
        """
        if not generated_question or not reference_questions:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
        best_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        for ref in reference_questions:
            scores = self.rouge_scorer.score(ref, generated_question)
            
            best_scores["rouge1"] = max(best_scores["rouge1"], scores["rouge1"].fmeasure)
            best_scores["rouge2"] = max(best_scores["rouge2"], scores["rouge2"].fmeasure)
            best_scores["rougeL"] = max(best_scores["rougeL"], scores["rougeL"].fmeasure)
            
        return best_scores
        
    def evaluate_distractors(self, correct_answer: str, options: list[str]) -> dict:
        """
        Evaluate quality of distractors.
        A good distractor should not be identical to the correct answer and shouldn't be empty.
        """
        distractors = [opt for opt in options if opt != correct_answer]
        
        if not distractors:
            return {"distractor_count": 0, "unique_distractors": 0}
            
        unique_distractors = set([d.lower().strip() for d in distractors])
        
        return {
            "distractor_count": len(distractors),
            "unique_distractors": len(unique_distractors),
            "diversity_ratio": len(unique_distractors) / len(distractors) if len(distractors) > 0 else 0
        }
