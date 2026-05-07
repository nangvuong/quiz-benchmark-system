from rouge_score import rouge_scorer

class RougeEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate(self, generated_text: str, reference_texts: list[str]) -> dict:
        """Calculate ROUGE scores, returning the max score across all references."""
        if not generated_text or not reference_texts:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            
        best_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        for ref in reference_texts:
            scores = self.scorer.score(ref, generated_text)
            best_scores["rouge1"] = max(best_scores["rouge1"], scores["rouge1"].fmeasure)
            best_scores["rouge2"] = max(best_scores["rouge2"], scores["rouge2"].fmeasure)
            best_scores["rougeL"] = max(best_scores["rougeL"], scores["rougeL"].fmeasure)
            
        return best_scores
