import string
import collections

class AccuracyEvaluator:
    def normalize_answer(self, s: str) -> str:
        """Lower text and remove punctuation, articles and extra whitespace."""
        if not s: return ""
        def remove_articles(text):
            return " ".join(b for b in text.split() if b not in ("a", "an", "the"))
            
        def white_space_fix(text):
            return " ".join(text.split())
            
        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)
            
        return white_space_fix(remove_articles(remove_punc(str(s).lower())))

    def f1_score(self, prediction: str, ground_truth: str) -> float:
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction: str, ground_truth: str) -> float:
        return 1.0 if self.normalize_answer(prediction) == self.normalize_answer(ground_truth) else 0.0

    def calculate(self, generated_answer: str, reference_answers: list[str]) -> dict:
        """Calculate max EM and F1 over all reference answers."""
        if not generated_answer or not reference_answers:
            return {"exact_match": 0.0, "f1": 0.0}
            
        em_scores = [self.exact_match_score(generated_answer, ref) for ref in reference_answers]
        f1_scores = [self.f1_score(generated_answer, ref) for ref in reference_answers]
        
        return {
            "exact_match": max(em_scores),
            "f1": max(f1_scores)
        }
