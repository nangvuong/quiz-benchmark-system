import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class BleuEvaluator:
    def __init__(self):
        self.smoothing = SmoothingFunction().method1

    def calculate(self, generated_text: str, reference_texts: list[str]) -> float:
        """Calculate BLEU score for a generated text against reference texts."""
        if not generated_text or not reference_texts:
            return 0.0
            
        gen_tokens = nltk.word_tokenize(generated_text.lower())
        ref_tokens_list = [nltk.word_tokenize(ref.lower()) for ref in reference_texts]
        
        try:
            return sentence_bleu(ref_tokens_list, gen_tokens, smoothing_function=self.smoothing)
        except Exception:
            return 0.0
