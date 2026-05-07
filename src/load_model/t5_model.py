
"""
T5-Based MCQ Generator.
Uses pre-trained T5 model for question generation and answer extraction.
"""

from typing import List, Dict, Optional
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import warnings

warnings.filterwarnings('ignore')


class T5MCQGenerator:
    """Generate MCQs using pre-trained T5 model."""
    
    def __init__(
        self, 
        model_name: str = "valhalla/t5-base-qg-hl",
        device: str = "cpu",
        num_questions: int = 5,
        num_options: int = 4,
        max_length: int = 512,
        num_beams: int = 4
    ):
        """
        Initialize T5 MCQ generator.
        
        Args:
            model_name: Pre-trained T5 model name
            device: Device to run model on ('cpu' or 'cuda')
            num_questions: Number of questions to generate
            num_options: Number of options per question
            max_length: Maximum sequence length
            num_beams: Number of beams for beam search
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.num_questions = num_questions
        self.num_options = num_options
        self.max_length = max_length
        self.num_beams = num_beams
        
        import os
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        
        print(f"Loading {model_name} (saving to {cache_dir})...")
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")
    
    def generate_questions(self, context: str, answer_phrases: List[str] = None) -> List[Dict]:
        """
        Generate MCQs using T5.
        
        Args:
            context: Source text
            answer_phrases: Specific phrases to generate questions for (optional)
            
        Returns:
            List of MCQ dictionaries
        """
        # Extract answer phrases if not provided
        if answer_phrases is None:
            answer_phrases = self._extract_answer_phrases(context)
        
        # Generate questions for each answer phrase
        mcqs = []
        for answer_phrase in answer_phrases[:self.num_questions]:
            question = self._generate_question(context, answer_phrase)
            
            if question:
                # Generate distractor options
                distractors = self._generate_distractors(context, answer_phrase)
                
                # Create MCQ
                mcq = {
                    'question': question,
                    'correct_answer': answer_phrase,
                    'options': self._create_options(answer_phrase, distractors),
                    'context': context,
                    'type': 'multiple_choice',
                    'model': self.model_name
                }
                
                # Set correct index
                mcq['correct_index'] = mcq['options'].index(answer_phrase)
                mcqs.append(mcq)
        
        return mcqs
    
    def _extract_answer_phrases(self, context: str) -> List[str]:
        """
        Extract candidate answer phrases using T5.
        
        Args:
            context: Source text
            
        Returns:
            List of answer phrases
        """
        # Even for finetuned models like valhalla, they usually don't support 'extract:' 
        # for answer extraction. It's safer and better to use NLTK to get good nouns.
        input_text = f"summarize: {context}"
        
        try:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            input_ids = input_ids.to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_length=100,
                    num_beams=self.num_beams,
                    num_return_sequences=1
                )
            
            # Extract nouns from the generated summary
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                import ssl
                try:
                    _create_unverified_https_context = ssl._create_unverified_context
                except AttributeError:
                    pass
                else:
                    ssl._create_default_https_context = _create_unverified_https_context
                nltk.download('averaged_perceptron_tagger', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
            
            tokens = word_tokenize(summary)
            pos_tags = pos_tag(tokens)
            
            # Nouns > 3 chars
            phrases = [word for word, pos in pos_tags if pos in ['NN', 'NNP', 'NNPS'] and len(word) > 3]
            
            # Deduplicate while preserving order
            unique_phrases = []
            for p in phrases:
                if p.lower() not in [u.lower() for u in unique_phrases]:
                    unique_phrases.append(p)
                    
            # Backfill from context if not enough
            if len(unique_phrases) < self.num_questions:
                context_tokens = word_tokenize(context)
                context_pos = pos_tag(context_tokens)
                context_phrases = [word for word, pos in context_pos if pos in ['NN', 'NNP', 'NNPS'] and len(word) > 3]
                for p in context_phrases:
                    if p.lower() not in [u.lower() for u in unique_phrases]:
                        unique_phrases.append(p)
                        if len(unique_phrases) >= self.num_questions:
                            break
                            
            return unique_phrases[:self.num_questions]
                
        except Exception as e:
            print(f"Error in T5 extraction, falling back: {e}")
            # Fallback: extract nouns using simple method
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk import pos_tag
            
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                pass
            
            tokens = word_tokenize(context)
            pos_tags = pos_tag(tokens)
            
            phrases = [word for word, pos in pos_tags if pos in ['NN', 'NNP', 'NNPS'] and len(word) > 3]
            
            unique_phrases = []
            for p in phrases:
                if p.lower() not in [u.lower() for u in unique_phrases]:
                    unique_phrases.append(p)
            return unique_phrases[:self.num_questions]

    def _generate_question(self, context: str, answer_phrase: str) -> Optional[str]:
        """
        Generate a question for given answer using T5.
        
        Args:
            context: Source text
            answer_phrase: Answer phrase
            
        Returns:
            Generated question or None
        """
        try:
            is_finetuned = "finetuned" in self.model_name.lower() or "qg" in self.model_name.lower()
            is_valhalla = "valhalla" in self.model_name.lower()
            
            # Create input for T5
            if is_valhalla:
                # Valhalla models require <hl> tags around the answer
                import re
                highlighted_context = re.sub(
                    f"(?i)\\b({re.escape(answer_phrase)})\\b", 
                    r"<hl> \1 <hl>", 
                    context, 
                    count=1
                )
                if "<hl>" not in highlighted_context:
                    highlighted_context = context.replace(answer_phrase, f"<hl> {answer_phrase} <hl>", 1)
                input_text = f"generate question: {highlighted_context}"
            elif is_finetuned:
                input_text = f"answer: {answer_phrase} context: {context} </s>"
            else:
                input_text = f"generate question: {context} </s> {answer_phrase}"
            
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
            input_ids = input_ids.to(self.device)
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_length=100,
                    num_beams=self.num_beams,
                    early_stopping=True
                )
            
            question = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Clean up question
            import re
            question = re.sub(r'^(question|extract):\s*', '', question, flags=re.IGNORECASE).strip()
            
            # Check if the generated question is poor (too short or just repeats answer)
            is_poor = len(question.split()) < 4 or question.lower().replace('?', '').strip() == answer_phrase.lower()
            
            if is_poor:
                import nltk
                try:
                    from nltk.tokenize import sent_tokenize
                    sentences = sent_tokenize(context)
                except:
                    sentences = context.split('.')
                
                target_sent = ""
                for sent in sentences:
                    if answer_phrase.lower() in sent.lower():
                        target_sent = sent
                        break
                
                if target_sent:
                    # Create fill-in-the-blank question
                    pattern = re.compile(r'\b' + re.escape(answer_phrase) + r'\b', re.IGNORECASE)
                    question = pattern.sub("_____", target_sent.strip())
                    if "_____" not in question:
                        pattern = re.compile(re.escape(answer_phrase), re.IGNORECASE)
                        question = pattern.sub("_____", target_sent.strip())
                else:
                    question = f"Which of the following best describes '{answer_phrase}' in the context?"
            
            if not question.endswith('?') and "_____" not in question:
                question += '?'
            
            return question
        except Exception as e:
            print(f"Error generating question: {e}")
            # Fallback template question
            return f"Which of the following best describes '{answer_phrase}' in the context?"
    
    def _generate_distractors(self, context: str, answer_phrase: str) -> List[str]:
        """
        Generate distractor options using T5.
        
        Args:
            context: Source text
            answer_phrase: Correct answer phrase
            
        Returns:
            List of distractor phrases
        """
        try:
            distractors = []
            
            # QG models generally don't support 'generate alternatives:', so we always use the noun extraction fallback
            if len(distractors) < self.num_options - 1:
                import nltk
                from nltk.tokenize import word_tokenize
                from nltk import pos_tag
                
                tokens = word_tokenize(context)
                pos_tags = pos_tag(tokens)
                
                # Extract other nouns as distractors
                candidate_nouns = [word for word, pos in pos_tags if pos in ['NN', 'NNP', 'NNPS'] and len(word) > 3]
                candidate_nouns = [n for n in candidate_nouns if n.lower() != answer_phrase.lower()]
                
                # Deduplicate
                unique_nouns = []
                for n in candidate_nouns:
                    if n.lower() not in [u.lower() for u in unique_nouns]:
                        unique_nouns.append(n)
                
                import random
                random.shuffle(unique_nouns)
                
                for noun in unique_nouns:
                    if noun not in distractors:
                        distractors.append(noun)
                    if len(distractors) >= self.num_options - 1:
                        break
                        
                # Final fallback if context has no other nouns
                if len(distractors) < self.num_options - 1:
                    words = context.split()
                    random.shuffle(words)
                    fallback = [' '.join(words[i:i+2]) for i in range(0, len(words)-1, 3)]
                    for fb in fallback:
                        if fb.lower() != answer_phrase.lower() and fb not in distractors:
                            distractors.append(fb)
                        if len(distractors) >= self.num_options - 1:
                            break
                            
            return distractors[:self.num_options - 1]
        except Exception as e:
            print(f"Error generating distractors: {e}")
            # Fallback: simple distractor generation
            words = context.split()
            import random
            random.shuffle(words)
            distractors = [' '.join(words[i:i+2]) for i in range(0, len(words)-1, 3)]
            return distractors[:self.num_options - 1]
    
    def _create_options(self, correct_answer: str, distractors: List[str]) -> List[str]:
        """
        Create option list with shuffle.
        
        Args:
            correct_answer: Correct answer
            distractors: List of distractors
            
        Returns:
            Shuffled list of options
        """
        options = [correct_answer] + distractors
        
        import random
        random.shuffle(options)
        
        return options
    
    def batch_generate_questions(self, contexts: List[str]) -> List[Dict]:
        """
        Generate questions for multiple contexts.
        
        Args:
            contexts: List of context texts
            
        Returns:
            List of all generated MCQs
        """
        all_mcqs = []
        
        for i, context in enumerate(contexts):
            print(f"Processing context {i+1}/{len(contexts)}...")
            mcqs = self.generate_questions(context)
            all_mcqs.extend(mcqs)
        
        return all_mcqs


if __name__ == "__main__":
    # Example usage (requires T5 model - will download automatically)
    try:
        generator = T5MCQGenerator(num_questions=3, device="cpu")
        
        sample_context = """
        The internet is a global system of interconnected computer networks that uses the Internet 
        protocol suite (TCP/IP) to communicate between networks and devices. It is a network of networks 
        that consists of private, public, academic, business, and government networks of local to global scope.
        """
        
        mcqs = generator.generate_questions(sample_context)
        
        for i, mcq in enumerate(mcqs, 1):
            print(f"\nQuestion {i}: {mcq['question']}")
            print(f"Options:")
            for j, option in enumerate(mcq['options'], 1):
                marker = "✓" if option == mcq['correct_answer'] else " "
                print(f"  {marker} {j}. {option}")
    except Exception as e:
        print(f"T5 model not available: {e}")
        print("The T5MCQGenerator will download the model on first use.")
