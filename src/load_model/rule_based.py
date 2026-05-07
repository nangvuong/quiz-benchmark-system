"""
Rule-Based MCQ Generator.
Generates multiple-choice questions using predefined rules and heuristics.
"""

import re
from typing import List, Dict, Tuple
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords

# Download required data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class RuleBasedMCQGenerator:
    """Generate MCQs using rule-based approach."""
    
    def __init__(self, num_options: int = 4, num_questions: int = 5):
        """
        Initialize rule-based MCQ generator.
        
        Args:
            num_options: Number of options per question (including correct answer)
            num_questions: Number of questions to generate
        """
        self.num_options = num_options
        self.num_questions = num_questions
        self.stop_words = set(stopwords.words('english'))
    
    def generate_questions(self, context: str) -> List[Dict]:
        """
        Generate MCQs from context using rule-based approach.
        
        Args:
            context: Source text
            
        Returns:
            List of MCQ dictionaries
        """
        sentences = sent_tokenize(context)
        
        # Extract candidate questions and answers
        qa_pairs = self._extract_qa_pairs(sentences, context)
        
        # Generate MCQs
        mcqs = []
        for qa_pair in qa_pairs[:self.num_questions]:
            mcq = self._create_mcq(qa_pair, context)
            if mcq:
                mcqs.append(mcq)
        
        return mcqs
    
    def _extract_qa_pairs(
        self, 
        sentences: List[str], 
        context: str
    ) -> List[Dict]:
        """
        Extract potential Q&A pairs from sentences.
        
        Args:
            sentences: List of sentences
            context: Full context text
            
        Returns:
            List of Q&A pair candidates
        """
        qa_pairs = []
        
        for sentence in sentences:
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Extract entities and key terms
            entities = self._extract_entities(tokens, pos_tags)
            
            if not entities:
                continue
            
            for entity in entities:
                # Create question templates
                question = self._generate_question(sentence, entity)
                if question:
                    qa_pairs.append({
                        'sentence': sentence,
                        'entity': entity,
                        'answer': entity,
                        'question': question,
                        'context_snippet': context
                    })
        
        return qa_pairs
    
    def _extract_entities(self, tokens: List[str], pos_tags: List[Tuple]) -> List[str]:
        """
        Extract named entities and key terms.
        
        Args:
            tokens: List of tokens
            pos_tags: POS tags for tokens
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for i, (word, pos) in enumerate(pos_tags):
            # Extract proper nouns, numbers, and important nouns
            if pos in ['NNP', 'NNPS']:  # Proper nouns
                entities.append(word)
            elif pos in ['NN', 'NNS'] and len(word) > 3:  # Common nouns
                # Only if not a stopword
                if word.lower() not in self.stop_words:
                    entities.append(word)
            elif pos == 'CD':  # Numbers
                entities.append(word)
        
        return entities
    
    def _generate_question(self, sentence: str, entity: str) -> str:
        """
        Generate question from sentence and entity.
        
        Args:
            sentence: Original sentence
            entity: Answer entity
            
        Returns:
            Generated question
        """
        # Remove the entity from sentence to create question
        question_text = sentence.replace(entity, '[blank]', 1)
        
        # Simple question templates
        question = f"What is the meaning of '{entity}' in the context: {question_text}"
        
        # Alternative: convert to "Fill in the blank"
        if len(question_text) < 100:
            question = f"Fill in the blank: {question_text.replace('[blank]', '______')}"
        
        return question
    
    def _create_mcq(self, qa_pair: Dict, context: str) -> Dict:
        """
        Create multiple-choice question with options.
        
        Args:
            qa_pair: Q&A pair dictionary
            context: Full context text
            
        Returns:
            MCQ dictionary with options
        """
        answer = qa_pair['answer']
        question = qa_pair['question']
        
        # Generate distractors (wrong options)
        distractors = self._generate_distractors(answer, context)
        
        if len(distractors) < self.num_options - 1:
            return None
        
        # Create options
        options = [answer] + distractors[:self.num_options - 1]
        
        # Shuffle options
        import random
        random.shuffle(options)
        correct_index = options.index(answer)
        
        return {
            'question': question,
            'options': options,
            'correct_answer': answer,
            'correct_index': correct_index,
            'context': context,
            'type': 'multiple_choice'
        }
    
    def _generate_distractors(self, answer: str, context: str) -> List[str]:
        """
        Generate distractor options (wrong answers).
        
        Args:
            answer: Correct answer
            context: Context text
            
        Returns:
            List of distractor options
        """
        distractors = []
        
        # Strategy 1: Extract other entities from context
        tokens = word_tokenize(context)
        pos_tags = pos_tag(tokens)
        
        for word, pos in pos_tags:
            if word != answer and pos in ['NNP', 'NNPS', 'NN', 'NNS']:
                if word not in distractors and len(word) > 2:
                    distractors.append(word)
                    if len(distractors) >= self.num_options - 1:
                        break
        
        # Strategy 2: Use synonyms if available
        if len(distractors) < self.num_options - 1:
            synonyms = self._get_synonyms(answer)
            for syn in synonyms:
                if syn not in distractors:
                    distractors.append(syn)
                    if len(distractors) >= self.num_options - 1:
                        break
        
        return distractors[:self.num_options - 1]
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Args:
            word: Input word
            
        Returns:
            List of synonyms
        """
        synonyms = set()
        
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace('_', ' '))
        except:
            pass
        
        # Remove the original word
        synonyms.discard(word)
        
        return list(synonyms)


if __name__ == "__main__":
    # Example usage
    generator = RuleBasedMCQGenerator(num_options=4, num_questions=5)
    
    sample_context = """
    Artificial Intelligence (AI) is the simulation of human intelligence by machines, 
    especially by computer systems. AI has applications in healthcare, finance, and education. 
    Machine Learning is a subset of AI that enables systems to learn and improve from experience.
    Deep Learning uses neural networks with multiple layers to process information.
    """
    
    mcqs = generator.generate_questions(sample_context)
    
    for i, mcq in enumerate(mcqs, 1):
        print(f"\nQuestion {i}: {mcq['question']}")
        print(f"Options:")
        for j, option in enumerate(mcq['options'], 1):
            print(f"  {j}. {option}")
        print(f"Correct Answer: {mcq['correct_answer']}")
