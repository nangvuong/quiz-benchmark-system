"""
TF-IDF Based MCQ Generator.
Generates questions using TF-IDF to identify important terms and concepts.
"""

from typing import List, Dict, Set
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class TFIDFMCQGenerator:
    """Generate MCQs using TF-IDF based approach."""
    
    def __init__(self, num_options: int = 4, num_questions: int = 5, min_term_length: int = 4):
        """
        Initialize TF-IDF MCQ generator.
        
        Args:
            num_options: Number of options per question
            num_questions: Number of questions to generate
            min_term_length: Minimum length for important terms
        """
        self.num_options = num_options
        self.num_questions = num_questions
        self.min_term_length = min_term_length
        self.vectorizer = TfidfVectorizer(
            max_features=100,
            ngram_range=(1, 2),
            min_df=1,
            stop_words='english'
        )
        self.stop_words = set(stopwords.words('english'))
    
    def generate_questions(self, context: str, document_set: List[str] = None) -> List[Dict]:
        """
        Generate MCQs using TF-IDF.
        
        Args:
            context: Source text
            document_set: Set of documents for IDF calculation (for better TF-IDF scores)
            
        Returns:
            List of MCQ dictionaries
        """
        # Use context as part of document set if not provided
        if document_set is None:
            document_set = [context]
        else:
            document_set = [context] + document_set
        
        # Calculate TF-IDF
        tfidf_matrix = self.vectorizer.fit_transform(document_set)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get important terms from context
        context_tfidf = tfidf_matrix[0].toarray()[0]
        important_terms = self._get_important_terms(context_tfidf, feature_names)
        
        # Extract sentences and create Q&A pairs
        sentences = sent_tokenize(context)
        qa_pairs = self._extract_qa_pairs_tfidf(sentences, important_terms, context)
        
        # Generate MCQs
        mcqs = []
        for qa_pair in qa_pairs[:self.num_questions]:
            mcq = self._create_mcq(qa_pair, important_terms, document_set)
            if mcq:
                mcqs.append(mcq)
        
        return mcqs
    
    def _get_important_terms(self, tfidf_scores: np.ndarray, feature_names: np.ndarray) -> List[str]:
        """
        Extract important terms based on TF-IDF scores.
        
        Args:
            tfidf_scores: TF-IDF scores array
            feature_names: Feature names from vectorizer
            
        Returns:
            List of important terms sorted by TF-IDF score
        """
        # Get indices of top scores
        top_indices = np.argsort(tfidf_scores)[::-1]
        
        important_terms = []
        for idx in top_indices:
            if tfidf_scores[idx] > 0:
                term = feature_names[idx]
                # Filter by length and non-stopwords
                if len(term) >= self.min_term_length or len(term.split()) > 1:
                    important_terms.append(term)
                    if len(important_terms) >= 10:
                        break
        
        return important_terms
    
    def _extract_qa_pairs_tfidf(
        self, 
        sentences: List[str], 
        important_terms: List[str],
        context: str
    ) -> List[Dict]:
        """
        Extract Q&A pairs from sentences using important terms.
        
        Args:
            sentences: List of sentences
            important_terms: List of important terms from TF-IDF
            context: Full context text
            
        Returns:
            List of Q&A pairs with scores
        """
        qa_pairs = []
        
        for sentence in sentences:
            # Find important terms in sentence
            for term in important_terms:
                if term.lower() in sentence.lower():
                    qa_pair = {
                        'sentence': sentence,
                        'term': term,
                        'answer': term,
                        'context': context
                    }
                    qa_pairs.append(qa_pair)
                    break  # One term per sentence
        
        return qa_pairs
    
    def _create_mcq(
        self, 
        qa_pair: Dict, 
        important_terms: List[str],
        document_set: List[str]
    ) -> Dict:
        """
        Create MCQ with options using TF-IDF similarity.
        
        Args:
            qa_pair: Q&A pair dictionary
            important_terms: List of important terms
            document_set: Set of documents for similarity calculation
            
        Returns:
            MCQ dictionary
        """
        answer = qa_pair['answer']
        sentence = qa_pair['sentence']
        
        # Create question by blanking out the answer term
        question_text = sentence
        pattern = r'\b' + re.escape(answer) + r'\b'
        question_text = re.sub(pattern, '_____', question_text, flags=re.IGNORECASE)
        
        # Generate distractors using similarity
        distractors = self._generate_similar_distractors(answer, important_terms, document_set)
        
        if len(distractors) < self.num_options - 1:
            return None
        
        # Create options
        options = [answer] + distractors[:self.num_options - 1]
        
        # Shuffle options
        import random
        random.shuffle(options)
        correct_index = options.index(answer)
        
        return {
            'question': f"Fill in the blank: {question_text}",
            'options': options,
            'correct_answer': answer,
            'correct_index': correct_index,
            'context': sentence,
            'type': 'fill_blank'
        }
    
    def _generate_similar_distractors(
        self, 
        answer: str, 
        important_terms: List[str],
        document_set: List[str]
    ) -> List[str]:
        """
        Generate distractors similar to the answer using TF-IDF.
        
        Args:
            answer: Correct answer term
            important_terms: List of important terms to choose from
            document_set: Document set for similarity calculation
            
        Returns:
            List of distractor terms
        """
        distractors = []
        
        # Vectorize all terms
        all_terms = [answer] + important_terms
        try:
            term_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 3))
            term_vectors = term_vectorizer.fit_transform(all_terms)
            
            # Calculate similarity between answer and other terms
            answer_vector = term_vectors[0]
            similarities = cosine_similarity(answer_vector, term_vectors[1:])[0]
            
            # Get most similar terms as distractors
            similar_indices = np.argsort(similarities)[::-1]
            
            for idx in similar_indices:
                if idx + 1 < len(important_terms):  # +1 because answer is at index 0
                    distractor = important_terms[idx]
                    if distractor != answer and distractor not in distractors:
                        distractors.append(distractor)
                        if len(distractors) >= self.num_options - 1:
                            break
        except:
            # Fallback: just use important_terms
            distractors = [t for t in important_terms if t != answer][:self.num_options - 1]
        
        return distractors[:self.num_options - 1]


if __name__ == "__main__":
    # Example usage
    generator = TFIDFMCQGenerator(num_options=4, num_questions=5)
    
    sample_context = """
    Natural Language Processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and human language. 
    NLP is used to apply machine learning algorithms to text and speech. 
    Common NLP tasks include sentiment analysis, machine translation, and question answering.
    Deep learning models like transformers have significantly improved NLP performance.
    """
    
    # Additional documents for better IDF calculation
    additional_docs = [
        "Computer Vision is the field of artificial intelligence that trains computers to interpret and understand the visual world.",
        "Robotics is an interdisciplinary branch of engineering and science that includes mechanical engineering, electrical engineering, and computer science."
    ]
    
    mcqs = generator.generate_questions(sample_context, additional_docs)
    
    for i, mcq in enumerate(mcqs, 1):
        print(f"\nQuestion {i}: {mcq['question']}")
        print(f"Options:")
        for j, option in enumerate(mcq['options'], 1):
            marker = "✓" if option == mcq['correct_answer'] else " "
            print(f"  {marker} {j}. {option}")
