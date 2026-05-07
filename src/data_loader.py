"""
Dataset loader for SQuAD and similar QA datasets.
Loads and processes data for MCQ generation models.
"""

import json
import os
from typing import List, Dict, Tuple, Optional
from datasets import load_dataset
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))


class SQuADDataLoader:
    """Load and process SQuAD dataset for MCQ generation."""
    
    def __init__(self, dataset_name: str = "squad", split: str = "validation", limit: Optional[int] = None):
        """
        Initialize SQuAD dataset loader.
        
        Args:
            dataset_name: Name of dataset (e.g., 'squad', 'squad_v2')
            split: Dataset split ('train', 'validation')
            limit: Maximum number of samples to load (None for all)
        """
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit
        self.data = None
        self.processed_data = None
        
    def load_dataset(self) -> List[Dict]:
        """
        Load SQuAD dataset from Hugging Face and save to data directory.
        
        Returns:
            List of processed dataset records
        """
        print(f"Loading {self.dataset_name} ({self.split} split)...")
        
        # Load from Hugging Face datasets
        dataset = load_dataset(self.dataset_name, split=self.split)
        
        # Limit dataset if specified
        if self.limit:
            dataset = dataset.select(range(min(self.limit, len(dataset))))
        
        # Convert to list of dictionaries
        self.data = [item for item in dataset]
        print(f"Loaded {len(self.data)} records")
        
        # Save raw dataset to data directory
        self._save_raw_dataset()
        
        return self.data
    
    def process_data(self) -> List[Dict]:
        """
        Process raw SQuAD data into standardized format and save to data directory.
        
        Returns:
            List of processed records with: context, question, answers, context_id
        """
        if self.data is None:
            self.load_dataset()
        
        processed = []
        
        for idx, item in enumerate(self.data):
            # Handle both original and flattened SQuAD format
            if 'qas' in item:
                # Original format: item has 'qas' list
                title = item.get('title', 'Unknown')
                context = item['context']
                
                for qa_idx, qa in enumerate(item['qas']):
                    question = qa['question']
                    answers = qa.get('answers', [])
                    answer_texts = [answer['text'] for answer in answers]
                    
                    processed.append({
                        'context_id': f"{idx}_{qa_idx}",
                        'title': title,
                        'context': context,
                        'question': question,
                        'answers': answer_texts,
                        'is_impossible': qa.get('is_impossible', False),
                    })
            else:
                # Flattened format from Hugging Face datasets: each item is a Q&A pair
                title = item.get('title', 'Unknown')
                context = item.get('context', '')
                question = item.get('question', '')
                
                # Extract answer texts
                answers_obj = item.get('answers', {})
                if isinstance(answers_obj, dict) and 'text' in answers_obj:
                    answer_texts = answers_obj['text'] if isinstance(answers_obj['text'], list) else [answers_obj['text']]
                elif isinstance(answers_obj, list):
                    answer_texts = [ans.get('text', '') if isinstance(ans, dict) else ans for ans in answers_obj]
                else:
                    answer_texts = []
                
                processed.append({
                    'context_id': f"{idx}",
                    'title': title,
                    'context': context,
                    'question': question,
                    'answers': answer_texts,
                })
        
        self.processed_data = processed
        print(f"Processed {len(processed)} QA pairs")
        
        # Save processed data to data directory
        self._save_processed_dataset()
        
        return processed
    
    def _save_raw_dataset(self) -> None:
        """Save raw dataset to data directory."""
        data_dir = os.getenv('DATA_DIR', './data')
        os.makedirs(data_dir, exist_ok=True)
        
        limit_suffix = f"_{self.limit}" if self.limit else ""
        filepath = os.path.join(data_dir, f"{self.dataset_name}_{self.split}_raw{limit_suffix}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved raw dataset to {filepath}")
    
    def _save_processed_dataset(self) -> None:
        """Save processed dataset to data directory."""
        data_dir = os.getenv('DATA_DIR', './data')
        os.makedirs(data_dir, exist_ok=True)
        
        limit_suffix = f"_{self.limit}" if self.limit else ""
        filepath = os.path.join(data_dir, f"{self.dataset_name}_{self.split}_processed{limit_suffix}.json")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Saved processed dataset to {filepath}")
    
    def get_contexts(self) -> Dict[str, str]:
        """
        Get unique contexts.
        
        Returns:
            Dictionary of context_id: context
        """
        if self.processed_data is None:
            self.process_data()
        
        contexts = {}
        seen = set()
        
        for item in self.processed_data:
            context = item['context']
            if context not in seen:
                context_id = f"ctx_{len(contexts)}"
                contexts[context_id] = context
                seen.add(context)
        
        return contexts
    
    def get_train_test_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Split processed data into train and test sets.
        
        Args:
            test_size: Proportion of test set
            random_state: Random seed
            
        Returns:
            Tuple of (train_data, test_data)
        """
        if self.processed_data is None:
            self.process_data()
        
        df = pd.DataFrame(self.processed_data)
        
        # Split by context to avoid data leakage
        unique_contexts = df['context'].unique()
        n_test = int(len(unique_contexts) * test_size)
        
        import numpy as np
        np.random.seed(random_state)
        test_contexts = np.random.choice(unique_contexts, n_test, replace=False)
        
        test_mask = df['context'].isin(test_contexts)
        train_data = df[~test_mask].to_dict('records')
        test_data = df[test_mask].to_dict('records')
        
        print(f"Train set: {len(train_data)} samples")
        print(f"Test set: {len(test_data)} samples")
        
        return train_data, test_data
    
    def save_to_json(self, output_path: str) -> None:
        """
        Save processed data to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        if self.processed_data is None:
            self.process_data()
        
        # If relative path, save to DATA_DIR
        if not os.path.isabs(output_path) and not output_path.startswith('./'):
            data_dir = os.getenv('DATA_DIR', './data')
            output_path = os.path.join(data_dir, output_path)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.processed_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved processed data to {output_path}")
    
    def load_from_json(self, input_path: str) -> List[Dict]:
        """
        Load processed data from JSON file.
        
        Args:
            input_path: Path to input JSON file
            
        Returns:
            List of processed records
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            self.processed_data = json.load(f)
        
        print(f"Loaded {len(self.processed_data)} records from {input_path}")
        return self.processed_data


class LocalDataLoader:
    """Load MCQ data from local files."""
    
    @staticmethod
    def load_json(filepath: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def load_csv(filepath: str) -> pd.DataFrame:
        """Load data from CSV file."""
        return pd.read_csv(filepath)
    
    @staticmethod
    def save_json(data: List[Dict], filepath: str) -> None:
        """Save data to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Load configuration from .env
    hf_token = os.getenv('HF_TOKEN')
    data_dir = os.getenv('DATA_DIR', './data')
    
    print("Configuration loaded from .env:")
    print(f"  DATA_DIR: {data_dir}")
    print(f"  HF_TOKEN: {'✓ Set' if hf_token and hf_token != 'your_huggingface_token_here' else '⚠ Not set or placeholder'}\n")
    
    if hf_token == 'your_huggingface_token_here':
        print("⚠ Please update .env file with your Hugging Face token:")
        print("  1. Go to https://huggingface.co/settings/tokens")
        print("  2. Copy your token")
        print("  3. Update HF_TOKEN in .env file\n")
    
    print("Loading SQuAD validation split (test data)...\n")
    loader = SQuADDataLoader(limit=100)
    loader.process_data()
    
    print("\n✓ Data automatically saved to data/ directory:")
    print("  - squad_validation_raw_100.json (raw dataset)")
    print("  - squad_validation_processed_100.json (processed QA pairs)")
