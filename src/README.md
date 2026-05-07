# MCQ Generation System - Source Code

This directory contains the complete implementation for generating multiple-choice questions (MCQs) using three different approaches: rule-based, TF-IDF, and T5 transformer models.

## Project Structure

```
src/
├── data_loader.py           # Load and process SQuAD dataset
├── utils.py                 # Utility functions for text processing
├── models/
│   ├── __init__.py
│   ├── rule_based.py        # Rule-based MCQ generator
│   ├── tfidf_model.py       # TF-IDF based MCQ generator
│   └── t5_model.py          # T5 transformer MCQ generator
├── test_models.py           # Test script for all models
├── examples.py              # Usage examples and quick start
└── README.md               # This file
```

## Components

### 1. Data Loader (`data_loader.py`)

Loads and processes the SQuAD dataset for MCQ generation.

**Key Classes:**
- `SQuADDataLoader`: Download, process, and manage SQuAD dataset
- `LocalDataLoader`: Load data from local JSON/CSV files

**Usage:**
```python
from data_loader import SQuADDataLoader

# Load dataset
loader = SQuADDataLoader(limit=100)
processed_data = loader.process_data()

# Split data
train_data, test_data = loader.get_train_test_split()

# Save for later
loader.save_to_json('data/squad.json')
```

### 2. Utilities (`utils.py`)

Text processing utilities for NLP tasks.

**Key Functions:**
- `preprocess_text()`: Clean and normalize text
- `tokenize_sentences()`: Split text into sentences
- `tokenize_words()`: Split text into words
- `extract_key_sentences()`: Extract important sentences
- `get_noun_phrases()`: Extract noun phrases
- `get_text_statistics()`: Calculate text metrics

### 3. Rule-Based Model (`models/rule_based.py`)

Fast, interpretable MCQ generation using linguistic rules and heuristics.

**Pros:**
- Fast execution
- No training required
- Easy to understand and debug

**Cons:**
- Limited question variety
- May generate awkward sentences
- Relies on predefined patterns

**Usage:**
```python
from models.rule_based import RuleBasedMCQGenerator

generator = RuleBasedMCQGenerator(num_options=4, num_questions=5)
mcqs = generator.generate_questions(context)
```

### 4. TF-IDF Model (`models/tfidf_model.py`)

MCQ generation based on term importance using TF-IDF scoring.

**Pros:**
- Identifies important terms automatically
- Better than rule-based for some texts
- Relatively fast
- Good for domain-specific text

**Cons:**
- Generated questions may be template-like
- Requires document collection for good IDF

**Usage:**
```python
from models.tfidf_model import TFIDFMCQGenerator

generator = TFIDFMCQGenerator(num_options=4, num_questions=5)
mcqs = generator.generate_questions(context, document_set)
```

### 5. T5 Model (`models/t5_model.py`)

Advanced MCQ generation using pre-trained T5 transformer model.

**Pros:**
- Most natural questions
- Good question variety
- Handles complex text well
- State-of-the-art performance

**Cons:**
- Requires GPU for fast inference
- Larger memory footprint
- Model download required (~900MB)
- Slower than rule-based/TF-IDF

**Usage:**
```python
from models.t5_model import T5MCQGenerator

generator = T5MCQGenerator(model_name="t5-base", device="cpu")
mcqs = generator.generate_questions(context)
```

## MCQ Format

All models return MCQs in the following format:

```python
{
    'question': 'What is ...?',
    'options': ['Option 1', 'Option 2', 'Option 3', 'Option 4'],
    'correct_answer': 'Option 1',
    'correct_index': 0,
    'context': 'Original context text...',
    'type': 'multiple_choice',
    'model': 'model_name'  # (for T5 only)
}
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Tests

```bash
python src/test_models.py
```

This will test all three models on sample data.

### 3. Run Examples

```bash
python src/examples.py
```

Explore different usage patterns and model comparisons.

### 4. Quick Start

```python
from data_loader import SQuADDataLoader
from models.rule_based import RuleBasedMCQGenerator
from models.tfidf_model import TFIDFMCQGenerator
from models.t5_model import T5MCQGenerator

# Load data
loader = SQuADDataLoader(limit=50)
data = loader.process_data()

# Get contexts
contexts = [item['context'] for item in data[:5]]

# Generate MCQs with all models
rb_gen = RuleBasedMCQGenerator()
tfidf_gen = TFIDFMCQGenerator()
t5_gen = T5MCQGenerator()

for context in contexts:
    rb_mcqs = rb_gen.generate_questions(context)
    tfidf_mcqs = tfidf_gen.generate_questions(context)
    t5_mcqs = t5_gen.generate_questions(context)
```

## Model Comparison

| Feature | Rule-Based | TF-IDF | T5 |
|---------|-----------|--------|-----|
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Quality | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Memory | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| Variety | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Training | ❌ | ❌ | ❌ (pre-trained) |

## Configuration

### Rule-Based Model Parameters
- `num_options`: Number of options (default: 4)
- `num_questions`: Questions to generate (default: 5)

### TF-IDF Model Parameters
- `num_options`: Number of options (default: 4)
- `num_questions`: Questions to generate (default: 5)
- `min_term_length`: Minimum term length for extraction (default: 4)

### T5 Model Parameters
- `model_name`: T5 variant ('t5-base', 't5-large', etc., default: 't5-base')
- `device`: Computation device ('cpu' or 'cuda', default: 'cpu')
- `num_questions`: Questions to generate (default: 5)
- `num_options`: Number of options (default: 4)
- `max_length`: Maximum sequence length (default: 512)
- `num_beams`: Beam search width (default: 4)

## Performance Tips

1. **Rule-Based Model:**
   - Fastest option
   - Best for quick generation
   - Use when speed matters more than quality

2. **TF-IDF Model:**
   - Good balance of speed and quality
   - Provide document set for better results
   - Best for technical/specialized texts

3. **T5 Model:**
   - Use GPU for 5-10x speedup
   - Batch process multiple contexts
   - Best for production quality
   - Cache model after first load

## Batch Processing

```python
from models.t5_model import T5MCQGenerator

generator = T5MCQGenerator(device="cpu")
contexts = [text1, text2, text3, ...]

all_mcqs = generator.batch_generate_questions(contexts)
```

## Output and Results

Save MCQs to file:

```python
import json

with open('mcqs.json', 'w', encoding='utf-8') as f:
    json.dump(mcqs, f, ensure_ascii=False, indent=2)
```

## Troubleshooting

### T5 Model Download Issues
```bash
# Pre-download T5 model
python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; \
T5Tokenizer.from_pretrained('t5-base'); \
T5ForConditionalGeneration.from_pretrained('t5-base')"
```

### Out of Memory Errors
- Reduce `max_length` parameter
- Use smaller T5 variant ('t5-small')
- Process in smaller batches
- Use CPU instead of GPU

### Slow Performance
- Use GPU if available: `device="cuda"`
- Reduce `num_beams` in T5 (trade-off with quality)
- Use rule-based model for speed
- Pre-compute and cache results

## Dataset Citation

SQuAD v1.1: https://arxiv.org/abs/1606.04965

```bibtex
@article{rajpurkar2016squad,
  title={SQuAD: 100,000+ Questions for Machine Reading Comprehension of Text},
  author={Rajpurkar, Pranav and Zhang, Michael and Liang, Percy and Liang, Dan},
  journal={arXiv preprint arXiv:1606.04965},
  year={2016}
}
```

## License

This project is part of the Quiz Benchmark System.

## Support

For issues or questions:
1. Check `examples.py` for usage patterns
2. Run `test_models.py` to verify installation
3. Review inline documentation in source files
