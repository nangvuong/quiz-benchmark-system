import os
import json
from src.data_loader import SQuADDataLoader
from src.load_model.rule_based import RuleBasedMCQGenerator
from src.load_model.tfidf_model import TFIDFMCQGenerator
from src.load_model.t5_model import T5MCQGenerator

def setup_nltk():
    import nltk
    print("Setting up NLTK resources...")
    resources = ['punkt', 'punkt_tab', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'wordnet', 'stopwords']
    for res in resources:
        try:
            if res in ['wordnet', 'stopwords']:
                nltk.data.find(f'corpora/{res}')
            elif res in ['punkt', 'punkt_tab']:
                nltk.data.find(f'tokenizers/{res}')
            else:
                nltk.data.find(f'taggers/{res}')
        except LookupError:
            print(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)

def main():
    setup_nltk()
    print("=== MCQ Generation Pipeline ===")
    
    # 1. Prepare and normalize input with SQuAD dataset
    print("\n1. Loading and normalizing SQuAD dataset...")
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
    
    limit_str = os.getenv('DATASET_LIMIT')
    limit = int(limit_str) if limit_str and limit_str.lower() != 'none' else None
    print(f"Loading dataset with limit: {limit if limit else 'ALL'}")
    
    loader = SQuADDataLoader(limit=limit)
    processed_data = loader.process_data()
    
    # Extract unique contexts from the processed data
    contexts = list({item['context'] for item in processed_data})
    print(f"Loaded {len(contexts)} unique contexts.")
    
    # 2. Initialize Models
    print("\n2. Initializing Models...")
    print("- Initializing Rule-Based Model...")
    rule_based_gen = RuleBasedMCQGenerator(num_questions=2)
    
    print("- Initializing TF-IDF Model...")
    tfidf_gen = TFIDFMCQGenerator(num_questions=2)
    
    print("- Initializing T5 Model...")
    try:
        t5_gen = T5MCQGenerator(num_questions=2, device="cpu")
        t5_available = True
    except Exception as e:
        print(f"Failed to load T5 model: {e}")
        t5_available = False
    
    # 3. Generate questions using models
    print("\n3. Generating MCQs from normalized texts...")
    results = []
    
    for i, context in enumerate(contexts):
        if i % 10 == 0 or i == len(contexts) - 1:
            print(f"\nProcessing Context {i+1}/{len(contexts)} ({(i+1)/len(contexts)*100:.1f}%):")
        
        context_results = {
            "context": context,
            "questions": {}
        }
        
        # Rule-based
        print("  -> Generating with Rule-Based Model...")
        rb_mcqs = rule_based_gen.generate_questions(context)
        context_results["questions"]["rule_based"] = rb_mcqs
        
        # TF-IDF
        print("  -> Generating with TF-IDF Model...")
        tfidf_mcqs = tfidf_gen.generate_questions(context, document_set=contexts)
        context_results["questions"]["tfidf"] = tfidf_mcqs
        
        # T5
        if t5_available:
            print("  -> Generating with T5 Model...")
            t5_mcqs = t5_gen.generate_questions(context)
            context_results["questions"]["t5"] = t5_mcqs
            
        results.append(context_results)
        
    # 4. Save results
    output_file = "data/generated_mcqs.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\n=== Done! Generated MCQs saved to {output_file} ===")
    
    # Print a sample output
    if results and results[0]["questions"].get("rule_based"):
        print("\nSample Output (Rule-Based):")
        sample_q = results[0]["questions"]["rule_based"][0]
        print(f"Q: {sample_q['question']}")
        print(f"Options: {sample_q['options']}")
        print(f"Correct Answer: {sample_q['correct_answer']}")

if __name__ == "__main__":
    main()
