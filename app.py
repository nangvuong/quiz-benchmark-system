from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import traceback
import os
from dotenv import load_dotenv

from src.load_model.rule_based import RuleBasedMCQGenerator
from src.load_model.tfidf_model import TFIDFMCQGenerator
from src.load_model.t5_model import T5MCQGenerator

load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
CORS(app)

models = {}

def init_models():
    print("Initializing models for Web Demo...")
    try:
        models['rule_based'] = RuleBasedMCQGenerator()
        print("Rule-Based model loaded.")
    except Exception as e:
        print("Failed to load Rule-Based model:", e)
        
    try:
        models['tfidf'] = TFIDFMCQGenerator()
        print("TF-IDF model loaded.")
    except Exception as e:
        print("Failed to load TF-IDF model:", e)
        
    try:
        models['t5'] = T5MCQGenerator(num_questions=2, device="cpu")
        print("T5 model loaded.")
    except Exception as e:
        print("Failed to load T5 model:", e)

@app.route('/')
def index():
    return render_template('index.html', available_models=list(models.keys()))

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    context = data.get('context', '').strip()
    selected_models = data.get('models', [])
    
    if not context:
        return jsonify({"error": "Context cannot be empty."}), 400
        
    results = {}
    
    for m in selected_models:
        if m in models:
            try:
                mcqs = models[m].generate_questions(context)
                results[m] = mcqs
            except Exception as e:
                traceback.print_exc()
                results[m] = [{"error": str(e)}]
                
    return jsonify({"results": results})

if __name__ == '__main__':
    # Initialize nltk first
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('stopwords', quiet=True)
    
    init_models()
    app.run(host='0.0.0.0', debug=True, port=5001, use_reloader=False)
