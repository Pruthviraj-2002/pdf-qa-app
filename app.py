# app.py (Backend)

from flask import Flask, request, jsonify
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# ----------- PDF Processing -----------
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = " ".join([page.get_text() for page in doc])
    return re.sub(r'\s+', ' ', text)

def split_text(text, chunk_size=700):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ----------- RAG-Style QA -----------
def rag_qa(question, chunks):
    vectorizer = TfidfVectorizer().fit([question] + chunks)
    vectors = vectorizer.transform([question] + chunks)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
    best_chunk = chunks[similarities.argmax()]
    
    sentences = re.split(r'(?<=[.!?]) +', best_chunk)
    for sent in sentences:
        if any(qword in sent.lower() for qword in question.lower().split()):
            return sent.strip()
    return best_chunk.strip()[:500] + "..."

# ----------- Load PDF Once -----------
pdf_path = r"test.pdf"
full_text = extract_text_from_pdf(pdf_path)
chunks = split_text(full_text)

# ----------- API Route -----------
@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = rag_qa(question, chunks)
    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
