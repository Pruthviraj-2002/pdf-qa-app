# app.py (Backend)

from flask import Flask, request, jsonify
from flask import Flask, render_template
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from flask_cors import CORS

from flask import Flask, render_template
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")



# ----------- PDF Processing ----------
import fitz  # PyMuPDF
def extract_text_from_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text if text.strip() else ""  # return empty string if no text found
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""

def split_text(text, chunk_size=700):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# ----------- RAG-Style QA -----------
def rag_qa(question, chunks):
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    # Clean headers and unwanted lines
    def clean_chunk(text):
        lines = text.split("\n")
        content = [line for line in lines if not re.match(r'(SCERT|X Class|Fig|Chapter|\d{1,3} ?Class)', line)]
        return " ".join(content).strip()

    cleaned_chunks = [clean_chunk(chunk) for chunk in chunks]

    # TF-IDF match
    vectorizer = TfidfVectorizer().fit([question] + cleaned_chunks)
    vectors = vectorizer.transform([question] + cleaned_chunks)
    similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    best_chunk = cleaned_chunks[similarities.argmax()]
    sentences = re.split(r'(?<=[.!?]) +', best_chunk)

    # Try to find a definition-like sentence
    definition_sentences = [s for s in sentences if re.match(rf"{question.strip().capitalize()} (is|refers to|means)", s, re.IGNORECASE)]

    if definition_sentences:
        # Return the first clear definition found
        return definition_sentences[0].strip()

    # Otherwise, fallback to TF-IDF on sentences
    sentence_vectors = vectorizer.transform(sentences)
    sentence_similarities = cosine_similarity(vectors[0:1], sentence_vectors).flatten()
    ranked = sentence_similarities.argsort()[::-1]

    answer_sentences = []
    for i in ranked:
        sentence = sentences[i].strip()
        if len(sentence) < 20 or not re.search(r'[a-zA-Z]', sentence):
            continue
        if any(kw in sentence.lower() for kw in ['figure', 'diagram', 'arrow', 'web']):
            continue
        answer_sentences.append(sentence)
        if len(answer_sentences) == 2:
            break

    answer = " ".join(answer_sentences).strip()
    return answer if answer else "Sorry, I couldn’t find a clear explanation for that."


# ----------- Load PDF Once -----------
pdf_path = r"test.pdf"

full_text = extract_text_from_pdf(pdf_path)
if not full_text:
    raise ValueError("No text could be extracted from the PDF.")

chunks = split_text(full_text)


# ----------- API Route -----------

@app.route("/ask", methods=["POST"])
def ask():
    question = request.form["question"]
    pdf = request.files["pdf"]
    
    # Save and process the uploaded PDF
    pdf_path = "uploaded.pdf"
    pdf.save(pdf_path)

    # Extract text AFTER uploading
    full_text = extract_text_from_pdf(pdf_path)
    chunks = split_text(full_text)
    ...


if __name__ == '__main__':
    app.run(debug=True)
