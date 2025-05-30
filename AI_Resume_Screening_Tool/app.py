from flask import Flask, request, jsonify, render_template
from flask_pymongo import PyMongo
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import docx2txt
import PyPDF2
import numpy as np
import ffmpeg
import speech_recognition as sr
from pydub import AudioSegment
from sentence_transformers import SentenceTransformer, util
from flask import session, redirect, url_for
from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Add this line for sessions

app.config["UPLOAD_FOLDER"] = "uploads"
app.config['MONGO_URI'] = 'mongodb://localhost:27017/resume_db'  # Replace with your actual MongoDB UR
mongo = PyMongo(app)


model = SentenceTransformer("all-MiniLM-L6-v2")


# Extract text from PDF
def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

# Extract text from DOCX
def extract_text_from_docx(filepath):
    return docx2txt.process(filepath)
def extract_text_from_video(filepath):
    audio_path = filepath.replace(".mp4", ".wav")

    # Extract audio using ffmpeg
    try:
        ffmpeg.input(filepath).output(audio_path, format='wav', ac=1, ar='8000').run(quiet=True, overwrite_output=True)

    except ffmpeg.Error as e:
        print("FFmpeg error:", e.stderr.decode())
        return ""

    # Transcribe audio
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        text = ""
    except sr.RequestError:
        text = ""

    return text



# Real relevance score using TF-IDF and cosine similarity
# Real relevance score using sentence-transformer
def compute_relevance_score(content, job_description):
    content_embedding = model.encode(content, convert_to_tensor=True)
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    cosine_sim = util.pytorch_cos_sim(jd_embedding, content_embedding)
    score = float(cosine_sim.item()) * 100  # scale to 0-100
    return round(score, 2)



def compute_cultural_fit(content):
    traits = ["teamwork", "adaptability", "communication", "integrity", "accountability"]
    traits_embeddings = model.encode(traits, convert_to_tensor=True)
    content_embedding = model.encode(content, convert_to_tensor=True)

    sims = util.pytorch_cos_sim(traits_embeddings, content_embedding)
    avg_score = sims.mean().item()
    return round(avg_score * 100, 2)



def compute_bias_score(content):
    biased_terms = ["young", "recent graduate", "energetic", "manpower", "native speaker"]

    # Encode all biased terms once
    biased_embeddings = model.encode(biased_terms, convert_to_tensor=True)

    content_embedding = model.encode(content, convert_to_tensor=True)

    # Compute cosine similarity between content and all biased terms
    sims = util.pytorch_cos_sim(biased_embeddings, content_embedding)

    bias_hits = (sims > 0.6).sum().item()  # count how many are similar

    score = max(0, 100 - bias_hits * 20)
    return round(score, 2)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username and password match in the MongoDB collection
        user = mongo.db.users.find_one({'username': username, 'password': password})
        if user:
            session['username'] = username
            return redirect(url_for('index'))  # Redirect to the index page after successful login
        else:
            return "Invalid username or password!"
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username already exists
        existing_user = mongo.db.users.find_one({'username': username})
        if existing_user:
            return "Username already exists! Please choose a different username."

        # Store new user in MongoDB
        mongo.db.users.insert_one({'username': username, 'password': password})
        return redirect(url_for('login'))  # Redirect to login page after successful registration

    return render_template('register.html')

@app.route('/index')
def index():
    if 'username' in session:
        return render_template('index.html', username=session['username'])  # Display your index.html
    else:
        return redirect(url_for('login'))  # If no session, redirect to login

@app.route('/logout')
def logout():
    session.pop('username', None)  # Log out the user by removing the session
    return redirect(url_for('login'))


@app.route("/upload", methods=["POST"])
def upload_resume():
    if 'username' not in session:
        return redirect(url_for('login'))

    uploaded_file = request.files["resume"]
    if uploaded_file.filename != "":
        filename = secure_filename(uploaded_file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded_file.save(filepath)
        if filename.endswith(".pdf"):
            content = extract_text_from_pdf(filepath)
        elif filename.endswith(".docx"):
             content = extract_text_from_docx(filepath)
        elif filename.endswith(".mp4"):
             content = extract_text_from_video(filepath)
        else:
            return jsonify({"error": "Unsupported file format"}), 400


        job_description = request.form["job_description"]
        relevance_score = compute_relevance_score(content, job_description)
        cultural_score = compute_cultural_fit(content)
        bias_score = compute_bias_score(content)

        # Normalize final score
        final_score = (
    0.6 * relevance_score +
    0.3 * cultural_score +
    0.1 * (100 - bias_score)
)
        final_score = round(final_score, 2)


        mongo.db.candidates.insert_one({
            "filename": filename,
            "relevance_score": relevance_score,
            "cultural_score": cultural_score,
            "bias_score": round(bias_score, 2),
            "final_score": final_score,
        })

        return jsonify({
            "message": "Resume uploaded and processed successfully.",
            "filename": filename,
            "relevance_score": relevance_score,
            "cultural_score": cultural_score,
            "bias_score": round(bias_score, 2),
            "final_score": final_score
        })

    return jsonify({"error": "No file uploaded"}), 400

@app.route("/candidates", methods=["GET"])
def get_candidates():
    if 'username' not in session:
        return redirect(url_for('login'))

    candidates = list(mongo.db.candidates.find({}, {"_id": 0}))
    sorted_candidates = sorted(candidates, key=lambda x: x['final_score'], reverse=True)
    
    # Delete candidates after sending them once
    mongo.db.candidates.delete_many({})
    
    return jsonify(sorted_candidates)
@app.route("/chatbot", methods=["POST"])
def chatbot():
    if 'username' not in session:
        return redirect(url_for('login'))

    data = request.json
    message = data.get("message", "").lower()

    # Predefined responses based on keywords
    responses = {
        "how to upload": "To upload, go to the Upload Resumes section and select a PDF file with your name.",
        "how scoring works": "Final score = 65% Relevance + 25% Cultural Fit + 10% Bias Score (less bias = better).",
        "how do i register": "Go to the login page and switch to Sign Up to create an account.",
        "hello": "Hello! How can I assist you today?",
        "hi": "Hi there! Need help with something?",
        "help": "You can ask me about uploading resumes, scoring, or registration.",
        "goodbye": "Goodbye! Have a great day!",
    }

    # Basic rule-based response
    for key, reply in responses.items():
        if key in message:
            return jsonify({"reply": reply})

    # If no keyword matched, provide a generic response
    return jsonify({"reply": "I'm sorry, I didn't understand that. Can you ask something related to uploading or scoring?"})


if __name__ == "__main__":
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)


