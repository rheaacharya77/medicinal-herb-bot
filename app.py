import os
import logging
from flask import Flask, request, jsonify, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOllama
import ollama

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

# Flask app initialization
app = Flask(__name__)

# Environment settings
CONFIG = {
    "DOC_PATH": "./medicinal-herbs.pdf",
    "MODEL_NAME": "llama3.2",
    "EMBEDDING_MODEL": "nomic-embed-text",
    "VECTOR_STORE_NAME": "simple-rag",
    "PERSIST_DIRECTORY": "./chroma_db",
    "CLASSIFICATION_MODEL_PATH": "./medicinal-herb-final-model.h5",
    "INPUT_IMAGE_SIZE": (150, 150),
    "UPLOAD_FOLDER": "static/uploads",
    "ALLOWED_EXTENSIONS": {"png", "jpg", "jpeg"},
    "CLASS_TO_PLANT_NAME": {
        0: "Aloevera",
        1: "Bamboo",
        2: "Castor",
        3: "Neem",
        4: "Tamarind",
    }
}
app.config['UPLOAD_FOLDER'] = CONFIG['UPLOAD_FOLDER']

# Utility function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in CONFIG['ALLOWED_EXTENSIONS']

# Load the TensorFlow classification model
def load_model():
    try:
        model = tf.keras.models.load_model(CONFIG['CLASSIFICATION_MODEL_PATH'])
        logging.info("TensorFlow model loaded successfully.")
        return model
    except Exception as e:
        logging.error(f"Error loading TensorFlow model: {e}")
        return None

model = load_model()

# Classify plant species from an uploaded image using the trained model
def classify_plant(image_path):
    try:
        image = Image.open(image_path).resize(CONFIG['INPUT_IMAGE_SIZE'])
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = tf.expand_dims(img_array, axis=0) / 255.0  

        predictions = model.predict(img_array)
        plant_class = tf.argmax(predictions, axis=-1).numpy()[0]
        return CONFIG['CLASS_TO_PLANT_NAME'].get(plant_class, "Unknown Plant")
    except Exception as e:
        logging.error(f"Error in classify_plant: {e}")
        return None

# Load and ingest PDF data for creating a vector database
def ingest_pdf():
    try:
        loader = UnstructuredPDFLoader(CONFIG['DOC_PATH'])
        return loader.load()
    except Exception as e:
        logging.error(f"Error loading PDF: {e}")
        return None

# Split documents into manageable chunks for embedding and retrieval
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    return splitter.split_documents(documents)

# Load or create a vector database from PDF content
def load_vector_db():
    try:
        ollama.pull(CONFIG['EMBEDDING_MODEL'])
        embedding = OllamaEmbeddings(model=CONFIG['EMBEDDING_MODEL'])

        if os.path.exists(CONFIG['PERSIST_DIRECTORY']):
            vector_db = Chroma(
                embedding_function=embedding,
                collection_name=CONFIG['VECTOR_STORE_NAME'],
                persist_directory=CONFIG['PERSIST_DIRECTORY']
            )
            logging.info("Loaded existing vector database.")
        else:
            data = ingest_pdf()
            if data is None:
                return None

            chunks = split_documents(data)
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=CONFIG['VECTOR_STORE_NAME'],
                persist_directory=CONFIG['PERSIST_DIRECTORY']
            )
            vector_db.persist()
            logging.info("Vector database created and persisted.")

        return vector_db
    except Exception as e:
        logging.error(f"Error loading vector database: {e}")
        return None
    
# Create a QA chain for retrieving answers from vector database
def create_chain(vector_db, model):
    try:
        retriever = vector_db.as_retriever()
        return RetrievalQA.from_chain_type(llm=model, retriever=retriever)
    except Exception as e:
        logging.error(f"Error creating chain: {e}")
        return None

# Routes
@app.route("/")
def home():
    return render_template("chat.html")

# Endpoint to upload and store images
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image part'}), 400
    file = request.files['image']

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        image_url = url_for('static', filename=f'uploads/{file.filename}')
        return jsonify({'image_url': image_url})

    return jsonify({'error': 'Invalid file type'}), 400

# Endpoint for plant classification
@app.route("/classify_plant", methods=["POST"])
def classify_plant_endpoint():
    if "herbImage" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files["herbImage"]
    if file.filename == "":
        return jsonify({"error": "Empty file uploaded."}), 400

    try:
        file_path = os.path.join(CONFIG['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        plant_class = classify_plant(file_path)
        os.remove(file_path)

        if not plant_class:
            return jsonify({"error": "Plant classification failed."}), 500

        return jsonify({"plant_class": plant_class})
    except Exception as e:
        logging.error(f"Error in classify_plant_endpoint: {e}")
        return jsonify({"error": "Internal server error."}), 500

# Endpoint for retrieving answers using vector database
@app.route("/get", methods=["POST"])
def get_answer():
    try:
        data = request.get_json()
        user_input = data.get("msg","")
        plant_class = data.get("plant_class","") 

        if not user_input:
            return jsonify({"error": "Missing input"}), 400

        vector_db = load_vector_db()
        if not vector_db:
            raise RuntimeError("Vector database unavailable.")

        chat_model = ChatOllama(model=CONFIG['MODEL_NAME'])
        chain = create_chain(vector_db, chat_model)
        if not chain:
            raise RuntimeError("Failed to create QA chain.")
        # If plant_class is available, include it in the query; otherwise, use user_input alone.   
        query = f"Herb: {plant_class}. {user_input}" if plant_class else user_input
        response = chain.invoke({"query": query})
        return jsonify({"response": response['result']})
    except Exception as e:
        logging.error(f"Error in get_answer: {e}")
        return jsonify({"error": "Internal server error."}), 500

# Run the app
if __name__ == "__main__":
    if not model:
        logging.critical("TensorFlow model is not loaded. Exiting.")
        exit(1)

    os.makedirs(CONFIG['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)