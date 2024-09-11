from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask-cors
from transformers import pipeline
from apscheduler.schedulers.background import BackgroundScheduler
import logging
import requests

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the text-generation pipeline
model_name = "Tianlin668/MentalBART"
# Use 'text2text-generation' pipeline for BART-based models
text_generator = pipeline("text2text-generation", model=model_name, trust_remote_code=True)

@app.route('/generate', methods=['POST'])
def generate():
    input_text = request.json.get('input')

    # Ensure input is a string and use the pipeline to generate a response
    generated_output = text_generator(input_text, max_length=150, num_return_sequences=1, temperature=1.0, top_k=40, top_p=0.9, repetition_penalty=1.2)

    # Extract generated text
    generated_text = generated_output[0]['generated_text']

    log_safe("Generated response for input", input_text=input_text)
    return jsonify({'text': generated_text})

@app.route('/feedback', methods=['POST'])
def feedback():
    feedback_text = request.json.get('feedback')
    log_safe("User feedback received", feedback=feedback_text)
    # Process and store feedback (e.g., in a database)
    return jsonify({'message': 'Thank you for your feedback!'})

def log_safe(message, **kwargs):
    # Basic logging
    logger.info(f"Safe log: {message} | Details: {kwargs}")

def update_resources():
    response = requests.get('https://api.example.com/mental-health-resources')
    if response.status_code == 200:
        resources = response.json()
        # Update local database or document store with new resources
        log_safe("Resources updated successfully", resource_count=len(resources))

# Background job to update resources
scheduler = BackgroundScheduler()
scheduler.add_job(update_resources, 'interval', days=1)
scheduler.start()

if __name__ == '__main__':
    app.run()  # Run with HTTP for local testing
