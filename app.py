import requests
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import logging
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
app = Flask(__name__)

# Load your fine-tuned toxicity detection model and tokenizer (replace with your actual path)
MODEL_PATH = "your_fine_tuned_model_path"  # e.g., "toxicity_model/"
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model.eval()  # Set to evaluation mode
except Exception as e:
    logging.error(f"Error loading model: {e}")
    # It's crucial to handle this error appropriately in a production setting.
    # You might want to raise an exception here to prevent the app from starting
    # if the model fails to load.  For this example, we'll keep going, but
    # the prediction endpoint will not work.
    tokenizer = None
    model = None

# --- Configuration ---
MAX_CONTEXT_LENGTH = 512  # Maximum token length for the model
CONTEXT_WINDOW_SIZE = 5  # Number of surrounding messages to consider
API_TIMEOUT = 10  # Timeout for API requests (seconds)
THREAD_POOL_SIZE = 10  # Number of threads for parallel context fetching
PLATFORM_API_URL = "your_social_media_api.com"  # Replace with your platform's API URL

# --- Helper Functions ---

def fetch_data_from_api(endpoint, item_id):
    """
    Fetches data from the platform's API with error handling and logging.
    """
    url = f"{PLATFORM_API_URL}/{endpoint}/{item_id}"
    try:
        response = requests.get(url, timeout=API_TIMEOUT)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching {endpoint} {item_id} from API: {e}")
        return None  # Return None to indicate failure

def get_post_and_context(post_id):
    """
    Retrieves a post and its surrounding context (thread and user info) from the API.
    """
    post_data = fetch_data_from_api("posts", post_id)
    if not post_data:
        return None, [], {}  # Return defaults on failure

    thread_data = fetch_data_from_api("threads", post_data.get('thread_id', ''))
    thread_messages = thread_data.get('messages', []) if thread_data else []

    user_data = fetch_data_from_api("users", post_data.get('user_id', '')) if post_data else {}

    return post_data, thread_messages, user_data

def extract_features_with_context(post_text, surrounding_texts, user_data, platform):
    """
    Extracts features from the post text and contextual information.
    """
    combined_text = " ".join(surrounding_texts + [post_text])
    inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_LENGTH)
    text_features = inputs['input_ids']  # Shape: [1, seq_length]

    context_features = {  # Store as a dictionary for flexibility
        'num_previous_posts': len(surrounding_texts),
        'user_follower_count': user_data.get('follower_count', 0),
        'is_verified_user': int(user_data.get('is_verified', False)),  # Convert bool to int
        'platform_is_twitter': 1 if platform.lower() == 'twitter' else 0,
        # Add more contextual features as needed
    }
    return text_features, context_features

def predict_toxicity(text_features, context_features):
    """
    Predicts the toxicity score of a post using the loaded model.
    """
    if model is None:
        logging.error("Model is not loaded. Cannot predict toxicity.")
        return 0.0  # Or raise an exception, depending on your error handling policy

    with torch.no_grad():
        outputs = model(text_features)
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        toxicity_score = probabilities[0][1].item()  # Probability of being toxic
    return toxicity_score

def process_post(post_id, platform):
    """
    Processes a single post: fetches data, extracts features, predicts toxicity.
    This function is designed to be run in a thread.
    """
    start_time = time()
    post_data, thread_messages, user_data = get_post_and_context(post_id)
    if not post_data:
        logging.warning(f"Failed to process post {post_id}: Unable to retrieve data.")
        return {'post_id': post_id, 'error': 'Failed to retrieve post data'}

    # Get surrounding context
    surrounding_texts = [msg['text'] for msg in thread_messages[-CONTEXT_WINDOW_SIZE:] if 'text' in msg]

    text_features, context_features = extract_features_with_context(
        post_data['text'], surrounding_texts, user_data, platform
    )

    toxicity_score = predict_toxicity(text_features, context_features)
    end_time = time()
    logging.info(f"Processed post {post_id} in {end_time - start_time:.2f} seconds")
    return {
        'post_id': post_id,
        'text': post_data['text'],
        'toxicity_score': toxicity_score,
        'context_features': context_features,  # Include context features in the response
        'user_id': post_data.get('user_id'), #return user_id
    }

# --- API Endpoint (Flask) ---

@app.route('/predict_toxicity', methods=['POST'])
def get_toxicity_prediction():
    """
    API endpoint to predict the toxicity of one or more posts.
    Handles both single and batch requests.
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Missing JSON payload'}), 400

    if isinstance(data, dict) and 'post_id' in data:
        # Single post request
        post_id = data['post_id']
        platform = data.get('platform', 'Generic')  # Default platform
        result = process_post(post_id, platform)
        return jsonify(result), 200
    elif isinstance(data, list):
        # Batch post request
        post_ids = [item.get('post_id') for item in data if isinstance(item, dict) and 'post_id' in item]
        platforms = [item.get('platform', 'Generic') for item in data if isinstance(item, dict)] #get platform, default platform
        if not post_ids:
            return jsonify({'error': 'No valid post_ids found in batch'}), 400

        # Use a thread pool to process posts in parallel
        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
            results = list(executor.map(process_post, post_ids, platforms)) #pass platforms

        return jsonify(results), 200
    else:
        return jsonify({'error': 'Invalid request format.  Expected a dict with "post_id" or a list of dicts with "post_id" and "platform" fields.'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

