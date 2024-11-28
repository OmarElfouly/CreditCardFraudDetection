from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
import numpy as np
import threading
import time
from datetime import datetime
import pickle
import os

app = Flask(__name__)
CORS(app)

# Global variables
model = None
model_lock = threading.Lock()
training_data = []
last_training_time = datetime.now()
RETRAINING_INTERVAL = 3600  # Retrain every hour if new data exists

# Load the model
def load_model():
    global model
    model = keras.models.load_model('model.keras')

# Initialize
load_model()

def preprocess_input(data):
    # TODO: finish this
    # Convert input data to the format expected by the model
    # Adjust this based on your model's requirements
    return np.array([list(data.values())])

def retrain_model():
    global model, training_data, last_training_time
    
    while True:
        time.sleep(RETRAINING_INTERVAL)
        
        if len(training_data) > 100:
            print("Starting model retraining...")
            
            # Create temporary copy of training data
            current_training_data = training_data.copy()
            training_data = []  # Reset the training data list
            
            # Prepare data for training
            X_train = np.array([d['features'] for d in current_training_data])
            y_train = np.array([d['label'] for d in current_training_data])
            
            # Train on new data
            with model_lock:
                try:
                    model.fit(X_train, y_train, epochs=1, batch_size=32)
                    model.save('model.keras')
                    last_training_time = datetime.now()
                    print("Model retrained successfully")
                except Exception as e:
                    print(f"Error during retraining: {e}")

# Start the retraining thread
retraining_thread = threading.Thread(target=retrain_model, daemon=True)
retraining_thread.start()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        processed_input = preprocess_input(data)
        
        with model_lock:
            prediction = model.predict(processed_input)
        
        return jsonify({
            'prediction': prediction.tolist()[0],
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        features = preprocess_input(data['features'])[0]  # Convert to correct format
        label = data['label']
        
        training_data.append({
            'features': features,
            'label': label
        })
        
        return jsonify({
            'status': 'success',
            'message': 'Feedback received for next training cycle'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'success',
        'last_training_time': last_training_time.isoformat(),
        'pending_training_samples': len(training_data)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)