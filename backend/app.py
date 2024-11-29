import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import keras
import threading
import time
from datetime import datetime
import tensorflow as tf
from scipy.stats import rayleigh, norm, lognorm, powerlaw, expon, gamma

@keras.saving.register_keras_serializable(package='custom_losses')
class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.9, gamma=4.0, **kwargs):  # Updated parameters
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * tf.pow((1 - p_t), self.gamma) * tf.math.log(p_t)
        return tf.reduce_mean(focal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "alpha": self.alpha,
            "gamma": self.gamma
        })
        return config

app = Flask(__name__)
CORS(app)

# Global variables
model = None
model_lock = threading.Lock()
training_data = []
last_training_time = datetime.now()
RETRAINING_INTERVAL = 3600  # Retrain every hour if new data exists

# Load categorical data options from {col_name}.txt files
CATEGORIES = []
with open('category.txt', 'r') as f:
    CATEGORIES = f.read().splitlines()

STATES = []
with open('state.txt', 'r') as f:
    STATES = f.read().splitlines()

GENDERS = []
with open('gender.txt', 'r') as f:
    GENDERS = f.read().splitlines()

JOBS = []
with open('job.txt', 'r') as f:
    JOBS = f.read().splitlines()

MERCHANTS = []
with open('merchant.txt', 'r') as f:
    MERCHANTS = f.read().splitlines()

ZIP_BUCKETS = []
with open('zip_bucket.txt', 'r') as f:
    ZIP_BUCKETS = f.read().splitlines()

# Load the best fits for the numeric columns
import pickle
with open('best_fits.pkl', 'rb') as f:
    best_fits = pickle.load(f)

    
def transform_data_to_distribution(data, best_fits):
    for col, (fit, params) in best_fits.items():
        print(f"Transforming column: {col} to distribution: {fit}")

        if fit == 'rayleigh':
            loc = params['loc']
            scale = params['scale']
            # Transform using Rayleigh inverse cumulative distribution function (CDF)
            data[col] = rayleigh.cdf(data[col], loc=loc, scale=scale)

        elif fit == 'norm':
            loc = params['loc']
            scale = params['scale']
            # Transform using Normal inverse CDF
            data[col] = norm.cdf(data[col], loc=loc, scale=scale)

        elif fit == 'lognorm':
            s = params['s']
            loc = params['loc']
            scale = params['scale']
            # Transform using Log-Normal inverse CDF
            data[col] = lognorm.cdf(data[col], s=s, loc=loc, scale=scale)

        elif fit == 'powerlaw':
            a = params['a']
            loc = params['loc']
            scale = params['scale']
            # Transform using Power-law inverse CDF
            data[col] = powerlaw.cdf(data[col], a, loc=loc, scale=scale)

        elif fit == 'expon':
            loc = params['loc']
            scale = params['scale']
            # Transform using Exponential inverse CDF
            data[col] = expon.cdf(data[col], loc=loc, scale=scale)

        elif fit == 'gamma':
            a = params['a']
            loc = params['loc']
            scale = params['scale']
            # Transform using Gamma inverse CDF
            data[col] = gamma.cdf(data[col], a=a, loc=loc, scale=scale)

        elif fit == 'cauchy':
            loc = params['loc']
            scale = params['scale']
            # Transform using Cauchy inverse CDF
            data[col] = cauchy.cdf(data[col], loc=loc, scale=scale)

        elif fit == 'chi2':
            df = params['df']
            # Transform using Chi-squared inverse CDF
            data[col] = chi2.cdf(data[col], df=df)

        elif fit == 'uniform':
            loc = params['loc']
            scale = params['scale']
            # Transform using Uniform inverse CDF
            data[col] = uniform.cdf(data[col], loc=loc, scale=scale)

    return data


def preprocess_input(data):
    # Convert input to DataFrame for easier processing
    df = pd.DataFrame([data])
    
    # Zip code to zip bucket (first three digits)
    df['zip_bucket'] = df['zip'].astype(str).str[:3]
    df = df.drop('zip', axis=1)
    
    # Calculate diff_long and diff_lat
    df['diff_long'] = df['merch_long'] - df['long']
    df['diff_lat'] = df['merch_lat'] - df['lat']
    df = df.drop(['merch_long', 'merch_lat', 'long', 'lat'], axis=1)
    
    # One-hot encode categorical variables
    # Category
    category_columns = [f'category_{cat}' for cat in CATEGORIES]
    df[category_columns] = 0
    df[f'category_{data["category"]}'] = 1
    
    # Gender
    gender_columns = [f'gender_{g}' for g in GENDERS]
    df[gender_columns] = 0
    df[f'gender_{data["gender"]}'] = 1
    
    # State
    state_columns = [f'state_{state}' for state in STATES]
    df[state_columns] = 0
    df[f'state_{data["state"]}'] = 1
    
    # Job
    job_columns = [f'job_{job}' for job in JOBS]
    df[job_columns] = 0
    df[f'job_{data["job"]}'] = 1
    
    # Merchant (assuming merchant names are in the best_fits dictionary)
    merchant_columns = [f'merchant_fraud_{m}' for m in MERCHANTS]
    df[merchant_columns] = 0
    df[f'merchant_fraud_{data["merchant"]}'] = 1
    
    # Transform numeric columns according to their distributions
    numeric_columns = ['amt', 'city_pop', 'unix_time', 'age', 'AreaLand', 
                      'AreaWater', 'AnnualPay', 'EmployedNumber', 'diff_lat', 'diff_long']
    
    df = transform_data_to_distribution(df, {k: v for k, v in best_fits.items() 
                                           if k in numeric_columns})
    
    # Drop original categorical columns
    df = df.drop(['category', 'gender', 'state', 'job', 'merchant'], axis=1)
    
    return df.values

# Load the model
def load_model():
    global model
    model = keras.models.load_model('bestModel_saveable.keras')

# Initialize
load_model()

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

@app.route('/get_metadata', methods=['GET'])
def get_metadata():
    """Endpoint to get all the possible values for categorical fields"""
    return jsonify({
        'categories': CATEGORIES,
        'states': STATES,
        'genders': GENDERS,
        'jobs': JOBS,
        'merchants': MERCHANTS,
        'zip_buckets': ZIP_BUCKETS
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)