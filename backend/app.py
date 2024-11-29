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

from scipy.stats import rayleigh, norm, lognorm, powerlaw, expon, gamma, cauchy, chi2, uniform

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
    """Preprocess input data with complete feature set."""
    print(f"Input data keys: {data.keys()}")
    
    # Convert input to DataFrame
    processed_data = {}
    for key, value in data.items():
        try:
            if key in ['amt', 'lat', 'long', 'merch_lat', 'merch_long',
                      'city_pop', 'age', 'AreaLand', 'AreaWater',
                      'AnnualPay', 'EmployedNumber']:
                processed_data[key] = float(value) if value != '' else 0.0
            else:
                processed_data[key] = value
        except (TypeError, ValueError):
            processed_data[key] = 0.0

    df = pd.DataFrame([processed_data])
    print(f"Initial DataFrame shape: {df.shape}")
    
    # ZIP code buckets
    zip_bucket_columns = [f'zip_bucket_{bucket}' for bucket in ZIP_BUCKETS]
    df[zip_bucket_columns] = 0
    current_zip_bucket = str(data.get('zip', ''))[:3]
    if current_zip_bucket in ZIP_BUCKETS:
        df[f'zip_bucket_{current_zip_bucket}'] = 1
    df = df.drop('zip', axis=1)
    print(f"After ZIP buckets ({len(zip_bucket_columns)} columns): {df.shape}")
    
    # Calculate geographic features
    df['diff_long'] = df['merch_long'].astype(float) - df['long'].astype(float)
    df['diff_lat'] = df['merch_lat'].astype(float) - df['lat'].astype(float)
    df = df.drop(['merch_long', 'merch_lat', 'long', 'lat'], axis=1)
    
    # Categorical encoding with verification
    encoding_info = [
        ('category', CATEGORIES, 'category_'),
        ('gender', GENDERS, 'gender_'),
        ('state', STATES, 'state_'),
        ('job', JOBS, 'job_'),
        ('merchant', MERCHANTS, 'merchant_fraud_')
    ]
    
    for field, values, prefix in encoding_info:
        # Create columns
        columns = [f"{prefix}{val}" for val in values]
        df[columns] = 0
        
        # Set the appropriate value
        if data.get(field):
            col = f"{prefix}{data[field]}"
            if col in df.columns:
                df[col] = 1
            else:
                print(f"WARNING: Value '{data[field]}' not found in {field} list")
        
        print(f"After {field} encoding ({len(values)} columns): {df.shape}")
    
    # Transform numeric columns
    numeric_columns = ['amt', 'city_pop', 'unix_time', 'age', 'AreaLand', 
                      'AreaWater', 'AnnualPay', 'EmployedNumber', 'diff_lat', 'diff_long']
    
    # Ensure all numeric columns exist and are float
    for col in numeric_columns:
        if col not in df.columns:
            print(f"WARNING: Missing numeric column {col}")
            df[col] = 0.0
        else:
            df[col] = df[col].astype(float)
    
    df = transform_data_to_distribution(df, {k: v for k, v in best_fits.items() 
                                           if k in numeric_columns})
    
    # Drop original categorical columns
    df = df.drop(['category', 'gender', 'state', 'job', 'merchant'], axis=1)
    
    # Verify final column count
    expected_features = (
        len(numeric_columns) +    # Numeric features
        len(CATEGORIES) +        # Category one-hot
        len(STATES) +           # State one-hot
        len(GENDERS) +          # Gender one-hot
        len(JOBS) +             # Job one-hot
        len(MERCHANTS) +        # Merchant one-hot
        len(ZIP_BUCKETS)        # ZIP bucket one-hot
    )
    
    print("\nColumn count verification:")
    print(f"Numeric columns: {len(numeric_columns)}")
    print(f"Category columns: {len([c for c in df.columns if c.startswith('category_')])}")
    print(f"Gender columns: {len([c for c in df.columns if c.startswith('gender_')])}")
    print(f"State columns: {len([c for c in df.columns if c.startswith('state_')])}")
    print(f"Job columns: {len([c for c in df.columns if c.startswith('job_')])}")
    print(f"Merchant columns: {len([c for c in df.columns if c.startswith('merchant_fraud_')])}")
    print(f"ZIP bucket columns: {len([c for c in df.columns if c.startswith('zip_bucket_')])}")
    print(f"\nExpected total: {expected_features}")
    print(f"Actual total: {df.shape[1]}")
    
    if expected_features != df.shape[1]:
        print("\nWARNING: Column count mismatch!")
        # Print sets of column prefixes to find duplicates
        prefixes = {col.split('_')[0] for col in df.columns}
        print(f"Column prefixes found: {prefixes}")
        
        # Check for duplicate columns
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            print(f"Duplicate columns found: {dupes}")
    
    result = df.astype(np.float32).values
    return result

def verify_model_loading():
    """Verify the model is loaded correctly and has required attributes."""
    try:
        # Try to load the model
        model = keras.models.load_model('bestModel_saveable.keras')
        
        # Check if it's a Sequential model
        print(f"Model type: {type(model)}")
        
        # Print model summary
        print("\nModel summary:")
        model.summary()
        
        return True, "Model verified successfully"
    except Exception as e:
        return False, f"Model verification failed: {str(e)}"

def load_model():
    """Load the model and verify feature dimensions."""
    global model
    
    # First verify the model
    success, message = verify_model_loading()
    if not success:
        print(f"ERROR: {message}")
        raise RuntimeError(message)
        
    model = keras.models.load_model('bestModel_saveable.keras')
    
    # Calculate expected number of features
    numeric_cols = ['amt', 'city_pop', 'unix_time', 'age', 'AreaLand', 
                   'AreaWater', 'AnnualPay', 'EmployedNumber', 'diff_lat', 'diff_long']
    
    expected_features = (
        len(numeric_cols) +      # Numeric features
        len(CATEGORIES) +        # Category one-hot
        len(STATES) +           # State one-hot
        len(GENDERS) +          # Gender one-hot
        len(JOBS) +             # Job one-hot
        len(MERCHANTS) +        # Merchant one-hot
        len(ZIP_BUCKETS)        # ZIP bucket one-hot
    )
    
    # Get actual model input shape
    try:
        # Try different ways to get the input shape
        if hasattr(model, 'input_shape'):
            model_features = model.input_shape[-1]
        elif hasattr(model.layers[0], 'input_shape'):
            model_features = model.layers[0].input_shape[-1]
        elif hasattr(model, 'layers') and hasattr(model.layers[0], 'weights'):
            model_features = model.layers[0].weights[0].shape[0]
        else:
            raise ValueError("Could not determine model input shape")
            
        print("\nModel Feature Verification:")
        print(f"Model expects {model_features} features")
        print(f"Current configuration provides {expected_features} features")
        print("\nBreakdown:")
        print(f"Numeric features: {len(numeric_cols)}")
        print(f"Categories: {len(CATEGORIES)}")
        print(f"States: {len(STATES)}")
        print(f"Genders: {len(GENDERS)}")
        print(f"Jobs: {len(JOBS)}")
        print(f"Merchants: {len(MERCHANTS)}")
        print(f"ZIP buckets: {len(ZIP_BUCKETS)}")
        
        if model_features != expected_features:
            print("\nWARNING: Feature count mismatch!")
            print(f"Model expects {model_features} features but current configuration will provide {expected_features} features")
            print("This may cause prediction errors if the mismatch isn't resolved.")
            
    except Exception as e:
        print(f"WARNING: Could not verify input dimensions: {str(e)}")
        
    

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
        
        # Get model information
        print(f"\nModel information:")
        # Get expected input shape from the first layer
        expected_features = model.layers[0].input_shape[-1]  # Use input_shape property
        print(f"Expected features: {expected_features}")
        print(f"Processed input shape: {processed_input.shape}")
        
        # Verify input dimensions
        actual_features = processed_input.shape[-1]
        
        if expected_features != actual_features:
            raise ValueError(f"Input dimension mismatch. Model expects {expected_features} features but got {actual_features} features.\n" + 
                           f"This likely means there's a mismatch in the number of categorical variables between training and prediction.")
        
        # Convert to tensor and predict
        with model_lock:
            tensor_input = tf.convert_to_tensor(processed_input, dtype=tf.float32)
            prediction = model.predict(tensor_input, verbose=0)  # Add verbose=0 to reduce output noise
        
        return jsonify({
            'prediction': prediction.tolist()[0],
            'status': 'success'
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()  # Print full stack trace
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