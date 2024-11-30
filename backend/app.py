import json
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
        # print(f"Transforming column: {col} to distribution: {fit}")

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
    """Preprocess the input data for prediction."""
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
    
    # ZIP bucket encoding
    zip_bucket_columns = [f'zip_bucket_{bucket}' for bucket in ZIP_BUCKETS]
    df[zip_bucket_columns] = 0
    if data.get('zip_bucket'):  # Using zip_bucket directly
        df[f'zip_bucket_{data["zip_bucket"]}'] = 1
    
    # Calculate geographic features
    df['diff_long'] = df['merch_long'].astype(float) - df['long'].astype(float)
    df['diff_lat'] = df['merch_lat'].astype(float) - df['lat'].astype(float)
    df = df.drop(['merch_long', 'merch_lat', 'long', 'lat'], axis=1)
    
    # One-hot encode other categorical variables
    # Category
    category_columns = [f'category_{cat}' for cat in CATEGORIES]
    df[category_columns] = 0
    if data.get('category'):
        df[f'category_{data["category"]}'] = 1
    
    # Gender
    gender_columns = [f'gender_{g}' for g in GENDERS]
    df[gender_columns] = 0
    if data.get('gender'):
        df[f'gender_{data["gender"]}'] = 1
    
    # State
    state_columns = [f'state_{state}' for state in STATES]
    df[state_columns] = 0
    if data.get('state'):
        df[f'state_{data["state"]}'] = 1
    
    # Job
    job_columns = [f'job_{job}' for job in JOBS]
    df[job_columns] = 0
    if data.get('job'):
        df[f'job_{data["job"]}'] = 1
    
    # Merchant (fixed to remove double 'fraud')
    merchant_columns = [f'merchant_fraud_{m}' for m in MERCHANTS]  # Removed extra 'fraud_'
    df[merchant_columns] = 0
    if data.get('merchant'):
        df[f'merchant_fraud_{data["merchant"]}'] = 1
    
    # Transform numeric columns
    numeric_columns = ['amt', 'city_pop', 'unix_time', 'age', 'AreaLand', 
                      'AreaWater', 'AnnualPay', 'EmployedNumber', 'diff_lat', 'diff_long']
    
    # Ensure all numeric columns exist and are float
    for col in numeric_columns:
        if col not in df.columns:
            print(f"WARNING: Missing numeric column {col}")
            df[col] = 0.0
        df[col] = df[col].astype(float)
    
    df = transform_data_to_distribution(df, {k: v for k, v in best_fits.items() 
                                           if k in numeric_columns})
    
    # Print dimensions for verification
    # print(f"\nFeature counts:")
    # print(f"Numeric features: {len(numeric_columns)}")
    # print(f"Categories: {len(CATEGORIES)}")
    # print(f"States: {len(STATES)}")
    # print(f"Genders: {len(GENDERS)}")
    # print(f"Jobs: {len(JOBS)}")
    # print(f"Merchants: {len(MERCHANTS)}")
    # print(f"ZIP buckets: {len(ZIP_BUCKETS)}")
    # print(f"Total features: {df.shape[1]}")
    
    # Verify totals match expectation (should be 1739)
    expected_total = (len(numeric_columns) + len(CATEGORIES) + len(STATES) + 
                     len(GENDERS) + len(JOBS) + len(MERCHANTS) + len(ZIP_BUCKETS))
    
    # if df.shape[1] != expected_total:
    #     print(f"\nWARNING: Feature count mismatch!")
    #     print(f"Expected total: {expected_total}")
    #     print(f"Actual total: {df.shape[1]}")
    
    # Convert to float32 numpy array
    # result = df.astype(np.float32).values
    return df

def verify_feature_counts():
    """Compare expected column names with actual column names."""
    # Create expected column list
    expected_columns = []
    
    # Add numeric columns
    numeric_columns = ['amt', 'city_pop', 'unix_time', 'age', 'AreaLand', 
                      'AreaWater', 'AnnualPay', 'EmployedNumber', 'diff_lat', 'diff_long']
    expected_columns.extend(numeric_columns)
    
    # Add categorical columns
    expected_columns.extend([f'category_{cat}' for cat in CATEGORIES])
    expected_columns.extend([f'state_{state}' for state in STATES])
    expected_columns.extend([f'gender_{g}' for g in GENDERS])
    expected_columns.extend([f'job_{job}' for job in JOBS])
    expected_columns.extend([f'merchant_fraud_{m}' for m in MERCHANTS])
    expected_columns.extend([f'zip_bucket_{bucket}' for bucket in ZIP_BUCKETS])
    
    # Sort expected columns
    expected_columns = sorted(expected_columns)
    print(f"\nExpected column count: {len(expected_columns)}")
    
    # Create test DataFrame with all columns
    df = pd.DataFrame(columns=expected_columns)
    actual_columns = sorted(df.columns)
    print(f"Actual column count: {len(actual_columns)}")
    
    # Find any duplicate columns
    duplicates = [col for col in actual_columns if actual_columns.count(col) > 1]
    if duplicates:
        print("\nDuplicate columns found:")
        for dup in duplicates:
            print(f"- {dup}")
    
    # Group columns by prefix to check category counts
    prefixes = {}
    for col in actual_columns:
        prefix = col.split('_')[0]
        if prefix not in prefixes:
            prefixes[prefix] = []
        prefixes[prefix].append(col)
    
    # print("\nColumns by prefix:")
    # for prefix, cols in sorted(prefixes.items()):
    #     print(f"{prefix}: {len(cols)} columns")
    #     if len(cols) <= 5:  # Show all columns for small groups
    #         print(f"  {cols}")
    #     else:  # Show first and last few for large groups
    #         print(f"  First few: {cols[:3]}")
    #         print(f"  Last few: {cols[-3:]}")
    
    # Check ZIP buckets specifically
    zip_columns = [col for col in actual_columns if col.startswith('zip_bucket_')]
    print(f"\nZIP bucket columns: {len(zip_columns)}")
    print(f"ZIP buckets in file: {len(ZIP_BUCKETS)}")
    
    # Compare the lists to find any discrepancy
    zip_numbers = set(col.replace('zip_bucket_', '') for col in zip_columns)
    file_numbers = set(ZIP_BUCKETS)
    
    extra_in_columns = zip_numbers - file_numbers
    extra_in_file = file_numbers - zip_numbers
    
    if extra_in_columns:
        print(f"\nExtra ZIP buckets in columns: {sorted(extra_in_columns)}")
    if extra_in_file:
        print(f"Extra ZIP buckets in file: {sorted(extra_in_file)}")

# Call the verification function
verify_feature_counts()

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
    
    # Verify feature counts and column names
    print("\nVerifying feature counts and column names...")
    verify_feature_counts()
    
    # Get model input shape
    # try:
    #     model_features = model.layers[0].input_shape[-1]
    #     print(f"\nModel expects {model_features} features")
    # except Exception as e:
    #     print(f"WARNING: Could not verify input dimensions: {str(e)}")
        
    

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

        print("\n=== Raw Input Data ===")
        print(json.dumps(data, indent=2))

        processed_input = preprocess_input(data)
        
        print("\n=== Processed Input Data ===")
        print("Shape:", processed_input.shape)
        print("\nFirst few columns:")
        print(processed_input.iloc[:, :10])  # Print first 10 columns
        print("\nNumeric columns:")
        numeric_cols = processed_input.select_dtypes(include=[np.number]).columns
        print(numeric_cols.tolist())
        print("\nSample of numeric values:")
        print(processed_input[numeric_cols].head())

        # Get model information
        # print(f"\nModel information:")
        # # Get expected input shape from the first layer
        # # expected_features = model.layers[0].input_shape[-1]  # Use input_shape property
        # # print(f"Expected features: {expected_features}")
        # print(f"Processed input shape: {processed_input.shape}")
        
        # Verify input dimensions
        actual_features = processed_input.shape[-1]
        
        # if expected_features != actual_features:
        #     raise ValueError(f"Input dimension mismatch. Model expects {expected_features} features but got {actual_features} features.\n" + 
        #                    f"This likely means there's a mismatch in the number of categorical variables between training and prediction.")
        
        # Convert to tensor and predict
        with model_lock:
            numeric_columns = processed_input.select_dtypes(include=[np.number]).columns
            tensor_input = tf.convert_to_tensor(processed_input[numeric_columns].astype(np.float32).values, dtype=tf.float32)
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