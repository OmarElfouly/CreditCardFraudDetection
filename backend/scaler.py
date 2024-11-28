import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
from scipy import stats

class DistributionScaler(BaseEstimator, TransformerMixin):
    """
    Scaler that handles the specific distributions from the feature table
    """
    def __init__(self):
        self.feature_distributions = {
            'unit': {'dist': 'Normal', 'loc': -113.998, 'scale': 174.669},
            'imp': {'dist': 'Normal', 'loc': -88.5, 'scale': 136.898},
            'log': {'dist': 'Normal', 'loc': -99.232, 'scale': 147.98},
            'city_pop': {'dist': 'Normal', 'loc': 2.516, 'scale': 4.111},
            'unix_time': {'dist': 'Powerlaw', 'loc': 1.112, 'scale': 192537576},
            'merch_lat': {'dist': 'Normal', 'loc': -5.84, 'scale': 9.122},
            'merch_long': {'dist': 'Normal', 'loc': -108.21, 'scale': 172.331},
            'is_fraud': {'dist': 'Exponential', 'loc': 0, 'scale': 0.006},
            'category_entertainment': {'dist': 'Exponential', 'loc': 0, 'scale': 0.008},
            'category_food_dining': {'dist': 'Exponential', 'loc': 0, 'scale': 0.571},
            'category_gas_transport': {'dist': 'Exponential', 'loc': 0, 'scale': 0.019},
            'category_grocery_net': {'dist': 'Exponential', 'loc': 0, 'scale': 0.596},
            'category_grocery_pos': {'dist': 'Exponential', 'loc': 0, 'scale': 0.887},
            'category_health_fitness': {'dist': 'Exponential', 'loc': 0, 'scale': 0.592},
            'category_kids_pets': {'dist': 'Exponential', 'loc': 0, 'scale': 0.857},
            'category_misc_net': {'dist': 'Exponential', 'loc': 0, 'scale': 0.095},
            'category_misc_pos': {'dist': 'Exponential', 'loc': 0, 'scale': 0.190},
            'category_personal_care': {'dist': 'Exponential', 'loc': 0, 'scale': 0.093},
            'category_shopping_net': {'dist': 'Exponential', 'loc': 0, 'scale': 0.033},
            'category_travel': {'dist': 'Exponential', 'loc': 0, 'scale': 0.106},
            'gender_F': {'dist': 'Gamma', 'loc': 0, 'scale': 0.422, 'a': 0.385},
            'gender_M': {'dist': 'Gamma', 'loc': 0, 'scale': 0.422, 'a': 0.385}
        }
        self.scalers = {}
        
    def fit(self, X, feature_names):
        """
        Fit the scaler to the data
        
        Parameters:
        X: array-like of shape (n_samples, n_features)
        feature_names: list of feature names corresponding to X columns
        """
        X = np.array(X)
        
        for idx, feature_name in enumerate(feature_names):
            dist_info = self.feature_distributions[feature_name]
            feature_data = X[:, idx]
            
            if dist_info['dist'] == 'Normal':
                # For normal distribution, we'll standardize using given parameters
                self.scalers[feature_name] = {
                    'loc': dist_info['loc'],
                    'scale': dist_info['scale']
                }
                
            elif dist_info['dist'] == 'Powerlaw':
                # For powerlaw, we'll use log transformation and scale
                self.scalers[feature_name] = {
                    'loc': dist_info['loc'],
                    'scale': dist_info['scale']
                }
                
            elif dist_info['dist'] == 'Exponential':
                # For exponential, we'll use the given scale parameter (λ = 1/scale)
                self.scalers[feature_name] = {
                    'scale': dist_info['scale']
                }
                
            elif dist_info['dist'] == 'Gamma':
                # For gamma, we'll use both shape (a) and scale parameters
                self.scalers[feature_name] = {
                    'a': dist_info['a'],
                    'scale': dist_info['scale']
                }
        
        return self
    
    def transform(self, X, feature_names):
        """
        Transform the data according to each feature's distribution
        """
        X = np.array(X)
        X_scaled = np.zeros_like(X, dtype=float)
        
        for idx, feature_name in enumerate(feature_names):
            dist_info = self.feature_distributions[feature_name]
            feature_data = X[:, idx]
            scaler_params = self.scalers[feature_name]
            
            if dist_info['dist'] == 'Normal':
                # Standardize normal distributions
                X_scaled[:, idx] = (feature_data - scaler_params['loc']) / scaler_params['scale']
                
            elif dist_info['dist'] == 'Powerlaw':
                # Transform powerlaw using log and scale
                positive_data = np.maximum(feature_data, 1e-10)  # Avoid log(0)
                X_scaled[:, idx] = (np.log(positive_data) - np.log(scaler_params['loc'])) / np.log(scaler_params['scale'])
                
            elif dist_info['dist'] == 'Exponential':
                # Scale exponential using rate parameter
                X_scaled[:, idx] = feature_data / scaler_params['scale']
                
            elif dist_info['dist'] == 'Gamma':
                # Transform gamma using its parameters
                X_scaled[:, idx] = stats.gamma.cdf(feature_data, a=scaler_params['a'], scale=scaler_params['scale'])
        
        return X_scaled
    
    def inverse_transform(self, X_scaled, feature_names):
        """
        Convert scaled values back to original scale
        """
        X_scaled = np.array(X_scaled)
        X = np.zeros_like(X_scaled, dtype=float)
        
        for idx, feature_name in enumerate(feature_names):
            dist_info = self.feature_distributions[feature_name]
            feature_data = X_scaled[:, idx]
            scaler_params = self.scalers[feature_name]
            
            if dist_info['dist'] == 'Normal':
                X[:, idx] = (feature_data * scaler_params['scale']) + scaler_params['loc']
                
            elif dist_info['dist'] == 'Powerlaw':
                X[:, idx] = np.exp(feature_data * np.log(scaler_params['scale']) + np.log(scaler_params['loc']))
                
            elif dist_info['dist'] == 'Exponential':
                X[:, idx] = feature_data * scaler_params['scale']
                
            elif dist_info['dist'] == 'Gamma':
                X[:, idx] = stats.gamma.ppf(feature_data, a=scaler_params['a'], scale=scaler_params['scale'])
        
        return X

# Example usage:
def create_and_save_scaler(X_train, feature_names, save_path='scaler.pkl'):
    """Create, fit, and save the distribution-specific scaler"""
    scaler = DistributionScaler()
    scaler.fit(X_train, feature_names)
    joblib.dump(scaler, save_path)
    return scaler

def load_scaler(path='distribution_scaler.pkl'):
    """Load the saved scaler"""
    return joblib.load(path)

