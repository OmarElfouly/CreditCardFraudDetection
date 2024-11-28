import unittest
import json
import numpy as np
from app import app

class TestFraudDetectionAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
    
    def test_predict_endpoint(self):
        # Test data
        test_data = {
            'unit': 0,
            'imp': 0,
            'log': 0,
            'city_pop': 100000,
            'unix_time': 1600000000,
            'merch_lat': 40.7128,
            'merch_long': -74.0060,
            'is_fraud': 0,
            'category_entertainment': 1,
            'category_food_dining': 0,
            'category_gas_transport': 0,
            'category_grocery_net': 0,
            'category_grocery_pos': 0,
            'category_health_fitness': 0,
            'category_kids_pets': 0,
            'category_misc_net': 0,
            'category_misc_pos': 0,
            'category_personal_care': 0,
            'category_shopping_net': 0,
            'category_travel': 0,
            'gender_F': 1,
            'gender_M': 0
        }
        
        response = self.app.post('/predict',
                               data=json.dumps(test_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue('prediction' in data)
        self.assertTrue('success' in data)
        self.assertTrue(data['success'])

    def test_feedback_endpoint(self):
        feedback_data = {
            'features': {
                'unit': 0,
                'imp': 0,
                # ... (same as test_data above)
            },
            'actual_value': 1
        }
        
        response = self.app.post('/feedback',
                               data=json.dumps(feedback_data),
                               content_type='application/json')
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue('success' in data)
        self.assertTrue(data['success'])
