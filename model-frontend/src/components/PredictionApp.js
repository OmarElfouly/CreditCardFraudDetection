import React, { useState, useEffect } from 'react';

const PredictionApp = () => {
    const [inputs, setInputs] = useState({
        amt: '',
        lat: '',
        long: '',
        merch_lat: '',
        merch_long: '',
        city_pop: '',
        unix_time: Date.now() / 1000,
        zip_bucket: '',  // Changed from zip to zip_bucket
        age: '',
        AreaLand: '',
        AreaWater: '',
        AnnualPay: '',
        EmployedNumber: '',
        category: '',
        gender: '',
        state: '',
        job: '',
        merchant: ''
    });

    const [metadata, setMetadata] = useState({
        categories: [],
        states: [],
        genders: [],
        jobs: [],
        merchants: [],
        zip_buckets: []  // Added zip_buckets to metadata
    });

    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        // Fetch metadata on component mount
        const fetchMetadata = async () => {
            try {
                const response = await fetch('http://localhost:5000/get_metadata');
                const data = await response.json();
                setMetadata(data);
            } catch (err) {
                setError('Failed to fetch form options: ' + err.message);
            }
        };

        fetchMetadata();
    }, []);

    const handleInputChange = (name, value) => {
        setInputs(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(inputs)
            });

            const data = await response.json();
            if (data.status === 'success') {
                setPrediction(data.prediction);
            } else {
                setError(data.error);
            }
        } catch (err) {
            setError('Failed to get prediction: ' + err.message);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <div className="bg-white rounded-lg shadow p-6">
                <h1 className="text-2xl font-bold mb-6">Fraud Detection Prediction</h1>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Transaction Details */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Transaction Details</h3>
                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Amount</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={inputs.amt}
                                onChange={(e) => handleInputChange('amt', e.target.value)}
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Category</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={inputs.category}
                                onChange={(e) => handleInputChange('category', e.target.value)}
                            >
                                <option value="">Select Category</option>
                                {metadata.categories.map(cat => (
                                    <option key={cat} value={cat}>
                                        {cat.replace('_', ' ').toUpperCase()}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Merchant</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={inputs.merchant}
                                onChange={(e) => handleInputChange('merchant', e.target.value)}
                            >
                                <option value="">Select Merchant</option>
                                {metadata.merchants.map(merchant => (
                                    <option key={merchant} value={merchant}>
                                        {merchant}
                                    </option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Location Details */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Location Details</h3>
                        <div className="grid grid-cols-2 gap-2">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Your Latitude</label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded"
                                    value={inputs.lat}
                                    onChange={(e) => handleInputChange('lat', e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Your Longitude</label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded"
                                    value={inputs.long}
                                    onChange={(e) => handleInputChange('long', e.target.value)}
                                />
                            </div>
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Merchant Latitude</label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded"
                                    value={inputs.merch_lat}
                                    onChange={(e) => handleInputChange('merch_lat', e.target.value)}
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700">Merchant Longitude</label>
                                <input
                                    type="number"
                                    className="w-full p-2 border rounded"
                                    value={inputs.merch_long}
                                    onChange={(e) => handleInputChange('merch_long', e.target.value)}
                                />
                            </div>
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">ZIP Bucket</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={inputs.zip_bucket}
                                onChange={(e) => handleInputChange('zip_bucket', e.target.value)}
                            >
                                <option value="">Select ZIP Bucket</option>
                                {metadata.zip_buckets.map(bucket => (
                                    <option key={bucket} value={bucket}>
                                        {bucket}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">State</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={inputs.state}
                                onChange={(e) => handleInputChange('state', e.target.value)}
                            >
                                <option value="">Select State</option>
                                {metadata.states.map(state => (
                                    <option key={state} value={state}>{state}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Rest of your form components */}
                    {/* ... */}
                </div>

                <button
                    className="w-full mt-6 bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
                    onClick={handlePredict}
                    disabled={loading}
                >
                    {loading ? 'Predicting...' : 'Predict'}
                </button>

                {prediction !== null && (
                    <div className="mt-4 p-4 bg-green-50 rounded">
                        <h3 className="font-medium">Prediction Result:</h3>
                        <p>Fraud Probability: {(prediction * 100).toFixed(2)}%</p>
                    </div>
                )}

                {error && (
                    <div className="mt-4 p-4 bg-red-50 text-red-700 rounded">
                        {error}
                    </div>
                )}
            </div>
        </div>
    );
};

export default PredictionApp;