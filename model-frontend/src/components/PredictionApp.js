import React, { useState, useEffect } from 'react';

const PredictionApp = () => {
    const [inputs, setInputs] = useState({});
    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [modelStatus, setModelStatus] = useState(null);
    const [feedback, setFeedback] = useState('');

    // Example feature definitions - adjust based on your model
    const features = [
        { name: 'feature1', label: 'Feature 1', type: 'number' },
        { name: 'feature2', label: 'Feature 2', type: 'select', options: ['option1', 'option2', 'option3'] },
        // Add more features as needed
    ];

    useEffect(() => {
        // Fetch model status periodically
        const fetchStatus = async () => {
            try {
                const response = await fetch('http://localhost:5000/status');
                const data = await response.json();
                setModelStatus(data);
            } catch (err) {
                console.error('Error fetching model status:', err);
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 60000); // Update every minute
        return () => clearInterval(interval);
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

    const handleFeedback = async () => {
        try {
            const response = await fetch('http://localhost:5000/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    features: inputs,
                    label: parseFloat(feedback)
                })
            });

            const data = await response.json();
            if (data.status === 'success') {
                alert('Thank you for your feedback!');
                setFeedback('');
            }
        } catch (err) {
            setError('Failed to submit feedback: ' + err.message);
        }
    };

    return (
        <div className="max-w-4xl mx-auto p-4">
            <div className="bg-white rounded-lg shadow p-6">
                <h1 className="text-2xl font-bold mb-6">Model Prediction Interface</h1>

                <div className="space-y-4">
                    {/* Input Fields */}
                    {features.map(feature => (
                        <div key={feature.name} className="flex flex-col space-y-2">
                            <label className="font-medium">{feature.label}</label>
                            {feature.type === 'select' ? (
                                <select
                                    className="p-2 border rounded"
                                    value={inputs[feature.name] || ''}
                                    onChange={(e) => handleInputChange(feature.name, e.target.value)}
                                >
                                    <option value="">Select...</option>
                                    {feature.options.map(option => (
                                        <option key={option} value={option}>{option}</option>
                                    ))}
                                </select>
                            ) : (
                                <input
                                    type={feature.type}
                                    className="p-2 border rounded"
                                    value={inputs[feature.name] || ''}
                                    onChange={(e) => handleInputChange(feature.name, e.target.value)}
                                />
                            )}
                        </div>
                    ))}

                    {/* Predict Button */}
                    <button
                        className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
                        onClick={handlePredict}
                        disabled={loading}
                    >
                        {loading ? 'Predicting...' : 'Predict'}
                    </button>

                    {/* Prediction Result */}
                    {prediction !== null && (
                        <div className="mt-4 p-4 bg-green-50 rounded">
                            <h3 className="font-medium">Prediction Result:</h3>
                            <p>{prediction}</p>

                            {/* Feedback Section */}
                            <div className="mt-4">
                                <h4 className="font-medium">Provide Feedback</h4>
                                <input
                                    type="number"
                                    className="p-2 border rounded mt-2"
                                    value={feedback}
                                    onChange={(e) => setFeedback(e.target.value)}
                                    placeholder="Enter correct value"
                                />
                                <button
                                    className="ml-2 bg-green-500 text-white p-2 rounded hover:bg-green-600"
                                    onClick={handleFeedback}
                                >
                                    Submit Feedback
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Error Display */}
                    {error && (
                        <div className="p-4 bg-red-50 text-red-700 rounded">
                            {error}
                        </div>
                    )}

                    {/* Model Status */}
                    {modelStatus && (
                        <div className="mt-4 p-4 bg-gray-50 rounded">
                            <h3 className="font-medium">Model Status:</h3>
                            <p>Last Training: {new Date(modelStatus.last_training_time).toLocaleString()}</p>
                            <p>Pending Training Samples: {modelStatus.pending_training_samples}</p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default PredictionApp;