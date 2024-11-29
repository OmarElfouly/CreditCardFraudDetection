import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Select, SelectTrigger, SelectValue, SelectContent, SelectItem } from '@/components/ui/select';
import { Input } from '@/components/ui/input';

const PredictionApp = () => {
    const [inputs, setInputs] = useState({
        amt: '',
        lat: '',
        long: '',
        merch_lat: '',
        merch_long: '',
        city_pop: '',
        unix_time: '',
        zip: '',
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
        merchants: []
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
        <Card className="w-full max-w-4xl mx-auto">
            <CardHeader>
                <CardTitle>Fraud Detection Prediction</CardTitle>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-2 gap-4">
                    {/* Transaction Details */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Transaction Details</h3>
                        <Input
                            type="number"
                            placeholder="Amount"
                            value={inputs.amt}
                            onChange={(e) => handleInputChange('amt', e.target.value)}
                        />

                        <Select
                            value={inputs.category}
                            onValueChange={(value) => handleInputChange('category', value)}
                        >
                            <SelectTrigger>
                                <SelectValue placeholder="Category" />
                            </SelectTrigger>
                            <SelectContent>
                                {metadata.categories.map(cat => (
                                    <SelectItem key={cat} value={cat}>
                                        {cat.replace('_', ' ').toUpperCase()}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>

                        <Select
                            value={inputs.merchant}
                            onValueChange={(value) => handleInputChange('merchant', value)}
                        >
                            <SelectTrigger>
                                <SelectValue placeholder="Merchant" />
                            </SelectTrigger>
                            <SelectContent>
                                {metadata.merchants.map(merchant => (
                                    <SelectItem key={merchant} value={merchant}>
                                        {merchant}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Location Details */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Location Details</h3>
                        <div className="grid grid-cols-2 gap-2">
                            <Input
                                type="number"
                                placeholder="Latitude"
                                value={inputs.lat}
                                onChange={(e) => handleInputChange('lat', e.target.value)}
                            />
                            <Input
                                type="number"
                                placeholder="Longitude"
                                value={inputs.long}
                                onChange={(e) => handleInputChange('long', e.target.value)}
                            />
                        </div>

                        <Input
                            type="text"
                            placeholder="ZIP Code"
                            value={inputs.zip}
                            onChange={(e) => handleInputChange('zip', e.target.value)}
                        />

                        <Select
                            value={inputs.state}
                            onValueChange={(value) => handleInputChange('state', value)}
                        >
                            <SelectTrigger>
                                <SelectValue placeholder="State" />
                            </SelectTrigger>
                            <SelectContent>
                                {metadata.states.map(state => (
                                    <SelectItem key={state} value={state}>
                                        {state}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Personal Details */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Personal Details</h3>
                        <Input
                            type="number"
                            placeholder="Age"
                            value={inputs.age}
                            onChange={(e) => handleInputChange('age', e.target.value)}
                        />

                        <Select
                            value={inputs.gender}
                            onValueChange={(value) => handleInputChange('gender', value)}
                        >
                            <SelectTrigger>
                                <SelectValue placeholder="Gender" />
                            </SelectTrigger>
                            <SelectContent>
                                {metadata.genders.map(gender => (
                                    <SelectItem key={gender} value={gender}>
                                        {gender}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>

                        <Select
                            value={inputs.job}
                            onValueChange={(value) => handleInputChange('job', value)}
                        >
                            <SelectTrigger>
                                <SelectValue placeholder="Job" />
                            </SelectTrigger>
                            <SelectContent>
                                {metadata.jobs.map(job => (
                                    <SelectItem key={job} value={job}>
                                        {job}
                                    </SelectItem>
                                ))}
                            </SelectContent>
                        </Select>
                    </div>

                    {/* Additional Details */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Additional Details</h3>
                        <Input
                            type="number"
                            placeholder="Annual Pay"
                            value={inputs.AnnualPay}
                            onChange={(e) => handleInputChange('AnnualPay', e.target.value)}
                        />

                        <Input
                            type="number"
                            placeholder="Employed Number"
                            value={inputs.EmployedNumber}
                            onChange={(e) => handleInputChange('EmployedNumber', e.target.value)}
                        />

                        <Input
                            type="number"
                            placeholder="Area Land"
                            value={inputs.AreaLand}
                            onChange={(e) => handleInputChange('AreaLand', e.target.value)}
                        />

                        <Input
                            type="number"
                            placeholder="Area Water"
                            value={inputs.AreaWater}
                            onChange={(e) => handleInputChange('AreaWater', e.target.value)}
                        />
                    </div>
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
            </CardContent>
        </Card>
    );
};

export default PredictionApp;