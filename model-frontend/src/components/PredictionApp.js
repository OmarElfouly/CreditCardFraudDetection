import React, { useState, useEffect } from 'react';
//import Axios from "axios"; // Import Axios or use Fetch.
import area_data from '../area_data.json';
import employment_data from '../employment_data.json';
const PredictionApp = () => {
    const [inputs, setInputs] = useState({
        amt: '',
        lat: '',
        long: '',
        merch_lat: '',
        merch_long: '',
        city_pop: '',
        unix_time: Math.floor(Date.now() / 1000),
        zipcode: '',  // For internal use only
        zip_bucket: '',
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
        zip_buckets: []
    });

    const [areaData, setAreaData] = useState([]);
    const [employmentData, setEmploymentData] = useState([]);

    const [prediction, setPrediction] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);


    const [validBucket, setValidBucket] = useState(false);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:5000/get_metadata');
                const data = await response.json();
                setMetadata(data);

                // const areaContent = await fs.readFile('area_data.json', { encoding: 'utf8' });
                setAreaData(area_data);
                //let areaContent = area_data;
                //console.log('Area content:', areaContent);
                //const areaJSON = JSON.parse(areaContent);
                //setAreaData(areaJSON);    

                //const employmentContent = await fs.readFile('employment_data.json', { encoding: 'utf8' });
                setEmploymentData(employment_data);
                //let employmentContent = employment_data;
                //const employmentJSON = JSON.parse(employmentContent);
                //setEmploymentData(employmentJSON);
            } catch (err) {
                setError('Failed to fetch data: ' + err.message);
            }
        };

        fetchData();
    }, []);

    const determineZipBucket = (zipcode) => {
        const zip = parseInt(zipcode);
        //console.log('Zip:', zip);
        if (!zip) return '';

        // for (const bucket of metadata.zip_buckets) {
        //     const [start, end] = bucket.split('-').map(num => parseInt(num));
        //     if (zip >= start && zip <= end) {
        //         console.log('Bucket:', bucket);
        //         return bucket;
        //     }
        // }
        // first 3 digits of zip code is the bucket so attempt to find that in buckets, if not found show an error message
        let bucketAttempt = zip.toString().substring(0, 3);
        if (metadata.zip_buckets.includes(bucketAttempt)) {
            setValidBucket(true);
            return bucketAttempt;
        } else {
            return 'Zip code not found in bucket list';
        }
    };

    const handleZipCodeChange = (zipcode) => {
        setValidBucket(false);
        const newInputs = { ...inputs, zipcode };

        // Find zip bucket
        const zip_bucket = determineZipBucket(zipcode);
        newInputs.zip_bucket = zip_bucket;

        //console.log('Zip bucket:', zip_bucket);

        // Find area data - skip header row by checking length
        console.log('Area data:', areaData);
        const areaInfo = areaData.find(row => row[3] === zipcode);
        if (areaInfo) {
            newInputs.AreaLand = areaInfo[1];
            newInputs.AreaWater = areaInfo[2];
        }
        console.log('Area water:', newInputs.AreaWater);
        console.log('Area land:', newInputs.AreaLand);

        // Find employment data - skip header row by checking length
        const employmentInfo = employmentData.find(row => row.length === 3 && row[2] === zipcode);
        if (employmentInfo) {
            newInputs.AnnualPay = employmentInfo[0];
            newInputs.EmployedNumber = employmentInfo[1];
        }

        setInputs(newInputs);
    };

    const handleInputChange = (name, value) => {
        console.log(name, value);
        if (name === 'zipcode') {
            handleZipCodeChange(value);
        } else {
            setInputs(prev => ({
                ...prev,
                [name]: value
            }));
        }
    };

    const handlePredict = async () => {
        setLoading(true);
        setError(null);
        try {
            // Create a copy of inputs without the zipcode field
            const { zipcode, ...predictionData } = inputs;

            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(predictionData)
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
                            <label className="block text-sm font-medium text-gray-700">ZIP Code</label>
                            <input
                                type="text"
                                className="w-full p-2 border rounded"
                                value={inputs.zipcode}
                                onChange={(e) => handleInputChange('zipcode', e.target.value)}
                                pattern="[0-9]{5}"
                                maxLength="5"
                                placeholder="Enter 5-digit ZIP code"
                            />
                            {inputs.zip_bucket && (
                                <span className="text-xs text-gray-500">ZIP Bucket: {inputs.zip_bucket}</span>
                            )}
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">City Population</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={inputs.city_pop}
                                onChange={(e) => handleInputChange('city_pop', e.target.value)}
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Transaction Time (Unix)</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={inputs.unix_time}
                                onChange={(e) => handleInputChange('unix_time', e.target.value)}
                            />
                            <span className="text-xs text-gray-500">Current Unix timestamp: {Math.floor(Date.now() / 1000)}</span>
                        </div>

                        {/* Category and Merchant dropdowns */}
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

                    {/* Personal Details */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Personal Details</h3>
                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Age</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                placeholder="Age"
                                value={inputs.age}
                                onChange={(e) => handleInputChange('age', e.target.value)}
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Gender</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={inputs.gender}
                                onChange={(e) => handleInputChange('gender', e.target.value)}
                            >
                                <option value="">Select Gender</option>
                                {metadata.genders.map(gender => (
                                    <option key={gender} value={gender}>{gender}</option>
                                ))}
                            </select>
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Job</label>
                            <select
                                className="w-full p-2 border rounded"
                                value={inputs.job}
                                onChange={(e) => handleInputChange('job', e.target.value)}
                            >
                                <option value="">Select Job</option>
                                {metadata.jobs.map(job => (
                                    <option key={job} value={job}>{job}</option>
                                ))}
                            </select>
                        </div>
                    </div>

                    {/* Demographics Section */}
                    <div className="space-y-4">
                        <h3 className="text-lg font-semibold">Demographics</h3>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Annual Pay</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={inputs.AnnualPay}
                                onChange={(e) => handleInputChange('AnnualPay', e.target.value)}
                                placeholder="Annual Pay"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Employed Number</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={inputs.EmployedNumber}
                                onChange={(e) => handleInputChange('EmployedNumber', e.target.value)}
                                placeholder="Employed Number"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Area Land</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={inputs.AreaLand}
                                onChange={(e) => handleInputChange('AreaLand', e.target.value)}
                                placeholder="Area Land"
                            />
                        </div>

                        <div className="space-y-2">
                            <label className="block text-sm font-medium text-gray-700">Area Water</label>
                            <input
                                type="number"
                                className="w-full p-2 border rounded"
                                value={inputs.AreaWater}
                                onChange={(e) => handleInputChange('AreaWater', e.target.value)}
                                placeholder="Area Water"
                            />
                        </div>
                    </div>
                </div>

                <button
                    className="w-full mt-6 bg-blue-500 text-white p-2 rounded hover:bg-blue-600 disabled:bg-gray-400"
                    onClick={handlePredict}
                    disabled={loading || !validBucket}
                >
                    {loading ? 'Predicting...' : 'Predict'}
                </button>

                {prediction !== null && (
                    <div className="mt-4 p-4 bg-green-50 rounded">
                        <h3 className="font-medium">Prediction Result:</h3>
                        <p>Fraud: {(prediction)}</p>
                    </div>
                )}

                {error && (
                    <div className="mt-4 p-4 bg-red-50 text-red-700 rounded">
                        {error}
                    </div>
                )}
            </div>
        </div >
    );
};

export default PredictionApp;