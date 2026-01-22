import { useState } from 'react';
import { uploadAPI, mlAPI } from '../services/api';
import { Upload as UploadIcon, Brain, Zap, CheckCircle, AlertCircle, Loader } from 'lucide-react';

// ============================================================================
// UPLOAD PAGE COMPONENT
// ============================================================================

const Upload = () => {
  // File upload state
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadSuccess, setUploadSuccess] = useState(null);
  const [uploadError, setUploadError] = useState(null);

  // Model training state
  const [training, setTraining] = useState(false);
  const [trainingSuccess, setTrainingSuccess] = useState(null);
  const [trainingError, setTrainingError] = useState(null);

  // Prediction state
  const [predicting, setPredicting] = useState(false);
  const [predictionSuccess, setPredictionSuccess] = useState(null);
  const [predictionError, setPredictionError] = useState(null);

  // Drag and drop state
  const [isDragging, setIsDragging] = useState(false);

  // ============================================================================
  // FILE UPLOAD HANDLERS
  // ============================================================================

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file && file.name.endsWith('.csv')) {
      setSelectedFile(file);
      setUploadSuccess(null);
      setUploadError(null);
    } else {
      setUploadError('Please select a valid CSV file');
    }
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);

    const file = event.dataTransfer.files[0];
    if (file && file.name.endsWith('.csv')) {
      setSelectedFile(file);
      setUploadSuccess(null);
      setUploadError(null);
    } else {
      setUploadError('Please drop a valid CSV file');
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setUploadError('Please select a file first');
      return;
    }

    setUploading(true);
    setUploadError(null);
    setUploadSuccess(null);

    try {
      const response = await uploadAPI.uploadCSV(selectedFile);
      console.log('Upload response:', response);

      setUploadSuccess({
        message: response.message || 'Upload successful!',
        customersCreated: response.customers_created || response.data?.customers_created || 0,
        customersUpdated: response.customers_updated || response.data?.customers_updated || 0,
      });

      // Clear selected file after successful upload
      setSelectedFile(null);
    } catch (error) {
      console.error('Upload error:', error);
      setUploadError(error.userMessage || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  // ============================================================================
  // MODEL TRAINING HANDLERS
  // ============================================================================

  const handleTrainModel = async () => {
    setTraining(true);
    setTrainingError(null);
    setTrainingSuccess(null);

    try {
      const response = await mlAPI.trainModel();
      console.log('Training response:', response);

      // Extract metrics from response
      const accuracy = response.metrics?.accuracy || response.test_accuracy || 0;
      const rocAuc = response.metrics?.roc_auc || response.roc_auc || 0;

      setTrainingSuccess({
        message: 'Model trained successfully!',
        accuracy: accuracy.toFixed(2), // Convert to percentage
        rocAuc: rocAuc.toFixed(3),
        trainingTime: response.training_time_seconds || response.training_time || 0,
      });
    } catch (error) {
      console.error('Training error:', error);
      setTrainingError(error.userMessage || 'Failed to train model');
    } finally {
      setTraining(false);
    }
  };

  // ============================================================================
  // PREDICTION HANDLERS
  // ============================================================================

  const handlePredictAll = async () => {
    // Confirmation dialog
    const confirmed = window.confirm(
      'This will generate predictions for all customers in the database. This may take a while. Continue?'
    );

    if (!confirmed) return;

    setPredicting(true);
    setPredictionError(null);
    setPredictionSuccess(null);

    try {
      const response = await mlAPI.predictAll();
      console.log('Prediction response:', response);

      setPredictionSuccess({
        message: response.message || 'Predictions generated successfully!',
        predictionsCount: response.summary?.successful_predictions || 0,
        modelId: response.model_info?.model_id || 'N/A',
        processingTime: response.summary?.processing_time_seconds || 0,
      });
    } catch (error) {
      console.error('Prediction error:', error);
      setPredictionError(error.userMessage || 'Failed to generate predictions');
    } finally {
      setPredicting(false);
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-5xl mx-auto">

      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Upload & Train</h1>
        <p className="text-gray-600">
          Upload customer data, train the ML model, and generate predictions
        </p>
      </div>

      {/* Instructions Box */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-semibold text-blue-900 mb-3">Workflow Instructions</h2>
        <ol className="space-y-2 text-blue-800">
          <li className="flex items-start">
            <span className="font-bold mr-2">1.</span>
            <span><strong>Upload CSV:</strong> Upload your customer data file (must be .csv format)</span>
          </li>
          <li className="flex items-start">
            <span className="font-bold mr-2">2.</span>
            <span><strong>Train Model:</strong> Train the machine learning model on your data (takes 30-60 seconds)</span>
          </li>
          <li className="flex items-start">
            <span className="font-bold mr-2">3.</span>
            <span><strong>Generate Predictions:</strong> Apply the trained model to all customers to predict churn risk</span>
          </li>
        </ol>
      </div>

      {/* File Upload Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <div className="flex items-center mb-4">
          <UploadIcon className="w-6 h-6 text-blue-600 mr-2" />
          <h2 className="text-xl font-bold text-gray-900">Step 1: Upload Data</h2>
        </div>

        {/* Drag & Drop Area */}
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            isDragging
              ? 'border-blue-500 bg-blue-50'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          <UploadIcon className="w-12 h-12 text-gray-400 mx-auto mb-4" />

          {selectedFile ? (
            <div className="mb-4">
              <p className="text-gray-700 font-medium mb-1">Selected file:</p>
              <p className="text-blue-600">{selectedFile.name}</p>
              <p className="text-sm text-gray-500 mt-1">
                {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
          ) : (
            <div className="mb-4">
              <p className="text-gray-700 mb-2">Drag and drop your CSV file here</p>
              <p className="text-sm text-gray-500">or</p>
            </div>
          )}

          <label className="inline-block">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileSelect}
              className="hidden"
            />
            <span className="cursor-pointer bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors inline-block">
              {selectedFile ? 'Choose Different File' : 'Choose File'}
            </span>
          </label>
        </div>

        {/* Upload Button */}
        <div className="mt-4">
          <button
            onClick={handleUpload}
            disabled={!selectedFile || uploading}
            className={`w-full py-3 rounded-lg font-medium transition-colors ${
              !selectedFile || uploading
                ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            {uploading ? (
              <span className="flex items-center justify-center">
                <Loader className="w-5 h-5 mr-2 animate-spin" />
                Uploading...
              </span>
            ) : (
              'Upload CSV'
            )}
          </button>
        </div>

        {/* Upload Success Message */}
        {uploadSuccess && (
          <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4 flex items-start">
            <CheckCircle className="w-5 h-5 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-green-800 font-medium">{uploadSuccess.message}</p>
              <p className="text-green-700 text-sm mt-1">
                Created: {uploadSuccess.customersCreated} | Updated: {uploadSuccess.customersUpdated}
              </p>
            </div>
          </div>
        )}

        {/* Upload Error Message */}
        {uploadError && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
            <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
            <p className="text-red-800">{uploadError}</p>
          </div>
        )}
      </div>

      {/* Model Training Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <div className="flex items-center mb-4">
          <Brain className="w-6 h-6 text-purple-600 mr-2" />
          <h2 className="text-xl font-bold text-gray-900">Step 2: Train Model</h2>
        </div>

        <p className="text-gray-600 mb-4">
          Train the machine learning model on your uploaded data. This process takes approximately 30-60 seconds.
        </p>

        <button
          onClick={handleTrainModel}
          disabled={training}
          className={`w-full py-3 rounded-lg font-medium transition-colors ${
            training
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-purple-600 text-white hover:bg-purple-700'
          }`}
        >
          {training ? (
            <span className="flex items-center justify-center">
              <Loader className="w-5 h-5 mr-2 animate-spin" />
              Training Model... (this may take up to 60 seconds)
            </span>
          ) : (
            'Train Model'
          )}
        </button>

        {/* Training Success Message */}
        {trainingSuccess && (
          <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-start mb-2">
              <CheckCircle className="w-5 h-5 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
              <p className="text-green-800 font-medium">{trainingSuccess.message}</p>
            </div>
            <div className="ml-8 space-y-1 text-sm text-green-700">
              <p>Accuracy: <strong>{trainingSuccess.accuracy}%</strong></p>
              <p>ROC-AUC: <strong>{trainingSuccess.rocAuc}</strong></p>
              <p>Training Time: <strong>{trainingSuccess.trainingTime.toFixed(2)}s</strong></p>
            </div>
          </div>
        )}

        {/* Training Error Message */}
        {trainingError && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
            <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
            <p className="text-red-800">{trainingError}</p>
          </div>
        )}
      </div>

      {/* Prediction Section */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center mb-4">
          <Zap className="w-6 h-6 text-orange-600 mr-2" />
          <h2 className="text-xl font-bold text-gray-900">Step 3: Generate Predictions</h2>
        </div>

        <p className="text-gray-600 mb-4">
          Apply the trained model to all customers in the database to generate churn risk predictions.
        </p>

        <button
          onClick={handlePredictAll}
          disabled={predicting}
          className={`w-full py-3 rounded-lg font-medium transition-colors ${
            predicting
              ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
              : 'bg-orange-600 text-white hover:bg-orange-700'
          }`}
        >
          {predicting ? (
            <span className="flex items-center justify-center">
              <Loader className="w-5 h-5 mr-2 animate-spin" />
              Generating Predictions...
            </span>
          ) : (
            'Predict All Customers'
          )}
        </button>

        {/* Prediction Success Message */}
        {predictionSuccess && (
          <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4">
            <div className="flex items-start mb-2">
              <CheckCircle className="w-5 h-5 text-green-600 mr-3 flex-shrink-0 mt-0.5" />
              <p className="text-green-800 font-medium">{predictionSuccess.message}</p>
            </div>
            <div className="ml-8 space-y-1 text-sm text-green-700">
              <p>Predictions Generated: <strong>{predictionSuccess.predictionsCount.toLocaleString()}</strong></p>
              <p>Model ID: <strong>{predictionSuccess.modelId}</strong></p>
              <p>Processing Time: <strong>{predictionSuccess.processingTime.toFixed(1)}s</strong></p>
            </div>
          </div>
        )}

        {/* Prediction Error Message */}
        {predictionError && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4 flex items-start">
            <AlertCircle className="w-5 h-5 text-red-600 mr-3 flex-shrink-0 mt-0.5" />
            <p className="text-red-800">{predictionError}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default Upload;