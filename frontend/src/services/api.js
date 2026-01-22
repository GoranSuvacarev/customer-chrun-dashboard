// frontend/src/services/api.js

import axios from 'axios';

// ============================================================================
// AXIOS INSTANCE CONFIGURATION
// ============================================================================

/**
 * Base API URL - points to your FastAPI backend
 * Backend runs on: http://localhost:8000
 * All API endpoints start with /api
 */
const API_BASE_URL = 'http://localhost:8000/api';

/**
 * Create axios instance with default configuration
 * This instance will be used for all API calls
 */
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds timeout (useful for ML operations)
});

// ============================================================================
// RESPONSE INTERCEPTOR - ERROR HANDLING
// ============================================================================

/**
 * Intercept all responses to handle errors consistently
 * This runs before any .then() or .catch() in your components
 */
apiClient.interceptors.response.use(
  // Success handler - just return the response
  (response) => {
    return response;
  },

  // Error handler - log and format errors
  (error) => {
    // Log error details to console for debugging
    console.error('API Error:', {
      message: error.message,
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      data: error.response?.data,
    });

    // Format error message for display to user
    let errorMessage = 'An unexpected error occurred';

    if (error.response) {
      // Server responded with error status
      errorMessage = error.response.data?.detail ||
                     error.response.data?.message ||
                     `Server error: ${error.response.status}`;
    } else if (error.request) {
      // Request made but no response received
      errorMessage = 'Cannot connect to server. Is the backend running?';
    }

    // Attach formatted message to error object
    error.userMessage = errorMessage;

    // Reject promise so .catch() handlers in components can use it
    return Promise.reject(error);
  }
);

// ============================================================================
// CUSTOMER API - Customer data operations
// ============================================================================

export const customerAPI = {
  /**
   * Get all customers with optional pagination and filtering
   * @param {Object} params - Query parameters
   * @param {number} params.skip - Number of records to skip (default: 0)
   * @param {number} params.limit - Number of records to return (default: 100)
   * @param {string} params.risk_level - Filter by risk level (High/Medium/Low)
   * @returns {Promise} Array of customer objects
   *
   * Example: customerAPI.getCustomers({ skip: 0, limit: 50, risk_level: 'High' })
   */
  getCustomers: async (params = {}) => {
    const response = await apiClient.get('/customers', { params });
    return response.data;
  },

  /**
   * Get a single customer by ID
   * @param {string} customerId - MongoDB ObjectId of the customer
   * @returns {Promise} Single customer object with all details
   *
   * Example: customerAPI.getCustomer('507f1f77bcf86cd799439011')
   */
  getCustomer: async (customerId) => {
    const response = await apiClient.get(`/customers/${customerId}`);
    return response.data;
  },

  /**
   * Get total count of customers (for pagination)
   * @returns {Promise} Object with count: number
   */
  getCustomerCount: async () => {
    const response = await apiClient.get('/customers/count');
    return response.data;
  },
};

// ============================================================================
// UPLOAD API - File upload operations
// ============================================================================

export const uploadAPI = {
  /**
   * Upload CSV file to backend
   * @param {File} file - The CSV file from input element
   * @returns {Promise} Object with upload results (customers_created, etc.)
   *
   * IMPORTANT: This uses FormData, not JSON
   * The Content-Type header is automatically set to multipart/form-data
   *
   * Example usage in component:
   *   const file = event.target.files[0];
   *   const result = await uploadAPI.uploadCSV(file);
   */
  uploadCSV: async (file) => {
    // Create FormData object to send file
    const formData = new FormData();
    formData.append('file', file);

    // Send with special config for file upload
    const response = await apiClient.post('/upload/csv', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      // Track upload progress (optional - can add onUploadProgress callback)
      timeout: 60000, // 60 seconds for large files
    });

    return response.data;
  },
};

// ============================================================================
// ML API - Machine Learning operations
// ============================================================================

export const mlAPI = {
  /**
   * Train the churn prediction model
   * This can take 30-60 seconds depending on data size
   * @returns {Promise} Training results (accuracy, ROC-AUC, model_id, etc.)
   *
   * Example: const results = await mlAPI.trainModel();
   */
  trainModel: async () => {
    const response = await apiClient.post('/ml/train');
    return response.data;
  },

  /**
   * Get information about the current trained model
   * @returns {Promise} Model metadata (accuracy, features, training date, etc.)
   *
   * Example: const info = await mlAPI.getModelInfo();
   */
  getModelInfo: async () => {
    const response = await apiClient.get('/ml/model-info');
    return response.data;
  },

  /**
   * Predict churn risk for a single customer
   * @param {string} customerId - MongoDB ObjectId of the customer
   * @returns {Promise} Prediction object (risk_level, churn_probability, confidence)
   *
   * Example: const prediction = await mlAPI.predictCustomer('507f1f77bcf86cd799439011');
   */
  predictCustomer: async (customerId) => {
    const response = await apiClient.post(`/ml/predict/${customerId}`);
    return response.data;
  },

  /**
   * Predict churn risk for all customers in database
   * This can take 10-30 seconds for large datasets
   * @returns {Promise} Batch prediction results (predictions_made, model_id, timestamp)
   *
   * Example: const results = await mlAPI.predictAll();
   */
  predictAll: async () => {
    const response = await apiClient.post('/ml/predict-all', {}, {
      timeout: 600000
    });
    return response.data;
  },
};

// ============================================================================
// ANALYTICS API - Dashboard statistics and metrics
// ============================================================================

export const analyticsAPI = {
  /**
   * Get overview statistics for dashboard
   */
  getOverview: async () => {
    const response = await apiClient.get('/analytics/overview');
    return response.data;
  },

  /**
   * Get distribution of risk levels (for pie chart)
   */
  getRiskDistribution: async () => {
    const response = await apiClient.get('/analytics/risk-distribution');
    return response.data;
  },

  /**
   * Get feature importance from ML model
   */
  getFeatureImportance: async () => {
    const response = await apiClient.get('/analytics/feature-importance');
    return response.data;
  },

  /**
   * Get churn metrics by customer segments
   */
  getChurnBySegment: async () => {
    const response = await apiClient.get('/analytics/churn-by-segment');
    return response.data;
  },

  /**
   * Get churn rate by contract type  ← ADD THIS METHOD
   */
  getChurnByContract: async () => {
    const response = await apiClient.get('/analytics/churn-by-contract');
    return response.data;
  },

  /**
   * Get churn rate by tenure brackets  ← BONUS: You have this endpoint too
   */
  getChurnByTenure: async () => {
    const response = await apiClient.get('/analytics/churn-by-tenure');
    return response.data;
  },
};

// ============================================================================
// EXPORT DEFAULT API OBJECT (alternative usage pattern)
// ============================================================================

/**
 * Export all APIs as a single object (optional - use named exports above instead)
 * This allows: import api from './api'; api.customerAPI.getCustomers();
 */
const api = {
  customer: customerAPI,
  upload: uploadAPI,
  ml: mlAPI,
  analytics: analyticsAPI,
};

export default api;