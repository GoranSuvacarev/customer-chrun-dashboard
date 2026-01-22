import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { customerAPI } from '../services/api';
import { ArrowLeft, RefreshCw, User, CreditCard, Wifi, DollarSign, AlertTriangle, TrendingUp } from 'lucide-react';

// ============================================================================
// CIRCULAR PROGRESS COMPONENT (Probability Gauge)
// ============================================================================

const CircularProgress = ({ value, size = 150, strokeWidth = 12 }) => {
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  // Color based on risk level
  let color = '#10B981'; // Green (Low)
  if (value >= 70) color = '#EF4444'; // Red (High)
  else if (value >= 30) color = '#F59E0B'; // Orange (Medium)

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="#E5E7EB"
          strokeWidth={strokeWidth}
          fill="none"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke={color}
          strokeWidth={strokeWidth}
          fill="none"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      {/* Center text */}
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-3xl font-bold text-gray-900">{value.toFixed(1)}%</span>
        <span className="text-sm text-gray-600">Churn Risk</span>
      </div>
    </div>
  );
};

// ============================================================================
// SECTION CARD COMPONENT
// ============================================================================

const SectionCard = ({ title, icon: Icon, children, iconColor = "text-blue-600" }) => (
  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
    <div className="flex items-center mb-4">
      <Icon className={`w-5 h-5 ${iconColor} mr-2`} />
      <h2 className="text-lg font-semibold text-gray-900">{title}</h2>
    </div>
    {children}
  </div>
);

// ============================================================================
// INFO ROW COMPONENT
// ============================================================================

const InfoRow = ({ label, value, highlight = false }) => (
  <div className="flex justify-between py-2 border-b border-gray-100 last:border-b-0">
    <span className="text-gray-600">{label}:</span>
    <span className={`font-medium ${highlight ? 'text-blue-600' : 'text-gray-900'}`}>
      {value}
    </span>
  </div>
);

// ============================================================================
// RISK BADGE COMPONENT
// ============================================================================

const RiskBadge = ({ riskLevel }) => {
  const colors = {
    High: 'bg-red-100 text-red-800 border-red-200',
    Medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    Low: 'bg-green-100 text-green-800 border-green-200',
  };

  return (
    <span className={`px-4 py-2 rounded-full text-lg font-bold border-2 ${colors[riskLevel]}`}>
      {riskLevel} Risk
    </span>
  );
};

// ============================================================================
// LOADING COMPONENT
// ============================================================================

const LoadingSpinner = () => (
  <div className="flex items-center justify-center p-12">
    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
  </div>
);

// ============================================================================
// MAIN CUSTOMER DETAIL COMPONENT
// ============================================================================

const CustomerDetail = () => {
  const { id } = useParams(); // Get customer _id from URL
  const navigate = useNavigate();

  // State
  const [customer, setCustomer] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  // ============================================================================
  // LOAD CUSTOMER DATA
  // ============================================================================

  const loadCustomer = async () => {
    setLoading(true);
    setError(null);

    try {
      const data = await customerAPI.getCustomer(id);
      console.log('Customer data:', data);
      setCustomer(data);
    } catch (err) {
      console.error('Error loading customer:', err);
      setError(err.userMessage || 'Failed to load customer data');
    } finally {
      setLoading(false);
    }
  };

  // Load on mount
  useEffect(() => {
    loadCustomer();
  }, [id]);

  // ============================================================================
  // REFRESH PREDICTION (TODO: Implement backend endpoint)
  // ============================================================================

  const handleRefreshPrediction = async () => {
    setRefreshing(true);
    try {
      // TODO: Call API to regenerate prediction for this customer
      // await mlAPI.predictCustomer(id);
      // await loadCustomer(); // Reload customer data

      alert('Refresh prediction feature - Backend endpoint needed');
    } catch (err) {
      console.error('Error refreshing prediction:', err);
      alert('Failed to refresh prediction');
    } finally {
      setRefreshing(false);
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  if (loading) return <LoadingSpinner />;

  if (error) {
    return (
      <div className="p-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <p className="text-red-800 mb-4">{error}</p>
          <button
            onClick={() => navigate('/customers')}
            className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700"
          >
            Back to Customers
          </button>
        </div>
      </div>
    );
  }

  if (!customer) return null;

  return (
    <div className="p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto">

      {/* Header with Back Button */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <button
            onClick={() => navigate('/customers')}
            className="flex items-center text-gray-600 hover:text-gray-900 mb-2"
          >
            <ArrowLeft className="w-4 h-4 mr-1" />
            Back to Customers
          </button>
          <h1 className="text-3xl font-bold text-gray-900">
            Customer {customer.customerID}
          </h1>
        </div>

        <button
          onClick={handleRefreshPrediction}
          disabled={refreshing}
          className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
          Refresh Prediction
        </button>
      </div>

      {/* Prediction Section - Prominent Display */}
      {customer.prediction && (
        <div className="bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg shadow-sm border border-blue-200 p-8 mb-6">
          <div className="flex flex-col lg:flex-row items-center justify-between gap-8">

            {/* Left: Circular Progress */}
            <div className="flex flex-col items-center">
              <CircularProgress value={customer.prediction.churn_probability} />
              <div className="mt-4">
                <RiskBadge riskLevel={customer.prediction.risk_level} />
              </div>
              <p className="text-sm text-gray-600 mt-2">
                Prediction: {customer.prediction.will_churn ? '⚠️ Will Churn' : '✅ Will Stay'}
              </p>
            </div>

            {/* Right: Top Risk Factors */}
            <div className="flex-1 w-full">
              <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                <AlertTriangle className="w-5 h-5 text-orange-600 mr-2" />
                Top Risk Factors
              </h3>
              <div className="space-y-3">
                {customer.prediction.top_risk_factors?.map((factor, index) => (
                  <div key={index} className="bg-white rounded-lg p-4 border border-gray-200">
                    <div className="flex items-start justify-between mb-1">
                      <span className="font-semibold text-gray-900">
                        {index + 1}. {factor.factor}
                      </span>
                      <span className="text-sm text-gray-600">
                        {(factor.importance * 100).toFixed(1)}% importance
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">{factor.explanation}</p>
                    <p className="text-sm text-blue-600 mt-1">Value: {factor.value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Details Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

        {/* Personal Information */}
        <SectionCard title="Personal Information" icon={User}>
          <div className="space-y-1">
            <InfoRow label="Customer ID" value={customer.customerID} highlight />
            <InfoRow label="Gender" value={customer.gender} />
            <InfoRow label="Senior Citizen" value={customer.SeniorCitizen === 1 ? 'Yes' : 'No'} />
            <InfoRow label="Partner" value={customer.Partner} />
            <InfoRow label="Dependents" value={customer.Dependents} />
          </div>
        </SectionCard>

        {/* Account Details */}
        <SectionCard title="Account Details" icon={CreditCard} iconColor="text-purple-600">
          <div className="space-y-1">
            <InfoRow label="Tenure" value={`${customer.tenure} months`} />
            <InfoRow label="Contract" value={customer.Contract} highlight />
            <InfoRow label="Payment Method" value={customer.PaymentMethod} />
            <InfoRow label="Paperless Billing" value={customer.PaperlessBilling} />
            <InfoRow label="Churn Status" value={customer.Churn} />
          </div>
        </SectionCard>

        {/* Financial Information */}
        <SectionCard title="Financial" icon={DollarSign} iconColor="text-green-600">
          <div className="space-y-1">
            <InfoRow
              label="Monthly Charges"
              value={`$${customer.MonthlyCharges.toFixed(2)}`}
              highlight
            />
            <InfoRow
              label="Total Charges"
              value={`$${customer.TotalCharges.toFixed(2)}`}
            />
            <InfoRow
              label="Average per Month"
              value={`$${(customer.TotalCharges / customer.tenure).toFixed(2)}`}
            />
          </div>
        </SectionCard>

        {/* Services */}
        <SectionCard title="Services" icon={Wifi} iconColor="text-orange-600">
          <div className="space-y-1">
            <InfoRow label="Phone Service" value={customer.services.PhoneService} />
            <InfoRow label="Multiple Lines" value={customer.services.MultipleLines} />
            <InfoRow label="Internet Service" value={customer.services.InternetService} highlight />
            <InfoRow label="Online Security" value={customer.services.OnlineSecurity} />
            <InfoRow label="Online Backup" value={customer.services.OnlineBackup} />
            <InfoRow label="Device Protection" value={customer.services.DeviceProtection} />
            <InfoRow label="Tech Support" value={customer.services.TechSupport} />
            <InfoRow label="Streaming TV" value={customer.services.StreamingTV} />
            <InfoRow label="Streaming Movies" value={customer.services.StreamingMovies} />
          </div>
        </SectionCard>
      </div>

      {/* Prediction Metadata */}
      {customer.prediction && (
        <div className="mt-6 bg-gray-50 rounded-lg p-4 text-sm text-gray-600">
          <p>
            <strong>Prediction Date:</strong>{' '}
            {new Date(customer.prediction.predicted_at).toLocaleString()}
          </p>
          <p>
            <strong>Model ID:</strong> {customer.prediction.model_id}
          </p>
        </div>
      )}
    </div>
  );
};

export default CustomerDetail;