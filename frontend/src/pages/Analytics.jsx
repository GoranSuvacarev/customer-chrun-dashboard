import { useState, useEffect } from 'react';
import { analyticsAPI } from '../services/api';
import { PieChart, Pie, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, Cell, ResponsiveContainer } from 'recharts';
import { Loader, AlertCircle, TrendingUp, Brain, Users } from 'lucide-react';

// ============================================================================
// COLOR CONSTANTS
// ============================================================================

const RISK_COLORS = {
  High: '#EF4444',    // Red
  Medium: '#F59E0B',  // Yellow/Orange
  Low: '#10B981',     // Green
};

const CHART_COLORS = ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B', '#10B981'];

// ============================================================================
// LOADING COMPONENT
// ============================================================================

const LoadingSpinner = () => (
  <div className="flex items-center justify-center p-12">
    <div className="text-center">
      <Loader className="w-12 h-12 text-blue-600 animate-spin mx-auto mb-4" />
      <p className="text-gray-600">Loading analytics data...</p>
    </div>
  </div>
);

// ============================================================================
// ERROR COMPONENT
// ============================================================================

const ErrorMessage = ({ message, onRetry }) => (
  <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
    <AlertCircle className="w-12 h-12 text-red-600 mx-auto mb-4" />
    <p className="text-red-800 mb-4">{message}</p>
    <button
      onClick={onRetry}
      className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition-colors"
    >
      Retry
    </button>
  </div>
);

// ============================================================================
// CHART CARD COMPONENT
// ============================================================================

const ChartCard = ({ title, icon: Icon, children, iconColor }) => (
  <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
    <div className="flex items-center mb-4">
      <Icon className={`w-6 h-6 ${iconColor} mr-2`} />
      <h2 className="text-xl font-bold text-gray-900">{title}</h2>
    </div>
    {children}
  </div>
);

// ============================================================================
// CUSTOM TOOLTIP COMPONENTS
// ============================================================================

const RiskTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
        <p className="font-semibold text-gray-900">{payload[0].name} Risk</p>
        <p className="text-gray-600">
          Count: <span className="font-bold">{payload[0].value.toLocaleString()}</span>
        </p>
        <p className="text-gray-600 text-sm">
          {((payload[0].value / payload[0].payload.total) * 100).toFixed(1)}% of total
        </p>
      </div>
    );
  }
  return null;
};

const FeatureTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
        <p className="font-semibold text-gray-900">{payload[0].payload.name}</p>
        <p className="text-gray-600">
          Importance: <span className="font-bold">{payload[0].value.toFixed(2)}%</span>
        </p>
      </div>
    );
  }
  return null;
};

const ContractTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3">
        <p className="font-semibold text-gray-900">{payload[0].payload.name}</p>
        <p className="text-gray-600">
          Churn Rate: <span className="font-bold">{payload[0].value.toFixed(2)}%</span>
        </p>
        <p className="text-gray-600 text-sm">
          Total: {payload[0].payload.total.toLocaleString()} customers
        </p>
        <p className="text-gray-600 text-sm">
          Churned: {payload[0].payload.churned.toLocaleString()} customers
        </p>
      </div>
    );
  }
  return null;
};

// ============================================================================
// MAIN ANALYTICS COMPONENT
// ============================================================================

const Analytics = () => {
  // State management
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Chart data state
  const [riskData, setRiskData] = useState([]);
  const [featureData, setFeatureData] = useState([]);
  const [contractData, setContractData] = useState([]);

  // ============================================================================
  // LOAD ANALYTICS DATA
  // ============================================================================

  const loadAnalytics = async () => {
    setLoading(true);
    setError(null);

    try {
      // Fetch all analytics data in parallel
      const [riskResponse, featureResponse, contractResponse] = await Promise.all([
        analyticsAPI.getRiskDistribution(),
        analyticsAPI.getFeatureImportance(),
        analyticsAPI.getChurnByContract(),
      ]);

      console.log('Risk Distribution Response:', riskResponse);
      console.log('Feature Importance Response:', featureResponse);
      console.log('Contract Response:', contractResponse);

      // Transform Risk Distribution data for Pie Chart
      const transformedRiskData = riskResponse.data.map(item => ({
        name: item.risk_level,
        value: item.count,
        total: riskResponse.data.reduce((sum, d) => sum + d.count, 0) // For tooltip percentage
      }));

      // Transform Feature Importance data for Bar Chart
      // Convert importance to percentage (0.2298 → 22.98)
      const transformedFeatureData = featureResponse.data.map(item => ({
        name: item.feature,
        value: item.importance * 100, // Convert to percentage
      }));

      // Transform Contract data for Bar Chart
      const transformedContractData = contractResponse.data.map(item => ({
        name: item.contract_type,
        churn_rate: item.churn_rate,
        total: item.total_customers,
        churned: item.churned_customers,
      }));

      // Update state
      setRiskData(transformedRiskData);
      setFeatureData(transformedFeatureData);
      setContractData(transformedContractData);

    } catch (err) {
      console.error('Error loading analytics:', err);
      setError(err.userMessage || 'Failed to load analytics data');
    } finally {
      setLoading(false);
    }
  };

  // Load data on component mount
  useEffect(() => {
    loadAnalytics();
  }, []);

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className="p-4 sm:p-6 lg:p-8">

      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Analytics</h1>
        <p className="text-gray-600">
          Detailed insights and visualizations of customer churn predictions
        </p>
      </div>

      {/* Loading State */}
      {loading && <LoadingSpinner />}

      {/* Error State */}
      {error && !loading && (
        <ErrorMessage message={error} onRetry={loadAnalytics} />
      )}

      {/* Charts Grid */}
      {!loading && !error && (
        <div className="space-y-6">

          {/* Row 1: Risk Distribution Pie Chart (Full Width) */}
          <ChartCard
            title="Risk Distribution"
            icon={TrendingUp}
            iconColor="text-blue-600"
          >
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={riskData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                  outerRadius={120}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {riskData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={RISK_COLORS[entry.name]} />
                  ))}
                </Pie>
                <Tooltip content={<RiskTooltip />} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>

            {/* Summary Stats */}
            <div className="mt-4 grid grid-cols-3 gap-4 border-t pt-4">
              {riskData.map((item) => (
                <div key={item.name} className="text-center">
                  <div className="text-2xl font-bold" style={{ color: RISK_COLORS[item.name] }}>
                    {item.value.toLocaleString()}
                  </div>
                  <div className="text-sm text-gray-600">{item.name} Risk</div>
                </div>
              ))}
            </div>
          </ChartCard>

          {/* Row 2: Two Charts Side by Side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">

            {/* Feature Importance Bar Chart */}
            <ChartCard
              title="Top Features Driving Churn"
              icon={Brain}
              iconColor="text-purple-600"
            >
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={featureData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="name"
                    angle={-45}
                    textAnchor="end"
                    height={100}
                    interval={0}
                  />
                  <YAxis
                    label={{ value: 'Importance (%)', angle: -90, position: 'insideLeft' }}
                    domain={[0, 25]}
                  />
                  <Tooltip content={<FeatureTooltip />} />
                  <Bar dataKey="value" fill="#8B5CF6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            {/* Churn by Contract Bar Chart */}
            <ChartCard
              title="Churn Rate by Contract Type"
              icon={Users}
              iconColor="text-orange-600"
            >
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={contractData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis unit="%" />
                  <Tooltip content={<ContractTooltip />} />
                  <Bar dataKey="churn_rate" fill="#F59E0B" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        </div>
      )}
    </div>
  );
};

export default Analytics;