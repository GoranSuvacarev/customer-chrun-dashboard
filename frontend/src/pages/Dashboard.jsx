import { Link } from 'react-router-dom';
import { Users, TrendingDown, AlertTriangle, Target, Upload, BarChart3 } from 'lucide-react';
import { useState, useEffect } from 'react';
import { analyticsAPI } from '../services/api';

// ============================================================================
// METRIC CARD COMPONENT
// ============================================================================

const MetricCard = ({ title, value, icon: Icon, bgColor, textColor, iconColor }) => {
  return (
    <div className={`${bgColor} rounded-lg shadow-sm p-6 border border-gray-200`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <p className="text-sm font-medium text-gray-600 mb-1">{title}</p>
          <p className={`text-3xl font-bold ${textColor}`}>{value}</p>
        </div>
        <div className={`${iconColor} p-3 rounded-lg`}>
          <Icon className="w-8 h-8" />
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// QUICK ACTION BUTTON COMPONENT
// ============================================================================

const QuickActionButton = ({ to, icon: Icon, title, description, color }) => {
  return (
    <Link
      to={to}
      className={`block p-6 bg-white rounded-lg shadow-sm border-2 border-gray-200 hover:border-${color}-500 hover:shadow-md transition-all group`}
    >
      <div className="flex items-start space-x-4">
        <div className={`bg-${color}-50 p-3 rounded-lg group-hover:bg-${color}-100 transition-colors`}>
          <Icon className={`w-6 h-6 text-${color}-600`} />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-gray-900 mb-1">{title}</h3>
          <p className="text-sm text-gray-600">{description}</p>
        </div>
      </div>
    </Link>
  );
};

// ============================================================================
// LOADING SPINNER COMPONENT
// ============================================================================

const LoadingSpinner = () => {
  return (
    <div className="flex items-center justify-center p-8">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    </div>
  );
};

// ============================================================================
// ERROR MESSAGE COMPONENT
// ============================================================================

const ErrorMessage = ({ message, onRetry }) => {
  return (
    <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
      <p className="text-red-800 mb-4">{message}</p>
      <button
        onClick={onRetry}
        className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition-colors"
      >
        Retry
      </button>
    </div>
  );
};

// ============================================================================
// MAIN DASHBOARD COMPONENT
// ============================================================================

const Dashboard = () => {
  // State management
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch overview data from API
  const fetchOverview = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await analyticsAPI.getOverview();

      // Extract data from response.data
      const data = response.data;

      // Set metrics with correctly formatted values
      setMetrics({
        totalCustomers: data.total_customers,
        churnRate: data.churn_rate,  // Already a percentage
        atRisk: data.at_risk_count,
        modelAccuracy: data.model_accuracy * 100,  // Convert decimal to percentage
      });

    } catch (err) {
      console.error('Error fetching overview:', err);
      setError(err.userMessage || 'Failed to load dashboard data');
    } finally {
      setLoading(false);
    }
  };

  // Fetch data on component mount
  useEffect(() => {
    fetchOverview();
  }, []);

  return (
    <div className="p-4 sm:p-6 lg:p-8">

      {/* Page Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Dashboard</h1>
        <p className="text-gray-600">
          Overview of customer churn predictions and key metrics
        </p>
      </div>

      {/* Loading State */}
      {loading && <LoadingSpinner />}

      {/* Error State */}
      {error && !loading && (
        <ErrorMessage message={error} onRetry={fetchOverview} />
      )}

      {/* Metric Cards Grid - Only show when data is loaded */}
      {!loading && !error && metrics && (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">

            <MetricCard
              title="Total Customers"
              value={metrics.totalCustomers.toLocaleString()}
              icon={Users}
              bgColor="bg-blue-50"
              textColor="text-blue-700"
              iconColor="bg-blue-100 text-blue-600"
            />

            <MetricCard
              title="Churn Rate"
              value={`${metrics.churnRate.toFixed(1)}%`}
              icon={TrendingDown}
              bgColor="bg-red-50"
              textColor="text-red-700"
              iconColor="bg-red-100 text-red-600"
            />

            <MetricCard
              title="At Risk Customers"
              value={metrics.atRisk.toLocaleString()}
              icon={AlertTriangle}
              bgColor="bg-orange-50"
              textColor="text-orange-700"
              iconColor="bg-orange-100 text-orange-600"
            />

            <MetricCard
              title="Model Accuracy"
              value={`${metrics.modelAccuracy.toFixed(2)}%`}
              icon={Target}
              bgColor="bg-green-50"
              textColor="text-green-700"
              iconColor="bg-green-100 text-green-600"
            />
          </div>

          {/* Quick Actions Section */}
          <div className="mb-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Quick Actions</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">

              <QuickActionButton
                to="/upload"
                icon={Upload}
                title="Upload Data"
                description="Import new customer data from CSV files"
                color="blue"
              />

              <QuickActionButton
                to="/customers"
                icon={Users}
                title="View Customers"
                description="Browse customer list with risk predictions"
                color="purple"
              />

              <QuickActionButton
                to="/analytics"
                icon={BarChart3}
                title="Analytics"
                description="Explore detailed charts and insights"
                color="green"
              />
            </div>
          </div>

          {/* Recent Activity Section (placeholder for future) */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Activity</h2>
            <div className="text-center py-8 text-gray-500">
              <p className="mb-2">No recent activity</p>
              <p className="text-sm">Activity logs will appear here</p>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default Dashboard;