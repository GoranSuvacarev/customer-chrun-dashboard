import { useState, useEffect } from 'react';
import { customerAPI } from '../services/api';
import { Search, ChevronLeft, ChevronRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

// ============================================================================
// RISK BADGE COMPONENT
// ============================================================================

const RiskBadge = ({ riskLevel }) => {
  // Color mapping for risk levels
  const colors = {
    High: 'bg-red-100 text-red-800 border-red-200',
    Medium: 'bg-yellow-100 text-yellow-800 border-yellow-200',
    Low: 'bg-green-100 text-green-800 border-green-200',
  };

  const colorClass = colors[riskLevel] || 'bg-gray-100 text-gray-800 border-gray-200';

  return (
    <span className={`px-3 py-1 rounded-full text-sm font-medium border ${colorClass}`}>
      {riskLevel}
    </span>
  );
};

// ============================================================================
// LOADING SPINNER
// ============================================================================

const LoadingSpinner = () => {
  return (
    <div className="flex items-center justify-center p-12">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
    </div>
  );
};

// ============================================================================
// MAIN CUSTOMER LIST COMPONENT
// ============================================================================

const CustomerList = () => {
  // State management
  const [customers, setCustomers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalItems, setTotalItems] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [hasPrevious, setHasPrevious] = useState(false);

  // Filter state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [riskFilter, setRiskFilter] = useState('');

  // Page size (items per page)
  const pageSize = 20;

  // Navigation
  const navigate = useNavigate();

  // ============================================================================
  // LOAD CUSTOMERS FUNCTION
  // ============================================================================

  const loadCustomers = async () => {
    setLoading(true);
    setError(null);

    try {
      // Build query parameters
      const params = {
        page: currentPage,
        limit: pageSize,
      };

      // Add search if exists
      if (searchQuery) {
        params.search = searchQuery;
      }

      // Add risk filter if selected
      if (riskFilter) {
        params.risk_level = riskFilter;
      }

      console.log('Loading customers with params:', params);

      // Call API
      const response = await customerAPI.getCustomers(params);

      console.log('API Response:', response);

      // Extract data from response
      const customersData = response.data || [];
      const paginationData = response.pagination || {};

      // Update state
      setCustomers(customersData);
      setTotalPages(paginationData.total_pages || 1);
      setTotalItems(paginationData.total_items || 0);
      setHasNext(paginationData.has_next || false);
      setHasPrevious(paginationData.has_previous || false);

    } catch (err) {
      console.error('Error loading customers:', err);
      setError(err.userMessage || 'Failed to load customers');
    } finally {
      setLoading(false);
    }
  };

  // ============================================================================
  // EFFECTS
  // ============================================================================

  // Load customers when page, search, or filter changes
  useEffect(() => {
    loadCustomers();
  }, [currentPage, searchQuery, riskFilter]);

  // ============================================================================
  // EVENT HANDLERS
  // ============================================================================

  // Handle search form submission
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    setSearchQuery(searchInput);
    setCurrentPage(1); // Reset to first page on new search
  };

  // Handle risk filter change
  const handleRiskFilterChange = (e) => {
    setRiskFilter(e.target.value);
    setCurrentPage(1); // Reset to first page on filter change
  };

  // Handle row click (logs customer ID for now)
  const handleRowClick = (customer) => {
    navigate(`/customers/${customer._id}`);
  };

  // Pagination handlers
  const handlePreviousPage = () => {
    if (hasPrevious) {
      setCurrentPage(prev => prev - 1);
    }
  };

  const handleNextPage = () => {
    if (hasNext) {
      setCurrentPage(prev => prev + 1);
    }
  };

  // ============================================================================
  // RENDER
  // ============================================================================

  return (
    <div className="p-4 sm:p-6 lg:p-8">

      {/* Page Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Customers</h1>
        <p className="text-gray-600">
          Browse customer list with churn risk predictions
        </p>
      </div>

      {/* Search and Filter Bar */}
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
        <div className="flex flex-col md:flex-row gap-4">

          {/* Search Bar */}
          <form onSubmit={handleSearchSubmit} className="flex-1">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search by Customer ID..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              />
            </div>
          </form>

          {/* Risk Filter Dropdown */}
          <div className="w-full md:w-48">
            <select
              value={riskFilter}
              onChange={handleRiskFilterChange}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="">All Risk Levels</option>
              <option value="High">High Risk</option>
              <option value="Medium">Medium Risk</option>
              <option value="Low">Low Risk</option>
            </select>
          </div>
        </div>

        {/* Results count */}
        <div className="mt-3 text-sm text-gray-600">
          {!loading && (
            <span>
              Showing {customers.length} of {totalItems.toLocaleString()} customers
              {searchQuery && ` matching "${searchQuery}"`}
              {riskFilter && ` with ${riskFilter} risk`}
            </span>
          )}
        </div>
      </div>

      {/* Loading State */}
      {loading && <LoadingSpinner />}

      {/* Error State */}
      {error && !loading && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6 text-center">
          <p className="text-red-800 mb-4">{error}</p>
          <button
            onClick={loadCustomers}
            className="bg-red-600 text-white px-4 py-2 rounded hover:bg-red-700 transition-colors"
          >
            Retry
          </button>
        </div>
      )}

      {/* Empty State */}
      {!loading && !error && customers.length === 0 && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-12 text-center">
          <p className="text-gray-600 text-lg mb-2">No customers found</p>
          <p className="text-gray-500 text-sm">
            {searchQuery || riskFilter
              ? 'Try adjusting your search or filters'
              : 'Upload customer data to get started'}
          </p>
        </div>
      )}

      {/* Customer Table */}
      {!loading && !error && customers.length > 0 && (
        <>
          {/* Table Container - Responsive scroll on mobile */}
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50 border-b border-gray-200">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Customer ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Contract
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Tenure
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Monthly Charges
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Risk Level
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Churn Probability
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {customers.map((customer) => (
                    <tr
                      key={customer._id}
                      onClick={() => handleRowClick(customer)}
                      className="hover:bg-gray-50 cursor-pointer transition-colors"
                    >
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {customer.customerID}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {customer.Contract}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {customer.tenure} months
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        ${customer.MonthlyCharges.toFixed(2)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <RiskBadge riskLevel={customer.prediction?.risk_level || 'Unknown'} />
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-600">
                        {customer.prediction?.churn_probability
                          ? `${customer.prediction.churn_probability.toFixed(1)}%`
                          : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Pagination Controls */}
          <div className="mt-6 flex items-center justify-between">

            {/* Page info */}
            <div className="text-sm text-gray-600">
              Page {currentPage} of {totalPages.toLocaleString()}
            </div>

            {/* Navigation buttons */}
            <div className="flex gap-2">
              <button
                onClick={handlePreviousPage}
                disabled={!hasPrevious}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                  hasPrevious
                    ? 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                    : 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
                }`}
              >
                <ChevronLeft className="w-4 h-4" />
                Previous
              </button>

              <button
                onClick={handleNextPage}
                disabled={!hasNext}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg border transition-colors ${
                  hasNext
                    ? 'bg-white border-gray-300 text-gray-700 hover:bg-gray-50'
                    : 'bg-gray-100 border-gray-200 text-gray-400 cursor-not-allowed'
                }`}
              >
                Next
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default CustomerList;