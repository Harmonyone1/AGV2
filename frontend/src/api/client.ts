/**
 * API client for communicating with AGV2 FastAPI backend
 */
import axios from 'axios';
import type {
  MetricsSummary,
  Position,
  Trade,
  EquityCurvePoint,
  HealthResponse,
} from '../types';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// API endpoints
export const api = {
  // Health check
  async getHealth(): Promise<HealthResponse> {
    const response = await apiClient.get<HealthResponse>('/health');
    return response.data;
  },

  // Dashboard metrics
  async getMetricsSummary(): Promise<MetricsSummary> {
    const response = await apiClient.get<MetricsSummary>('/api/v1/metrics/summary');
    return response.data;
  },

  // Active positions
  async getActivePositions(): Promise<Position[]> {
    const response = await apiClient.get<Position[]>('/api/v1/positions/active');
    return response.data;
  },

  // Recent trades
  async getRecentTrades(limit: number = 10): Promise<Trade[]> {
    const response = await apiClient.get<Trade[]>('/api/v1/trades/recent', {
      params: { limit },
    });
    return response.data;
  },

  // Equity curve
  async getEquityCurve(
    startDate?: string,
    endDate?: string
  ): Promise<EquityCurvePoint[]> {
    const response = await apiClient.get<EquityCurvePoint[]>('/api/v1/equity-curve', {
      params: {
        start_date: startDate,
        end_date: endDate,
      },
    });
    return response.data;
  },
};

export default api;
