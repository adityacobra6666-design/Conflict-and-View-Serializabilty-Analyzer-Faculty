import axios, { AxiosInstance } from 'axios';
import { AnalysisResult, ExampleSchedule, HistoryItem, QuestionUploadResponse } from '../types';

/**
 * Resolves the backend API base URL:
 * - When VITE_API_URL is configured (e.g. https://backend.onrender.com or https://backend.onrender.com/api),
 *   normalizes it and guarantees that '/api' is appended exactly once.
 * - When VITE_API_URL is omitted or empty (local dev), returns '/api'
 *   which leverages Vite's development proxy (forwarding to http://localhost:8000).
 */
export const getApiBaseUrl = (): string => {
  const envUrl = import.meta.env.VITE_API_URL;
  if (!envUrl || typeof envUrl !== 'string' || !envUrl.trim()) {
    return '/api';
  }
  const cleanUrl = envUrl.trim().replace(/\/+$/, '');
  if (cleanUrl.endsWith('/api')) {
    return cleanUrl;
  }
  return `${cleanUrl}/api`;
};

export const API_BASE = getApiBaseUrl();

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
});

// Response interceptor to format error messages
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      const message = error.response.data?.detail || error.response.data?.message || error.message;
      return Promise.reject(new Error(typeof message === 'string' ? message : JSON.stringify(message)));
    } else if (error.request) {
      return Promise.reject(
        new Error(
          'Unable to reach backend service. If deployed on Render free tier, the instance may be spinning up from sleep (~50s). Please wait a moment and try again.'
        )
      );
    }
    return Promise.reject(error);
  }
);

export const apiService = {
  async checkHealth(): Promise<{ status: string; service: string; version: string }> {
    const response = await apiClient.get('/health');
    return response.data;
  },

  async analyzeSchedule(scheduleText?: string, operations?: any[]): Promise<AnalysisResult> {
    const response = await apiClient.post('/analyze', {
      schedule_text: scheduleText,
      operations: operations
    });
    return response.data;
  },

  async getExamples(): Promise<{ examples: ExampleSchedule[]; count: number }> {
    const response = await apiClient.get('/examples');
    return response.data;
  },

  async getHistory(): Promise<{ history: HistoryItem[]; count: number }> {
    const response = await apiClient.get('/history');
    return response.data;
  },

  async getHistoryById(id: number): Promise<AnalysisResult> {
    const response = await apiClient.get(`/history/${id}`);
    return response.data;
  },

  async deleteHistoryItem(id: number): Promise<{ success: boolean }> {
    const response = await apiClient.delete(`/history/${id}`);
    return response.data;
  },

  async exportPdf(analysisData: AnalysisResult): Promise<Blob> {
    const response = await apiClient.post('/export/pdf', { analysis_data: analysisData }, {
      responseType: 'blob'
    });
    return response.data;
  },

  async exportJson(analysisData: AnalysisResult): Promise<Blob> {
    const response = await apiClient.post('/export/json', analysisData, {
      responseType: 'blob'
    });
    return response.data;
  },

  async exportCsv(analysisData: AnalysisResult): Promise<Blob> {
    const response = await apiClient.post('/export/csv', analysisData, {
      responseType: 'blob'
    });
    return response.data;
  },

  async askAiTutor(
    analysisData?: AnalysisResult | null,
    promptType: string = 'explain_result',
    customQuestion?: string,
    byokConfig?: { provider?: string; baseUrl?: string; apiKey?: string; model?: string },
    messages?: Array<{ role: 'user' | 'assistant'; content: string }>
  ) {
    const response = await apiClient.post('/ai/explain', {
      analysis_data: analysisData || undefined,
      prompt_type: promptType,
      custom_question: customQuestion,
      messages: messages,
      api_key: byokConfig?.apiKey,
      provider: byokConfig?.provider,
      base_url: byokConfig?.baseUrl,
      model: byokConfig?.model
    });
    return response.data;
  },

  async testAiConnection(provider: string, apiKey: string, baseUrl?: string, model?: string) {
    const response = await apiClient.post('/ai/test-connection', {
      provider: provider,
      api_key: apiKey,
      base_url: baseUrl,
      model: model
    });
    return response.data;
  },

  async discoverAiModels(provider: string, apiKey: string, baseUrl?: string) {
    const response = await apiClient.post('/ai/discover-models', {
      provider: provider,
      api_key: apiKey,
      base_url: baseUrl
    });
    return response.data;
  },

  async compareAnalysis(original: AnalysisResult, modified: AnalysisResult) {
    const response = await apiClient.post('/compare', {
      original: original,
      modified: modified
    });
    return response.data;
  },

  async uploadQuestionFile(file: File): Promise<QuestionUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await apiClient.post<QuestionUploadResponse>('/upload/question', formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return response.data;
  }
};

