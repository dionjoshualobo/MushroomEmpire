/**
 * API Client for Nordic Privacy AI Backend
 * Base URL: http://localhost:8000
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface AnalyzeResponse {
  status: string;
  filename: string;
  dataset_info: {
    rows: number;
    columns: number;
    features: string[];
  };
  model_performance: {
    accuracy: number;
    precision: number;
    recall: number;
    f1_score: number;
  };
  bias_metrics: {
    overall_bias_score: number;
    disparate_impact: Record<string, any>;
    statistical_parity: Record<string, any>;
    violations_detected: any[];
  };
  risk_assessment: {
    overall_risk_score: number;
    privacy_risks: any[];
    ethical_risks: any[];
    compliance_risks: any[];
    data_quality_risks: any[];
  };
  recommendations: string[];
  report_file: string;
  timestamp: string;
}

export interface CleanResponse {
  status: string;
  filename: string;
  dataset_info: {
    original_rows: number;
    original_columns: number;
    cleaned_rows: number;
    cleaned_columns: number;
  };
  gpu_acceleration: {
    enabled: boolean;
    device: string;
  };
  summary: {
    columns_removed: string[];
    columns_anonymized: string[];
    total_cells_affected: number;
  };
  pii_detections: Record<string, {
    action: string;
    entity_types: string[];
    num_affected_rows: number;
    examples: Array<{ before: string; after: string }>;
  }>;
  gdpr_compliance: string[];
  files: {
    cleaned_csv: string;
    audit_report: string;
  };
  timestamp: string;
}

/**
 * Analyze dataset for bias and risk
 */
export async function analyzeDataset(file: File): Promise<AnalyzeResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Analysis failed');
  }

  return response.json();
}

/**
 * Clean dataset - detect and anonymize PII
 */
export async function cleanDataset(file: File): Promise<CleanResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await fetch(`${API_BASE_URL}/api/clean`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Cleaning failed');
  }

  return response.json();
}

/**
 * Download report file
 */
export function getReportUrl(reportPath: string): string {
  return `${API_BASE_URL}${reportPath}`;
}

/**
 * Health check
 */
export async function healthCheck() {
  const response = await fetch(`${API_BASE_URL}/health`);
  return response.json();
}
