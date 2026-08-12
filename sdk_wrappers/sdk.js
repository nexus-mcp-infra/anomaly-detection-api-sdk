/**
 * Cliente HTTP para AnomalyDetectionApi -- generado deterministicamente
 * desde el contrato OpenAPI real (src/agents/openapi_sdk_generator.py).
 * No edites rutas/params a mano aca -- sdk.py sale del mismo spec,
 * por diseno no puede divergir.
 */

class AnomalyDetectionApiError extends Error {
  constructor(message, statusCode) {
    super(message);
    this.name = 'AnomalyDetectionApiError';
    this.statusCode = statusCode;
  }
}

class AnomalyDetectionApi {
  constructor(apiKey, baseUrl = "https://anomaly-detection-api.railway.app", timeoutMs = 30000) {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.timeoutMs = timeoutMs;
  }

  _headers() {
    const h = { 'Content-Type': 'application/json' };
    if (this.apiKey) h['X-API-Key'] = this.apiKey;
    return h;
  }

  async openMultivariateAnomalySession({ variable_names, sampling_interval_ms, takens_embedding_dim, takens_delay_tau, kde_bandwidth_method, te_window_size, nmi_lag_max }) {
    // Initialize a new anomaly detection session for a set of time-series variables. Use this when starting real-time causal anomaly monitoring across multiple variables (min 2). Do NOT use for single-varia
    // Calls POST /anomaly/session/open
    const payload = { variable_names, sampling_interval_ms, takens_embedding_dim, takens_delay_tau, kde_bandwidth_method, te_window_size, nmi_lag_max };
    const url = `${this.baseUrl}/anomaly/session/open`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new AnomalyDetectionApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async ingestSlidingWindowObservations({ session_id, observations, timestamps_iso, flush_partial_window }) {
    // Push a batch of multivariate observations (sliding window data) into an active anomaly detection session. Provide synchronized timestamps in ISO format. Use this to continuously feed data into the ses
    // Calls POST /anomaly/session/ingest
    const payload = { session_id, observations, timestamps_iso, flush_partial_window };
    const url = `${this.baseUrl}/anomaly/session/ingest`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new AnomalyDetectionApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async streamCausalAnomalyEvents({ session_id, nmi_anomaly_threshold, te_net_flow_min, max_events }) {
    // Retrieve detected causal anomaly events from a session's computation pipeline. Specify NMI anomaly threshold and minimum net Transfer Entropy flow to filter events. Use this to pull detected anomalies
    // Calls POST /anomaly/events/stream
    const payload = { session_id, nmi_anomaly_threshold, te_net_flow_min, max_events };
    const url = `${this.baseUrl}/anomaly/events/stream`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new AnomalyDetectionApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async resolveAnomalyCausalMap({ session_id, event_id, include_reverse_te, min_net_te_weight }) {
    // Resolve the detailed causal graph for a specific anomaly event previously detected. Set include_reverse_te to true if you need both directions of Transfer Entropy; set min_net_te_weight to filter weak
    // Calls POST /anomaly/causal-map/resolve
    const payload = { session_id, event_id, include_reverse_te, min_net_te_weight };
    const url = `${this.baseUrl}/anomaly/causal-map/resolve`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new AnomalyDetectionApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

  async closeAnomalySession({ session_id, export_session_summary }) {
    // Terminate an active session and optionally export a summary of all observations and events. Use this when monitoring ends. Do NOT call on already closed sessions. After closure, the session_id is inva
    // Calls POST /anomaly/session/close
    const payload = { session_id, export_session_summary };
    const url = `${this.baseUrl}/anomaly/session/close`;
    const response = await fetch(url, { method: 'POST', headers: this._headers(), body: JSON.stringify(payload) });
    if (!response.ok) {
      const text = await response.text();
      throw new AnomalyDetectionApiError(`HTTP ${response.status}: ${text.slice(0, 500)}`, response.status);
    }
    return response.json();
  }

}

module.exports = { AnomalyDetectionApi, AnomalyDetectionApiError };