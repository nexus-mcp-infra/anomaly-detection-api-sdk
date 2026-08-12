from fastapi import status
"""
Cliente HTTP para AnomalyDetectionApi -- generado deterministicamente desde
el contrato OpenAPI real (src/agents/openapi_sdk_generator.py). No
edites rutas/params a mano aca -- se regenera en cada build desde
tool_spec; sdk.js sale del mismo spec, por diseno no puede divergir.
"""
from __future__ import annotations

import requests
from typing import Any, Optional


class AnomalyDetectionApiError(Exception):
    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class AnomalyDetectionApi:
    """HTTP client. Base URL real del deploy: https://anomaly-detection-api.railway.app"""

    def __init__(self, api_key: Optional[str] = None, base_url: str = 'https://anomaly-detection-api.railway.app', timeout: float = 30.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'X-API-Key': api_key})

    def open_multivariate_anomaly_session(self, variable_names: list[str], sampling_interval_ms: float, takens_embedding_dim: float, takens_delay_tau: float, kde_bandwidth_method: str, te_window_size: float, nmi_lag_max: float) -> dict:
        """Initialize a new anomaly detection session for a set of time-series variables. Use this when starting real-time causal anomaly monitoring across multiple variables (min 2). Do NOT use for single-variable anomaly detection; use univariate methods instead. Specify Takens embedding and KDE bandwidth parameters to configure the TE/NMI computation engine. Returns a session_id required for subsequent operations.

        Calls POST /anomaly/session/open
        """
        payload = {}
        payload['variable_names'] = variable_names
        payload['sampling_interval_ms'] = sampling_interval_ms
        payload['takens_embedding_dim'] = takens_embedding_dim
        payload['takens_delay_tau'] = takens_delay_tau
        payload['kde_bandwidth_method'] = kde_bandwidth_method
        payload['te_window_size'] = te_window_size
        payload['nmi_lag_max'] = nmi_lag_max
        url = self.base_url + '/anomaly/session/open'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise AnomalyDetectionApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def ingest_sliding_window_observations(self, session_id: str, observations: list[list[float]], timestamps_iso: list[str], flush_partial_window: bool) -> dict:
        """Push a batch of multivariate observations (sliding window data) into an active anomaly detection session. Provide synchronized timestamps in ISO format. Use this to continuously feed data into the session. Do NOT use to replace historical data; it's for live streaming. The flush_partial_window flag forces computation of a partial window. Returns buffer status and number of ready windows.

        Calls POST /anomaly/session/ingest
        """
        payload = {}
        payload['session_id'] = session_id
        payload['observations'] = observations
        payload['timestamps_iso'] = timestamps_iso
        payload['flush_partial_window'] = flush_partial_window
        url = self.base_url + '/anomaly/session/ingest'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise AnomalyDetectionApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def stream_causal_anomaly_events(self, session_id: str, nmi_anomaly_threshold: float, te_net_flow_min: float, max_events: float) -> dict:
        """Retrieve detected causal anomaly events from a session's computation pipeline. Specify NMI anomaly threshold and minimum net Transfer Entropy flow to filter events. Use this to pull detected anomalies that indicate a causal origin. Do NOT use for raw anomaly scores only; for causal map resolution, use resolve_anomaly_causal_map. Returns array of events with origin node and propagation depth.

        Calls POST /anomaly/events/stream
        """
        payload = {}
        payload['session_id'] = session_id
        payload['nmi_anomaly_threshold'] = nmi_anomaly_threshold
        payload['te_net_flow_min'] = te_net_flow_min
        payload['max_events'] = max_events
        url = self.base_url + '/anomaly/events/stream'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise AnomalyDetectionApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def resolve_anomaly_causal_map(self, session_id: str, event_id: str, include_reverse_te: bool, min_net_te_weight: float) -> dict:
        """Resolve the detailed causal graph for a specific anomaly event previously detected. Set include_reverse_te to true if you need both directions of Transfer Entropy; set min_net_te_weight to filter weak edges. Use this to understand propagation paths from the origin node. Do NOT use for browsing events; use stream_causal_anomaly_events first to get event_ids. Returns adjacency and edge weights.

        Calls POST /anomaly/causal-map/resolve
        """
        payload = {}
        payload['session_id'] = session_id
        payload['event_id'] = event_id
        payload['include_reverse_te'] = include_reverse_te
        payload['min_net_te_weight'] = min_net_te_weight
        url = self.base_url + '/anomaly/causal-map/resolve'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise AnomalyDetectionApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()

    def close_anomaly_session(self, session_id: str, export_session_summary: bool) -> dict:
        """Terminate an active session and optionally export a summary of all observations and events. Use this when monitoring ends. Do NOT call on already closed sessions. After closure, the session_id is invalidated and data is archived.

        Calls POST /anomaly/session/close
        """
        payload = {}
        payload['session_id'] = session_id
        payload['export_session_summary'] = export_session_summary
        url = self.base_url + '/anomaly/session/close'
        response = self.session.post(url, json=payload, timeout=self.timeout)
        if not response.ok:
            raise AnomalyDetectionApiError(f'HTTP {response.status_code}: {response.text[:500]}', status_code=response.status_code)
        return response.json()