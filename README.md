# Anomaly Detection API

## Overview

The **Anomaly Detection API** is a cutting-edge tool that leverages both **Normalized Mutual Information (NMI)** and **Transfer Entropy (TE)** to detect anomalies and infer the direction of causality between multivariate time series. Unlike traditional systems, our API not only identifies anomalies but also provides context by pinpointing their origins within complex systems.

## Problem Addressed

Current anomaly detection solutions either lack a method to determine the source of an anomaly or require frequent manual retraining when the underlying pattern changes. Our approach fills this gap by combining NMI for measuring non-linear dependencies and TE for inferring causality, making it uniquely suited for real-time analysis.

## Key Features

- **AnomalyCausalMap**: For each detected anomaly, our API generates a Directed Graph where edges are weighted based on the difference between forward and reverse Transfer Entropy. This map identifies the node (variable) from which the anomaly originated.
  
- **Time-Efficient Computation**: Utilizes adaptive Kernel Density Estimation (KDE) within time-embedded windows to achieve real-time performance with a complexity of O(n log n).

- **Stateless Design**: Ensures scalable deployment without external dependencies, making it easy to integrate into existing workflows.

## Installation

The API is not currently available as a pip or npm package. To use it, you need to clone the repository and install the required dependencies. You can then interact with the API using HTTP requests.

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/anomaly-detection-api.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Endpoints

### 1. Open a Session

#### Endpoint
```
POST /open_multivariate_anomaly_session
```

#### Request Body
```json
{
  "variable_names": ["VariableA", "VariableB"],
  "sampling_interval_ms": 500,
  "takens_embedding_dim": 5,
  "takens_delay_tau": 1,
  "kde_bandwidth_method": "scott",
  "te_window_size": 60,
  "nmi_lag_max": 1
}
```

#### Response Body
```json
{
  "session_id": "SESSION_12345"
}
```

### 2. Ingest Observations

#### Endpoint
```
POST /ingest_multivariate_anomaly_observations
```

#### Request Body
```json
{
  "session_id": "SESSION_12345",
  "observations": [
    [0.1, 0.2],
    [0.2, 0.3],
    [0.3, 0.4]
  ],
  "timestamps_iso": [
    "2023-10-01T00:00:00Z",
    "2023-10-01T00:05:00Z",
    "2023-10-01T00:10:00Z"
  ],
  "flush_partial_window": true
}
```

### 3. Stream Anomaly Events

#### Endpoint
```
POST /stream_multivariate_causal_anomaly_events
```

#### Request Body
```json
{
  "session_id": "SESSION_12345",
  "nmi_anomaly_threshold": 0.8,
  "te_net_flow_min": 0.1,
  "max_events": 10
}
```

### 4. Resolve Causal Map

#### Endpoint
```
POST /resolve_multivariate_causal_map
```

#### Request Body
```json
{
  "session_id": "SESSION_12345",
  "event_id": "EVENT_67890",
  "include_reverse_te": true,
  "min_net_te_weight": 0.05
}
```

### 5. Close a Session

#### Endpoint
```
POST /close_multivariate_anomaly_session
```

#### Request Body
```json
{
  "session_id": "SESSION_12345",
  "export_session_summary": true
}
```

## Usage Example

Here's a step-by-step example of how to use the API:

1. **Open a Session**:
   ```python
   import requests
   import json

   url = "http://localhost:8000/open_multivariate_anomaly_session"
   data = {
       "variable_names": ["VariableA", "VariableB"],
       "sampling_interval_ms": 500,
       "takens_embedding_dim": 5,
       "takens_delay_tau": 1,
       "kde_bandwidth_method": "scott",
       "te_window_size": 60,
       "nmi_lag_max": 1
   }
   response = requests.post(url, json=data)
   session_id = response.json().get("session_id")
   ```

2. **Ingest Observations**:
   ```python
   url = "http://localhost:8000/ingest_multivariate_anomaly_observations"
   data = {
       "session_id": session_id,
       "observations": [
           [0.1, 0.2],
           [0.2, 0.3],
           [0.3, 0.4]
       ],
       "timestamps_iso": [
           "2023-10-01T00:00:00Z",
           "2023-10-01T00:05:00Z",
           "2023-10-01T00:10:00Z"
       ],
       "flush_partial_window": true
   }
   requests.post(url, json=data)
   ```

3. **Stream Anomaly Events**:
   ```python
   url = "http://localhost:8000/stream_multivariate_causal_anomaly_events"
   data = {
       "session_id": session_id,
       "nmi_anomaly_threshold": 0.8,
       "te_net_flow_min": 0.1,
       "max_events": 10
   }
   response = requests.post(url, json=data)
   events = response.json()
   ```

4. **Resolve Causal Map**:
   ```python
   url = "http://localhost:8000/resolve_multivariate_causal_map"
   data = {
       "session_id": session_id,
       "event_id": "EVENT_67890",
       "include_reverse_te": true,
       "min_net_te_weight": 0.05
   }
   response = requests.post(url, json=data)
   causal_map = response.json()
   ```

5. **Close a Session**:
   ```python
   url = "http://localhost:8000/close_multivariate_anomaly_session"
   data = {
       "session_id": session_id,
       "export_session_summary": true
   }
   requests.post(url, json=data)
   ```

## Conclusion

The Anomaly Detection API provides a robust solution for multivariate anomaly detection and causality inference. By leveraging NMI and TE, it offers unparalleled insights into the dynamics of complex systems, enabling developers to make data-driven decisions with confidence.

For more detailed information on pricing and billing, please refer to `pricing.md`.

---

**Note**: This README is based on the provided specifications and should be used as a reference for integrating the Anomaly Detection API into your applications. For any issues or further questions, please contact the support team.

---

## Pricing

| Calls / month | Price per call |
|---|---|
| 0 - 100 | Free |
| 101 - 10,000 | $0.0025 |
| 10,001 - 100,000 | $0.0018 |
| 100,001 - 1,000,000 | $0.0012 |
| 1,000,001 - 10,000,000 | $0.0008 |
| 10,000,001+ | $0.0005 |

No base fee. No storage fee. No minimum commitment. You pay for computation, not for parking vectors you queried once.