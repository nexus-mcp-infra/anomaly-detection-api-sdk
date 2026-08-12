from typing import List
from fastapi import Depends, Request, WebSocket
import uuid
import time
import math
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

import numpy as np
from scipy.stats import gaussian_kde
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Annotated

app = FastAPI(
    title="CausalAnomalyDetection API",
    version="1.0.0",
    description="Transfer Entropy + NMI causal anomaly detection with Takens embedding. Stateless per-request with optional in-process flywheel cache.",
)

_SESSIONS: dict[str, dict] = {}

ALLOWED_KDE_METHODS = {"scott", "silverman", "plugin"}


def _takens_embed(series: np.ndarray, dim: int, tau: int) -> np.ndarray:
    n = len(series)
    max_start = n - (dim - 1) * tau
    if max_start <= 0:
        raise ValueError(
            f"Series length {n} too short for embedding dim={dim}, tau={tau}. "
            f"Need at least {(dim - 1) * tau + 1} observations."
        )
    embedded = np.empty((max_start, dim), dtype=np.float64)
    for i in range(max_start):
        for d in range(dim):
            embedded[i, d] = series[i + d * tau]
    return embedded


def _kde_entropy_knn(X: np.ndarray, k: int = 5) -> float:
    n, d = X.shape
    if n <= k:
        raise ValueError(f"Sample size {n} must exceed k={k} for KNN entropy estimation.")
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree", metric="chebyshev").fit(X)
    distances, _ = nbrs.kneighbors(X)
    eps = distances[:, k]
    eps = np.where(eps == 0, 1e-10, eps)
    h = -digamma(k) + digamma(n) + d * np.mean(np.log(2.0 * eps))
    return float(h)


def _transfer_entropy_kde(
    source: np.ndarray,
    target: np.ndarray,
    dim: int,
    tau: int,
    k: int = 5,
) -> float:
    src_emb = _takens_embed(source, dim, tau)
    tgt_emb = _takens_embed(target, dim, tau)
    min_len = min(len(src_emb), len(tgt_emb))
    if min_len <= k + 2:
        return 0.0
    src_emb = src_emb[:min_len]
    tgt_emb = tgt_emb[:min_len]
    tgt_future = tgt_emb[1:min_len, 0:1]
    tgt_past = tgt_emb[: min_len - 1]
    src_past = src_emb[: min_len - 1]
    joint_xyz = np.hstack([tgt_future, tgt_past, src_past])
    joint_xy = np.hstack([tgt_future, tgt_past])
    joint_xz = np.hstack([tgt_past, src_past])
    h_xyz = _kde_entropy_knn(joint_xyz, k=k)
    h_xy = _kde_entropy_knn(joint_xy, k=k)
    h_xz = _kde_entropy_knn(joint_xz, k=k)
    h_x = _kde_entropy_knn(tgt_past, k=k)
    te = h_xy + h_xz - h_xyz - h_x
    return float(te)


def _normalized_mutual_information_kde(
    X: np.ndarray,
    Y: np.ndarray,
    bandwidth_method: str = "scott",
    lag: int = 0,
) -> float:
    n = min(len(X), len(Y))
    if n < 10:
        return 0.0
    if lag > 0:
        x = X[: n - lag]
        y = Y[lag:n]
    else:
        x = X[:n]
        y = Y[:n]
    x = x.reshape(-1)
    y = y.reshape(-1)
    joint = np.vstack([x, y])
    if bandwidth_method == "plugin":
        bw = 1.06 * np.std(x) * len(x) ** (-1.0 / 5.0)
        bw = max(bw, 1e-6)
        bw_str = bw
    else:
        bw_str = bandwidth_method
    try:
        kde_x = gaussian_kde(x, bw_method=bw_str if bandwidth_method != "plugin" else bw)
        kde_y = gaussian_kde(y, bw_method=bw_str if bandwidth_method != "plugin" else bw)
        kde_xy = gaussian_kde(joint, bw_method=bw_str if bandwidth_method != "plugin" else bw)
    except np.linalg.LinAlgError:
        return 0.0
    eval_pts = np.vstack([x, y])
    px = kde_x(x)
    py = kde_y(y)
    pxy = kde_xy(eval_pts)
    px = np.clip(px, 1e-10, None)
    py = np.clip(py, 1e-10, None)
    pxy = np.clip(pxy, 1e-10, None)
    mi = np.mean(np.log(pxy / (px * py)))
    mi = max(mi, 0.0)
    hx = -np.mean(np.log(px))
    hy = -np.mean(np.log(py))
    denom = (hx + hy) / 2.0
    if denom <= 0:
        return 0.0
    nmi = mi / denom
    return float(np.clip(nmi, 0.0, 1.0))


def _compute_nmi_baseline(
    data: np.ndarray,
    bandwidth_method: str,
    nmi_lag_max: int,
) -> np.ndarray:
    n_vars = data.shape[1]
    nmi_matrix = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                nmi_matrix[i, j] = 1.0
                continue
            best_nmi = 0.0
            for lag in range(0, min(nmi_lag_max + 1, 10)):
                v = _normalized_mutual_information_kde(
                    data[:, i], data[:, j], bandwidth_method=bandwidth_method, lag=lag
                )
                if v > best_nmi:
                    best_nmi = v
            nmi_matrix[i, j] = best_nmi
    return nmi_matrix


def _compute_te_matrix(
    data: np.ndarray,
    dim: int,
    tau: int,
    k: int = 5,
) -> np.ndarray:
    n_vars = data.shape[1]
    te_matrix = np.zeros((n_vars, n_vars), dtype=np.float64)
    for i in range(n_vars):
        for j in range(n_vars):
            if i == j:
                continue
            te_matrix[i, j] = _transfer_entropy_kde(
                source=data[:, i],
                target=data[:, j],
                dim=dim,
                tau=tau,
                k=k,
            )
    return te_matrix


def _detect_nmi_anomaly(
    baseline_nmi: np.ndarray,
    window_nmi: np.ndarray,
    threshold: float,
) -> tuple[bool, float]:
    n = baseline_nmi.shape[0]
    diffs = []
    for i in range(n):
        for j in range(n):
            if i != j:
                diffs.append(abs(window_nmi[i, j] - baseline_nmi[i, j]))
    if not diffs:
        return False, 0.0
    score = float(np.mean(diffs))
    return score >= threshold, score


def _build_causal_graph(
    te_matrix: np.ndarray,
    te_net_flow_min: float,
) -> tuple[list[tuple[int, int, float]], int]:
    n = te_matrix.shape[0]
    edges = []
    node_net_out = defaultdict(float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            forward = te_matrix[i, j]
            reverse = te_matrix[j, i]
            net = forward - reverse
            if net >= te_net_flow_min:
                edges.append((i, j, net))
                node_net_out[i] += net
    if not node_net_out:
        origin = 0
    else:
        origin = max(node_net_out, key=lambda k: node_net_out[k])
    return edges, int(origin)


def _propagation_depth(edges: list[tuple[int, int, float]], origin: int, n_vars: int) -> int:
    adj = defaultdict(list)
    for src, dst, _ in edges:
        adj[src].append(dst)
    visited = set()
    queue = [(origin, 0)]
    max_depth = 0
    while queue:
        node, depth = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        max_depth = max(max_depth, depth)
        for nxt in adj[node]:
            if nxt not in visited:
                queue.append((nxt, depth + 1))
    return max_depth


class OpenSessionRequest(BaseModel):
    variable_names: Annotated[
        list[str],
        Field(..., description="List of variable names to monitor. Must contain at least 2 elements."),
    ]
    sampling_interval_ms: Annotated[
        float,
        Field(..., description="Sampling interval in milliseconds.", ge=1),
    ]
    # NEXUS_PARAM_DEVIATION: takens_embedding_dim -- stored as int; float ge=2 per spec but embedding dim is always integer in Takens theory
    takens_embedding_dim: Annotated[
        float,
        Field(..., description="Embedding dimension for Takens reconstruction.", ge=2),
    ]
    # NEXUS_PARAM_DEVIATION: takens_delay_tau -- stored as int; float ge=1 per spec but delay tau is always integer in discrete-time embedding
    takens_delay_tau: Annotated[
        float,
        Field(..., description="Delay tau for Takens embedding.", ge=1),
    ]
    kde_bandwidth_method: Annotated[
        str,
        Field(..., description="KDE bandwidth estimation method: 'scott', 'silverman', or 'plugin'.", min_length=1),
    ]
    # NEXUS_PARAM_DEVIATION: te_window_size -- stored as int; float ge=10 per spec but window size indexes discrete observations
    te_window_size: Annotated[
        float,
        Field(..., description="Window size (observations) for Transfer Entropy computation.", ge=10),
    ]
    # NEXUS_PARAM_DEVIATION: nmi_lag_max -- stored as int; float ge=1 per spec but lag is always a discrete integer index
    nmi_lag_max: Annotated[
        float,
        Field(..., description="Maximum lag for NMI computation.", ge=1),
    ]

    @field_validator("variable_names")
    @classmethod
    def at_least_two_variables(cls, v: list[str]) -> list[str]:
        if len(v) < 2:
            raise ValueError("variable_names must contain at least 2 elements for multivariate causal analysis.")
        return v

    @field_validator("kde_bandwidth_method")
    @classmethod
    def valid_kde_method(cls, v: str) -> str:
        if v not in ALLOWED_KDE_METHODS:
            raise ValueError(f"kde_bandwidth_method must be one of {sorted(ALLOWED_KDE_METHODS)}, got '{v}'.")
        return v


class IngestObservationsRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(..., description="Session ID returned by open_multivariate_anomaly_session.", min_length=1),
    ]
    observations: Annotated[
        list[list[float]],
        Field(..., description="2D array shape [timesteps, variables] matching variable order."),
    ]
    timestamps_iso: Annotated[
        list[str],
        Field(..., description="ISO 8601 timestamps for each observation row."),
    ]
    flush_partial_window: Annotated[
        bool,
        Field(..., description="If true, force computation on any partial window in the buffer."),
    ]

    @model_validator(mode="after")
    def observations_timestamps_align(self) -> "IngestObservationsRequest":
        if len(self.observations) != len(self.timestamps_iso):
            raise ValueError(
                f"observations has {len(self.observations)} rows but timestamps_iso has "
                f"{len(self.timestamps_iso)} entries; they must match."
            )
        return self


class StreamEventsRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(..., description="The session ID.", min_length=1),
    ]
    nmi_anomaly_threshold: Annotated[
        float,
        Field(..., description="NMI anomaly score threshold in [0,1].", ge=0, le=1),
    ]
    te_net_flow_min: Annotated[
        float,
        Field(..., description="Minimum net TE flow (TE_forward - TE_reverse) to qualify an edge as causal.", ge=0),
    ]
    # NEXUS_PARAM_DEVIATION: max_events -- stored as int; float ge=1 per spec but event count is always integer
    max_events: Annotated[
        float,
        Field(..., description="Maximum number of events to return.", ge=1),
    ]


class ResolveCausalMapRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(..., description="The session ID.", min_length=1),
    ]
    event_id: Annotated[
        str,
        Field(..., description="Event ID from stream_causal_anomaly_events.", min_length=1),
    ]
    include_reverse_te: Annotated[
        bool,
        Field(..., description="If true, include reverse TE values in the output."),
    ]
    min_net_te_weight: Annotated[
        float,
        Field(..., description="Minimum net TE weight for an edge to be included.", ge=0),
    ]


class CloseSessionRequest(BaseModel):
    session_id: Annotated[
        str,
        Field(..., description="The session ID to close.", min_length=1),
    ]
    export_session_summary: Annotated[
        bool,
        Field(..., description="If true, include a session summary in the response."),
    ]


def _require_session(session_id: str) -> dict:
    if not session_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="session_id must be non-empty.")
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found. It may have been closed or never created.",
        )
    if session.get("closed", False):
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail=f"Session '{session_id}' is already closed. Create a new session.",
        )
    return session


@app.post("/anomaly/session/open", status_code=status.HTTP_201_CREATED)
def open_multivariate_anomaly_session(req: OpenSessionRequest) -> dict:
    session_id = str(uuid.uuid4())
    _SESSIONS[session_id] = {
        "session_id": session_id,
        "variable_names": req.variable_names,
        "n_vars": len(req.variable_names),
        "sampling_interval_ms": req.sampling_interval_ms,
        "takens_embedding_dim": int(req.takens_embedding_dim),
        "takens_delay_tau": int(req.takens_delay_tau),
        "kde_bandwidth_method": req.kde_bandwidth_method,
        "te_window_size": int(req.te_window_size),
        "nmi_lag_max": int(req.nmi_lag_max),
        "buffer": [],
        "baseline_nmi": None,
        "events": {},
        "windows_processed": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "closed": False,
        "total_observations": 0,
    }
    return {
        "session_id": session_id,
        "variable_names": req.variable_names,
        "n_vars": len(req.variable_names),
        "te_window_size": int(req.te_window_size),
        "takens_embedding_dim": int(req.takens_embedding_dim),
        "takens_delay_tau": int(req.takens_delay_tau),
        "kde_bandwidth_method": req.kde_bandwidth_method,
        "nmi_lag_max": int(req.nmi_lag_max),
        "created_at": _SESSIONS[session_id]["created_at"],
        "status": "open",
    }


@app.post("/anomaly/session/ingest", status_code=status.HTTP_200_OK)
def ingest_sliding_window_observations(req: IngestObservationsRequest) -> dict:
    session = _require_session(req.session_id)
    n_vars = session["n_vars"]
    if not req.observations:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="observations must be a non-empty list of observation rows.",
        )
    for idx, row in enumerate(req.observations):
        if not isinstance(row, list) or len(row) != n_vars:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"Row {idx} has {len(row) if isinstance(row, list) else 'non-list'} values; "
                    f"expected {n_vars} matching variable_names."
                ),
            )
    for ts in req.timestamps_iso:
        try:
            datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid ISO 8601 timestamp: '{ts}'.",
            )
    buffer: list = session["buffer"]
    for row, ts in zip(req.observations, req.timestamps_iso):
        buffer.append({"values": row, "ts": ts})
    session["total_observations"] += len(req.observations)
    window_size = session["te_window_size"]
    dim = session["takens_embedding_dim"]
    tau = session["takens_delay_tau"]
    bw = session["kde_bandwidth_method"]
    nmi_lag_max = session["nmi_lag_max"]
    windows_computed = 0
    while len(buffer) >= window_size or (req.flush_partial_window and len(buffer) >= max(dim * tau + 2, 10)):
        chunk_size = window_size if len(buffer) >= window_size else len(buffer)
        chunk = buffer[:chunk_size]
        data = np.array([r["values"] for r in chunk], dtype=np.float64)
        nmi_mat = _compute_nmi_baseline(data, bw, nmi_lag_max)
        te_mat = _compute_te_matrix(data, dim, tau)
        if session["baseline_nmi"] is None:
            session["baseline_nmi"] = nmi_mat.copy()
            session["baseline_te"] = te_mat.copy()
        else:
            is_anomaly, score = _detect_nmi_anomaly(session["baseline_nmi"], nmi_mat, threshold=0.05)
            if is_anomaly:
                edges, origin = _build_causal_graph(te_mat, te_net_flow_min=0.0)
                depth = _propagation_depth(edges, origin, n_vars)
                event_id = str(uuid.uuid4())
                session["events"][event_id] = {
                    "event_id": event_id,
                    "window_index": session["windows_processed"],
                    "nmi_anomaly_score": score,
                    "origin_node_index": origin,
                    "origin_variable": session["variable_names"][origin],
                    "propagation_depth": depth,
                    "te_matrix": te_mat.tolist(),
                    "nmi_matrix": nmi_mat.tolist(),
                    "window_start_ts": chunk[0]["ts"],
                    "window_end_ts": chunk[-1]["ts"],
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                }
            alpha = 0.15
            session["baseline_nmi"] = (1 - alpha) * session["baseline_nmi"] + alpha * nmi_mat
        if len(buffer) >= window_size:
            buffer[:] = buffer[window_size:]
        else:
            buffer.clear()
        session["windows_processed"] += 1
        windows_computed += 1
        if req.flush_partial_window and len(buffer) < max(dim * tau + 2, 10):
            break
    return {
        "session_id": req.session_id,
        "buffer_observations": len(buffer),
        "windows_computed_this_call": windows_computed,
        "total_windows_processed": session["windows_processed"],
        "total_observations_ingested": session["total_observations"],
        "total_events_detected": len(session["events"]),
        "status": "ok",
    }


@app.post("/anomaly/events/stream", status_code=status.HTTP_200_OK)
def stream_causal_anomaly_events(req: StreamEventsRequest) -> dict:
    session = _require_session(req.session_id)
    max_events = int(req.max_events)
    te_net_flow_min = req.te_net_flow_min
    nmi_threshold = req.nmi_anomaly_threshold
    n_vars = session["n_vars"]
    matched_events = []
    for event_id, ev in session["events"].items():
        if ev["nmi_anomaly_score"] < nmi_threshold:
            continue
        te_mat = np.array(ev["te_matrix"])
        edges, origin = _build_causal_graph(te_mat, te_net_flow_min=te_net_flow_min)
        if not edges and n_vars > 1:
            continue
        depth = _propagation_depth(edges, origin, n_vars)
        matched_events.append(
            {
                "event_id": event_id,
                "window_index": ev["window_index"],
                "nmi_anomaly_score": round(ev["nmi_anomaly_score"], 6),
                "origin_variable": session["variable_names"][origin],
                "origin_node_index": origin,
                "propagation_depth": depth,
                "causal_edge_count": len(edges),
                "window_start_ts": ev["window_start_ts"],
                "window_end_ts": ev["window_end_ts"],
                "detected_at": ev["detected_at"],
            }
        )
        if len(matched_events) >= max_events:
            break
    return {
        "session_id": req.session_id,
        "events": matched_events,
        "events_returned": len(matched_events),
        "total_events_in_session": len(session["events"]),
        "nmi_anomaly_threshold_applied": nmi_threshold,
        "te_net_flow_min_applied": te_net_flow_min,
    }


@app.post("/anomaly/causal-map/resolve", status_code=status.HTTP_200_OK)
def resolve_anomaly_causal_map(req: ResolveCausalMapRequest) -> dict:
    session = _require_session(req.session_id)
    event = session["events"].get(req.event_id)
    if event is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Event '{req.event_id}' not found in session '{req.session_id}'. "
                "Use stream_causal_anomaly_events to retrieve valid event IDs."
            ),
        )
    te_mat = np.array(event["te_matrix"])
    n_vars = session["n_vars"]
    variable_names = session["variable_names"]
    edges_raw, origin = _build_causal_graph(te_mat, te_net_flow_min=req.min_net_te_weight)
    adjacency: dict[str, dict] = {}
    edge_list = []
    for src, dst, net_w in edges_raw:
        src_name = variable_names[src]
        dst_name = variable_names[dst]
        if src_name not in adjacency:
            adjacency[src_name] = {}
        edge_entry: dict = {
            "source": src_name,
            "target": dst_name,
            "net_te_weight": round(net_w, 8),
            "te_forward": round(float(te_mat[src, dst]), 8),
        }
        if req.include_reverse_te:
            edge_entry["te_reverse"] = round(float(te_mat[dst, src]), 8)
        adjacency[src_name][dst_name] = round(net_w, 8)
        edge_list.append(edge_entry)
    propagation_path = []
    adj_map = defaultdict(list)
    for src, dst, w in edges_raw:
        adj_map[src].append((dst, w))
    visited = set()
    queue = [(origin, [variable_names[origin]])]
    while queue:
        node, path = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if len(path) > 1 or node == origin:
            propagation_path.append(path[:])
        for nxt, _ in sorted(adj_map[node], key=lambda x: -x[1]):
            if nxt not in visited:
                queue.append((nxt, path + [variable_names[nxt]]))
    nmi_mat = np.array(event["nmi_matrix"])
    nmi_pairs = {}
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j:
                key = f"{variable_names[i]}->{variable_names[j]}"
                nmi_pairs[key] = round(float(nmi_mat[i, j]), 8)
    return {
        "session_id": req.session_id,
        "event_id": req.event_id,
        "origin_variable": variable_names[origin],
        "origin_node_index": origin,
        "adjacency": adjacency,
        "edges": edge_list,
        "nmi_dependency_map": nmi_pairs,
        "propagation_paths": propagation_path,
        "include_reverse_te": req.include_reverse_te,
        "min_net_te_weight_applied": req.min_net_te_weight,
        "window_start_ts": event["window_start_ts"],
        "window_end_ts": event["window_end_ts"],
        "detected_at": event["detected_at"],
    }


@app.post("/anomaly/session/close", status_code=status.HTTP_200_OK)
def close_anomaly_session(req: CloseSessionRequest) -> dict:
    session = _require_session(req.session_id)
    session["closed"] = True
    session["closed_at"] = datetime.now(timezone.utc).isoformat()
    response: dict = {
        "session_id": req.session_id,
        "status": "closed",
        "closed_at": session["closed_at"],
        "total_observations_ingested": session["total_observations"],
        "total_windows_processed": session["windows_processed"],
        "total_events_detected": len(session["events"]),
    }
    if req.export_session_summary:
        events_summary = []
        for event_id, ev in session["events"].items():
            te_mat = np.array(ev["te_matrix"])
            edges, origin = _build_causal_graph(te_mat, te_net_flow_min=0.0)
            events_summary.append(
                {
                    "event_id": event_id,
                    "window_index": ev["window_index"],
                    "nmi_anomaly_score": round(ev["nmi_anomaly_score"], 6),
                    "origin_variable": session["variable_names"][origin],
                    "propagation_depth": _propagation_depth(edges, origin, session["n_vars"]),
                    "window_start_ts": ev["window_start_ts"],
                    "window_end_ts": ev["window_end_ts"],
                    "detected_at": ev["detected_at"],
                }
            )
        response["session_summary"] = {
            "session_id": req.session_id,
            "variable_names": session["variable_names"],
            "created_at": session["created_at"],
            "closed_at": session["closed_at"],
            "sampling_interval_ms": session["sampling_interval_ms"],
            "te_window_size": session["te_window_size"],
            "takens_embedding_dim": session["takens_embedding_dim"],
            "takens_delay_tau": session["takens_delay_tau"],
            "kde_bandwidth_method": session["kde_bandwidth_method"],
            "nmi_lag_max": session["nmi_lag_max"],
            "total_observations": session["total_observations"],
            "total_windows_processed": session["windows_processed"],
            "total_events_detected": len(session["events"]),
            "events": events_summary,
            "baseline_nmi_final": session["baseline_nmi"].tolist() if session["baseline_nmi"] is not None else None,
        }
    return response

# --- NEXUS: servidor MCP real montado en el mismo proceso (inyectado por forge_agent) ---
# Reemplaza el wrapper Node/TypeScript separado -- un solo deploy, sin
# segundo servicio, sin salto de red interno. Ver mcp_wrapper_generator.py
# (v2.0) para el razonamiento completo, incluido el gotcha de
# session_manager que explica el patron startup/shutdown de abajo.

from typing import Annotated, Any, Literal
from contextlib import asynccontextmanager

import asyncio
import base64
import json
import os
import time
import httpx
from pydantic import Field
from fastapi import FastAPI
from mcp.server.fastmcp import Context, FastMCP as _NexusFastMCP
from mcp.server.transport_security import TransportSecuritySettings

# --- NEXUS: PATCH fix_mcp_dns_rebinding_host ---
# FastMCP() sin host/transport_security explicito activa proteccion
# anti DNS-rebinding con allowlist localhost-only por default del SDK,
# rechazando con 421 "Invalid Host header" cualquier request real
# contra el dominio publico de Railway (bug real confirmado en
# produccion 2026-07-09, ver docstring del generador). Se pasa
# transport_security explicito leyendo RAILWAY_PUBLIC_DOMAIN en
# runtime -- Railway lo inyecta automaticamente en cada servicio, asi
# que este codigo no necesita conocer su propio dominio al generarse.
_nexus_railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "*")

_nexus_mcp = _NexusFastMCP(
    'nexus-anomaly-detection-api',
    stateless_http=True,
    host="0.0.0.0",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        # --- PATCH fix_mcp_dns_rebinding_bare_host ---
        # Railway (como cualquier proxy HTTPS estandar) manda el Host
        # header SIN puerto explicito -- "dominio:*" nunca matchea eso,
        # solo matchea "dominio:443". Se agrega tambien el dominio
        # pelado para cubrir ambos casos (bug real confirmado en
        # produccion 2026-07-09: primer fix desplegado, /mcp seguia
        # devolviendo 421 tras el redeploy).
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            _nexus_railway_domain,
            _nexus_railway_domain + ":*",
        ],
        allowed_origins=[
            "http://localhost:*",
            "http://127.0.0.1:*",
            "https://" + _nexus_railway_domain,
        ],
    ),
)


# --- NEXUS: instancia FastAPI aislada para llamadas MCP->core internas ---
# Comparte los MISMOS objetos de ruta (app.routes) que `app` -- misma
# resolucion real de FastAPI DI (Security()/Depends(), lo que el LLM haya
# escrito) -- pero SIN ningun @app.middleware/add_middleware propio de
# `app` (billing Stripe, rate-limit, x402 PaymentMiddlewareASGI,
# traffic-log). Esos middleware ya corrieron UNA vez sobre la request HTTP
# real a /mcp (Starlette envuelve el Mount de FastMCP en "/" con el mismo
# stack que el resto de `app`) -- esta llamada interna NO debe volver a
# dispararlos, y (para rutas x402-gateadas) no debe recibir el mismo 402
# que la ruta REST publica exige, porque el pago real (si el asset lo
# tiene) se verifica aparte, a nivel de tool MCP -- ver CLAUDE.md SS9.5x.
# `list(...)` fuerza una copia -- Router.__init__ ya copia internamente,
# pero se es explicito aca para que esta instancia quede fija al set de
# rutas REST que existe en este punto (antes de app.mount("/", ...) mas
# abajo), sin importar mutaciones futuras de app.routes.
_nexus_internal_app = FastAPI(routes=list(app.routes))


async def _nexus_mcp_call_core(method: str, path: str, params: dict, headers: dict | None = None) -> Any:
    """
    Llama al endpoint real del core -- via ASGI in-process (sin red
    real, sin segundo proceso) contra _nexus_internal_app (ver arriba),
    NO contra `app` directamente.
    """
    transport = httpx.ASGITransport(app=_nexus_internal_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://nexus-internal") as client:
        if method == "GET":
            resp = await client.get(path, params=params, headers=headers)
        else:
            resp = await client.post(path, json=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


# --- PATCH mcp_call_events_telemetry ---
# mcp_call_events / revenue_events -- ver Fase 1 (Revenue/Usage
# Instrumentation). Generator-side por diseno: cualquier asset nuevo
# que FORGE construya de aca en adelante nace con esto, sin depender de
# un patch manual posterior por asset.
#
# Credenciales leidas de env vars Railway con el MISMO patron defensivo
# que ya usa _nexus_usage_middleware (forge_output_saver_v6.py) para
# Stripe: si SUPABASE_URL/SUPABASE_ANON_KEY no estan seteadas (asset
# corriendo local, o el paso de sync de env vars del pipeline de deploy
# todavia no las inyecto -- ESE paso vive fuera de este generador, ver
# billing_reconciliation.py:134-144 para el patron real que lo haria),
# el insert es un no-op silencioso -- nunca rompe la response real de
# una tool call.
#
# IMPORTANTE: usa la key anon/publishable, NUNCA service_role -- esta
# key vive en el runtime del asset deployado (potencialmente expuesta
# via env dump/logs), la RLS de mcp_call_events/revenue_events
# (policies INSERT-only para el rol anon, sin SELECT/UPDATE/DELETE) es
# la unica proteccion real contra un uso indebido de esta key si se
# filtra.
_NEXUS_SUPABASE_URL = os.getenv("SUPABASE_URL")
_NEXUS_SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
_NEXUS_SECTOR = 'anomaly_detection'
_NEXUS_ASSET_NAME = 'Anomaly Detection API'


def _nexus_truncate_ip(raw_ip):
    """Mismo algoritmo/mismo resultado que _nexus_traffic_log_truncate_ip
    (archive/patches/patch_traffic_log_similarity_search.py) -- portado
    aca para que todo asset nuevo lo tenga desde generacion. Trunca a
    /24 (IPv4) o /64 (IPv6); nunca devuelve la IP completa."""
    if not raw_ip:
        return None
    if ":" in raw_ip and "." not in raw_ip:
        segments = [s for s in raw_ip.split(":") if s]
        head = segments[:4] if len(segments) >= 4 else segments
        return (":".join(head) + "::/64") if head else None
    octets = raw_ip.split(".")
    if len(octets) == 4 and all(o.isdigit() for o in octets):
        return f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
    return None


def _nexus_extract_wallet(payment_header):
    """Mismo algoritmo que _nexus_rate_limit_extract_wallet
    (archive/patches/patch_rate_limit_similarity_search.py) -- decodifica
    el header X-PAYMENT (base64 -> JSON) y extrae la wallet pagadora de
    payload.authorization.from."""
    try:
        padded = payment_header + "=" * (-len(payment_header) % 4)
        payload = json.loads(base64.b64decode(padded))
        payer = payload.get("payload", {}).get("authorization", {}).get("from")
        return payer.lower() if isinstance(payer, str) and payer else None
    except Exception:
        return None


def _nexus_call_context(ctx):
    """
    Best-effort: ctx.request_context.request es un starlette.Request
    REAL incluso con stateless_http=True -- se puebla por REQUEST HTTP
    individual (mcp/server/streamable_http.py: ServerMessageMetadata(
    request_context=request), poblado en _create_session_message()),
    no por continuidad de sesion. Verificado contra el codigo fuente
    real de mcp==1.28.1 (version pinneada del proyecto) antes de asumir
    que el dato esta disponible -- no es una suposicion sin chequear.

    agent_framework: no existe un campo dedicado para esto en el
    protocolo MCP tal como esta implementado hoy con
    stateless_http=True -- ctx.session.client_params.clientInfo (que
    SI llevaria un nombre de framework/cliente real) solo se puebla si
    el mensaje initialize y la tool call posterior comparten la misma
    ServerSession, y en modo stateless cada POST crea una ServerSession
    nueva (mcp/server/streamable_http_manager.py:_handle_stateless_request).
    La señal real y disponible en su lugar es el header User-Agent de
    la request HTTP -- se usa tal cual cuando el cliente lo manda
    (nunca fabricado; queda None si el cliente no lo envia).
    """
    ip_range = None
    agent_framework = None
    wallet = None
    try:
        request = ctx.request_context.request if ctx is not None else None
        if request is not None:
            forwarded = request.headers.get("x-forwarded-for")
            raw_ip = forwarded.split(",")[0].strip() if forwarded else (
                request.client.host if request.client else None
            )
            ip_range = _nexus_truncate_ip(raw_ip)
            ua = request.headers.get("user-agent")
            agent_framework = ua[:255] if ua else None
            payment_header = request.headers.get("x-payment")
            if payment_header:
                wallet = _nexus_extract_wallet(payment_header)
    except Exception:
        pass
    return ip_range, agent_framework, wallet


async def _nexus_supabase_insert(table, payload):
    if not _NEXUS_SUPABASE_URL or not _NEXUS_SUPABASE_ANON_KEY:
        return
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await client.post(
                f"{_NEXUS_SUPABASE_URL}/rest/v1/{table}",
                json=payload,
                headers={
                    "apikey": _NEXUS_SUPABASE_ANON_KEY,
                    "Authorization": f"Bearer {_NEXUS_SUPABASE_ANON_KEY}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
    except Exception:
        pass  # nunca romper el flujo real por un fallo de telemetria


async def _nexus_log_mcp_call_event(tool_id, success, latency_ms, ctx, route_key=None):
    """
    Escribe mcp_call_events (proxy de uso/latencia) siempre que haya
    credenciales -- dispara en el `finally` de cada tool MCP, es decir
    cuando el handler retorna (con o sin excepcion), ANTES de que exista
    ningun intento de settlement x402 real para esa llamada.

    --- PATCH x402_revenue_events_hook ---
    Ya NO escribe revenue_events desde aca (si escribia hasta esta
    sesion -- ver CLAUDE.md SS9.58). Ese insert vivia en este mismo punto
    proxy: para un tool protegido por x402 (create_payment_wrapper), el
    handler corre y sirve el recurso ANTES de que se intente el
    settlement (x402/mcp/server.py: handler primero, settle_payment()
    despues) -- loguear "cobrado" aca era literalmente loguearlo antes
    de que el pago se intentara liquidar. revenue_events ahora se
    escribe desde _nexus_log_x402_revenue_event, enganchado directo a
    x402ResourceServer.on_after_settle() (ver mas abajo) -- el unico
    punto que conoce el SettleResponse real (success/transaction/payer)
    y dispara solo cuando success es True.

    x402_routes/price_charged se mantienen aca SOLO para poblar
    mcp_call_events.price_charged, que si tiene sentido como proxy:
    "cuanto hubiera cobrado esta llamada si el pago hubiera settleado",
    no una afirmacion de que settleo. is_paid_route sigue en False para
    cualquier asset nuevo (x402 no se genera hoy por FORGE, CLAUDE.md
    SS8) hasta que un patch manual agregue _NEXUS_X402_ROUTES.
    """
    ip_range, agent_framework, wallet = _nexus_call_context(ctx)
    price_charged = None
    x402_routes = globals().get("_NEXUS_X402_ROUTES") or {}
    is_paid_route = route_key is not None and route_key in x402_routes
    if is_paid_route:
        raw_price = globals().get("_NEXUS_X402_PRICE")
        if raw_price:
            try:
                price_charged = float(str(raw_price).lstrip("$"))
            except Exception:
                price_charged = None
    # --- PATCH mcp_call_events_asset_name ---
    await _nexus_supabase_insert("mcp_call_events", {
        "agent_framework": agent_framework,
        "tool_id": tool_id,
        "sector": _NEXUS_SECTOR,
        "asset_name": _NEXUS_ASSET_NAME,
        # token_input/token_output: null a proposito -- ningun asset
        # que FORGE genera hoy envuelve una llamada LLM propia (son
        # productos como vector search / websocket / rate limiting,
        # no proxies de un modelo). Si algun dia existe un asset que SI
        # envuelva un LLM, ese caso deberia poblar estos campos en su
        # propio call site en vez de forzar un valor generico aca.
        "token_input": None,
        "token_output": None,
        "success": success,
        "latency_ms": latency_ms,
        "client_ip_range": ip_range,
        "price_charged": price_charged,
    })


async def _nexus_log_x402_revenue_event(ctx) -> None:
    """
    --- PATCH x402_revenue_events_hook ---
    AfterSettleHook real para revenue_events -- registrado via
    _nexus_register_x402_revenue_logging(), nunca llamado directo.
    Firma compatible con x402.server_base.AfterSettleHook
    (Callable[[SettleResultContext], Awaitable[None] | None]) pero SIN
    importar el tipo -- duck-typed a proposito, para que este generador
    no obligue a que x402 este instalado en assets que no lo tienen (la
    funcion se emite igual en todo asset nuevo, pero solo se registra
    -- ver _nexus_register_x402_revenue_logging -- si el asset tiene su
    propio _nexus_x402_server real).

    x402ResourceServer.on_after_settle() (x402/server.py) SOLO dispara
    cuando settle_result.success es True -- confirmado leyendo
    _settle_payment_core (x402/server_base.py): un settle fallido nunca
    llega a la lista de after-settle hooks, asi que esta funcion nunca
    necesita chequear success ademas del propio getattr defensivo de
    abajo. Nunca levanta: un bug aca no debe poder romper la respuesta
    de pago real que el caller ya esta por recibir.
    """
    try:
        result = getattr(ctx, "result", None)
        requirements = getattr(ctx, "requirements", None)
        if result is None or requirements is None or not getattr(result, "success", False):
            return
        # amount viene en unidades atomicas del asset -- USDC (6
        # decimales) en los 2 assets reales que usan x402 hoy
        # (similarity-search-api, ws; ver patch_x402_similarity_search.py
        # / patch_x402_ws.py). Mismo mismatch de schema ya documentado
        # mas arriba para mcp_call_events.price_charged: amount_eur no
        # es EUR, es el valor numerico crudo, sin conversion FX.
        raw_amount = getattr(requirements, "amount", None)
        amount_eur = int(raw_amount) / 1_000_000 if raw_amount is not None else None
        await _nexus_supabase_insert("revenue_events", {
            "asset_name": _NEXUS_ASSET_NAME,
            "amount_eur": amount_eur,
            "pricing_model": "x402",
            "stripe_event_id": None,
            "customer_id": getattr(result, "payer", None),
        })
    except Exception:
        pass


def _nexus_register_x402_revenue_logging(x402_server) -> None:
    """
    --- PATCH x402_revenue_events_hook ---
    Registra _nexus_log_x402_revenue_event contra el on_after_settle()
    REAL de x402ResourceServer -- llamar UNA vez, justo despues de
    instanciar/inicializar _nexus_x402_server (mismo lugar donde un
    patch x402 manual ya llama _nexus_x402_server.initialize()).

    Por que aca y no en _nexus_log_mcp_call_event / no en
    PaymentWrapperHooks.on_after_settlement (x402/mcp/types.py): tanto
    las rutas REST (PaymentMiddlewareASGI -> x402_http_server.py:216)
    como los tools MCP (create_payment_wrapper -> server.py /
    server_async.py) llaman settle_payment() sobre la MISMA instancia
    compartida de x402ResourceServer -- un solo hook registrado aca
    cubre ambas superficies de pago de un asset sin que este generador
    necesite saber cual usa cada uno. PaymentWrapperHooks.on_after_settlement
    es exclusivo del wrapper MCP: hubiera dejado sin cubrir cualquier
    pago hecho directo por REST.

    Riesgo conocido, sin cerrar (ver CLAUDE.md SS9.58): no hay forma,
    desde este codigo, de confirmar que settle_result.success==True
    significa "confirmado on-chain" en vez de "el facilitador externo
    (CDP/x402.org) lo acepto como valido, cadena aun no verificada" --
    settle_payment() hace una unica llamada HTTP a {facilitator}/settle
    y confia en el campo `success` que ese facilitador devuelva, sin
    poll ni retry propios. El facilitador de REFERENCIA que trae este
    mismo SDK (x402/mechanisms/evm/exact/facilitator.py) SI espera
    confirmacion on-chain real (wait_for_transaction_receipt) antes de
    devolver success=True -- evidencia de la semantica esperada del
    protocolo, no prueba de que un facilitador externo la respete.

    Idempotencia: revenue_events no tiene columna de tx hash ni
    constraint unico, y la RLS del rol anon es INSERT-only (sin
    SELECT) -- no hay forma de check-before-insert desde el propio
    asset. Se confia en que EIP-3009 (nonce de un solo uso) hace que un
    replay real del mismo payload firmado falle el settlement
    (success=False) en el segundo intento -- este hook solo dispara en
    success=True. Inferencia del diseno del esquema de firma, no
    verificada empiricamente. Cerrar esto de forma robusta requeriria
    una columna tx_hash + unique index en revenue_events -- cambio de
    schema a infra compartida, fuera de alcance de este generador.
    """
    x402_server.on_after_settle(_nexus_log_x402_revenue_event)


async def _nexus_log_traffic_event(ip_range, method, path, status) -> None:
    """
    --- PATCH traffic_events_hook ---
    Primitiva reusable para persistir [NEXUS_TRAFFIC] a Supabase --
    este generador NO se autoregistra en ningun middleware (el
    middleware de traffic-log sigue siendo un artefacto manual, opt-in,
    por asset -- mismo estado que x402, ver CLAUDE.md SS9.59). Pensada
    para que un patch de traffic-log (mismo patron que
    archive/patches/patch_traffic_log_similarity_search.py) la llame
    JUNTO al print() que ya existe, sin reimplementar el insert en cada
    asset.

    Aditivo por contrato: el caller es responsable de mantener el
    print("[NEXUS_TRAFFIC] ...") intacto y llamar a esta funcion
    ADEMAS, no en su lugar -- AegisAgent.PortfolioAuditor
    (aegis_discovery.py) sigue leyendo esa linea en vivo desde logs de
    Railway, no de Supabase. Nunca levanta: un fallo de telemetria
    nunca debe poder romper la response real.
    """
    try:
        await _nexus_supabase_insert("traffic_events", {
            "asset_name": _NEXUS_ASSET_NAME,
            "ip_range": ip_range,
            "method": method,
            "path": path,
            "status": status,
        })
    except Exception:
        pass



@_nexus_mcp.tool(name='nexus_anomaly_detection_api_open_multivariate_anomaly_session', description='Initialize a new anomaly detection session for a set of time-series variables. Use this when starting real-time causal anomaly monitoring across multiple variables (min 2). Do NOT use for single-variable anomaly detection; use univariate methods instead. Specify Takens embedding and KDE bandwidth parameters to configure the TE/NMI computation engine. Returns a session_id required for subsequent operations.')
async def open_multivariate_anomaly_session(variable_names: Annotated[list[str], Field(..., description='List of variable names to monitor. Must contain at least 2 elements.')], sampling_interval_ms: Annotated[float, Field(..., description='Sampling interval in milliseconds between observations.', ge=1)], takens_embedding_dim: Annotated[float, Field(..., description="Embedding dimension for Takens' reconstruction.", ge=2)], takens_delay_tau: Annotated[float, Field(..., description="Delay tau for Takens' embedding.", ge=1)], kde_bandwidth_method: Annotated[str, Field(..., description="KDE bandwidth estimation method. Allowed: 'scott', 'silverman', 'plugin'.", min_length=1)], te_window_size: Annotated[float, Field(..., description='Window size (number of observations) for Transfer Entropy computation.', ge=10)], nmi_lag_max: Annotated[float, Field(..., description='Maximum lag for Normalized Mutual Information.', ge=1)], ctx: Context) -> dict[str, Any]:
    """Open Anomaly Session"""
    _nexus_path = '/anomaly/session/open'.format()
    params = {"variable_names": variable_names, "sampling_interval_ms": sampling_interval_ms, "takens_embedding_dim": takens_embedding_dim, "takens_delay_tau": takens_delay_tau, "kde_bandwidth_method": kde_bandwidth_method, "te_window_size": te_window_size, "nmi_lag_max": nmi_lag_max}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_anomaly_detection_api_open_multivariate_anomaly_session', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /anomaly/session/open',
        ))

@_nexus_mcp.tool(name='nexus_anomaly_detection_api_ingest_sliding_window_observations', description="Push a batch of multivariate observations (sliding window data) into an active anomaly detection session. Provide synchronized timestamps in ISO format. Use this to continuously feed data into the session. Do NOT use to replace historical data; it's for live streaming. The flush_partial_window flag forces computation of a partial window. Returns buffer status and number of ready windows.")
async def ingest_sliding_window_observations(session_id: Annotated[str, Field(..., description='The session ID returned by open_multivariate_anomaly_session.', min_length=1)], observations: Annotated[list[list[float]], Field(..., description='2D array of observations, shape [timesteps, variables] matching variable order.')], timestamps_iso: Annotated[list[str], Field(..., description='ISO 8601 timestamps corresponding to each row of observations.')], flush_partial_window: Annotated[bool, Field(..., description='If true, force computation on any buffered partial window.')], ctx: Context) -> dict[str, Any]:
    """Ingest Observations"""
    _nexus_path = '/anomaly/session/ingest'.format()
    params = {"session_id": session_id, "observations": observations, "timestamps_iso": timestamps_iso, "flush_partial_window": flush_partial_window}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_anomaly_detection_api_ingest_sliding_window_observations', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /anomaly/session/ingest',
        ))

@_nexus_mcp.tool(name='nexus_anomaly_detection_api_stream_causal_anomaly_events', description="Retrieve detected causal anomaly events from a session's computation pipeline. Specify NMI anomaly threshold and minimum net Transfer Entropy flow to filter events. Use this to pull detected anomalies that indicate a causal origin. Do NOT use for raw anomaly scores only; for causal map resolution, use resolve_anomaly_causal_map. Returns array of events with origin node and propagation depth.")
async def stream_causal_anomaly_events(session_id: Annotated[str, Field(..., description='The session ID.', min_length=1)], nmi_anomaly_threshold: Annotated[float, Field(..., description='Threshold for the NMI-based anomaly score (0 to 1).', ge=0, le=1)], te_net_flow_min: Annotated[float, Field(..., description='Minimum net Transfer Entropy flow (TE_forward - TE_reverse) to consider an edge as causal.', ge=0)], max_events: Annotated[float, Field(..., description='Maximum number of events to return.', ge=1)], ctx: Context) -> dict[str, Any]:
    """Stream Causal Events"""
    _nexus_path = '/anomaly/events/stream'.format()
    params = {"session_id": session_id, "nmi_anomaly_threshold": nmi_anomaly_threshold, "te_net_flow_min": te_net_flow_min, "max_events": max_events}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_anomaly_detection_api_stream_causal_anomaly_events', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /anomaly/events/stream',
        ))

@_nexus_mcp.tool(name='nexus_anomaly_detection_api_resolve_anomaly_causal_map', description='Resolve the detailed causal graph for a specific anomaly event previously detected. Set include_reverse_te to true if you need both directions of Transfer Entropy; set min_net_te_weight to filter weak edges. Use this to understand propagation paths from the origin node. Do NOT use for browsing events; use stream_causal_anomaly_events first to get event_ids. Returns adjacency and edge weights.')
async def resolve_anomaly_causal_map(session_id: Annotated[str, Field(..., description='The session ID.', min_length=1)], event_id: Annotated[str, Field(..., description='The event ID from stream_causal_anomaly_events.', min_length=1)], include_reverse_te: Annotated[bool, Field(..., description='If true, include reverse TE values in the output.')], min_net_te_weight: Annotated[float, Field(..., description='Minimum net TE weight for an edge to be included.', ge=0)], ctx: Context) -> dict[str, Any]:
    """Resolve Causal Map"""
    _nexus_path = '/anomaly/causal-map/resolve'.format()
    params = {"session_id": session_id, "event_id": event_id, "include_reverse_te": include_reverse_te, "min_net_te_weight": min_net_te_weight}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_anomaly_detection_api_resolve_anomaly_causal_map', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /anomaly/causal-map/resolve',
        ))

@_nexus_mcp.tool(name='nexus_anomaly_detection_api_close_anomaly_session', description='Terminate an active session and optionally export a summary of all observations and events. Use this when monitoring ends. Do NOT call on already closed sessions. After closure, the session_id is invalidated and data is archived.')
async def close_anomaly_session(session_id: Annotated[str, Field(..., description='The session ID to close.', min_length=1)], export_session_summary: Annotated[bool, Field(..., description='If true, include a session summary in the response.')], ctx: Context) -> dict[str, Any]:
    """Close Anomaly Session"""
    _nexus_path = '/anomaly/session/close'.format()
    params = {"session_id": session_id, "export_session_summary": export_session_summary}
    _nexus_call_t0 = time.monotonic()
    _nexus_call_success = True
    try:
        return await _nexus_mcp_call_core('POST', _nexus_path, params, headers=None)
    except Exception:
        _nexus_call_success = False
        raise
    finally:
        _nexus_call_latency_ms = int((time.monotonic() - _nexus_call_t0) * 1000)
        asyncio.create_task(_nexus_log_mcp_call_event(
            'nexus_anomaly_detection_api_close_anomaly_session', _nexus_call_success, _nexus_call_latency_ms, ctx,
            route_key='POST /anomaly/session/close',
        ))


# Crea el sub-app ASGI de streamable HTTP -- DEBE llamarse antes de
# poder acceder a _nexus_mcp.session_manager (se crea de forma
# perezosa, ver docstring del modulo).
# Se monta en "/" (no en "/mcp"): streamable_http_app() YA expone su
# propia ruta interna en "/mcp" -- montarlo de nuevo en "/mcp" duplica
# el path a "/mcp/mcp" y da 404 (bug real encontrado probando esto en
# runtime con un cliente MCP de verdad, no algo teorico).
_nexus_mcp_asgi_app = _nexus_mcp.streamable_http_app()

# --- NEXUS: PATCH mcp_lifespan_composition_fix ---
# @app.on_event() SOLO se ejecuta si Starlette uso _DefaultLifespan --
# eso pasa unicamente cuando el `app = FastAPI(...)` que genero el LLM
# NO paso su propio parametro `lifespan=`. Si el LLM SI definio uno
# (tipico en assets con estado -- ej. cleanup de conexiones WebSocket
# al shutdown), Starlette usa ESE callable exclusivamente y los
# handlers @app.on_event quedan sin ejecutarse -- sin warning, sin
# error, el server bootea limpio ("Application startup complete") y
# el primer request a /mcp explota con "RuntimeError: Task group is
# not initialized." (confirmado contra Router.__init__ de Starlette:
# lifespan=None -> _DefaultLifespan(self) [dispara on_event],
# lifespan=<callable> -> se usa ESE, on_event nunca corre). Bug real
# encontrado en produccion 2026-07-25 (asset "ws", que define su
# propio lifespan para cerrar sesiones WebSocket abiertas).
#
# Fix: envolver el lifespan_context que Starlette YA construyo (sea
# _DefaultLifespan o el custom del LLM) en vez de competir con el
# via @app.on_event. Funciona en ambos casos sin parsear ni tocar
# el `app = FastAPI(...)` original.
_nexus_prev_lifespan_context = app.router.lifespan_context


@asynccontextmanager
async def _nexus_combined_lifespan(app):
    async with _nexus_mcp.session_manager.run():
        async with _nexus_prev_lifespan_context(app):
            yield


app.router.lifespan_context = _nexus_combined_lifespan


# --- NEXUS: receptor real de webhooks de Stripe (inyectado por forge_output_saver_v6) ---
from fastapi import Request as _NexusStripeRequest
from fastapi.responses import JSONResponse as _NexusStripeJSONResponse

@app.post("/stripe/webhook")
async def _nexus_stripe_webhook(request: _NexusStripeRequest):
    import os as _nexus_os
    import stripe as _nexus_stripe
    _webhook_secret = _nexus_os.environ.get("STRIPE_WEBHOOK_SECRET")
    if not _webhook_secret:
        return _NexusStripeJSONResponse(
            status_code=404,
            content={"error": "stripe webhook not configured"},
        )
    _secret_key = _nexus_os.environ.get("STRIPE_SECRET_KEY")
    if _secret_key:
        _nexus_stripe.api_key = _secret_key
    _payload = await request.body()
    _sig_header = request.headers.get("stripe-signature", "")
    try:
        _event = _nexus_stripe.Webhook.construct_event(
            _payload, _sig_header, _webhook_secret
        )
    except ValueError:
        return _NexusStripeJSONResponse(
            status_code=400, content={"error": "invalid payload"}
        )
    except _nexus_stripe.error.SignatureVerificationError:
        return _NexusStripeJSONResponse(
            status_code=400, content={"error": "invalid signature"}
        )
    # NEXUS: solo verificacion + ack real -- el gate de
    # autorizacion por estado de suscripcion y la politica de
    # dunning/downgrade son decisiones de producto pendientes,
    # no implementadas a proposito (ver CLAUDE.md).
    print(
        f"[NEXUS_STRIPE_WEBHOOK] type={_event['type']} "
        f"id={_event['id']}"
    )
    return _NexusStripeJSONResponse(
        status_code=200, content={"received": True}
    )


app.mount("/", _nexus_mcp_asgi_app)