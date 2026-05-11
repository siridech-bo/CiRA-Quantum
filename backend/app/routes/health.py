"""Liveness endpoint. The GPU-availability check arrives with Phase 4."""

from __future__ import annotations

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)


@health_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "version": "0.1.0"})
