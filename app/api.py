# Built-in
from __future__ import annotations
from dataclasses import asdict

# External
from flask import Flask, Response, request, jsonify

# Internal
from app.normalizer import normalize_context
from app.bootstrap import build_application

app = Flask(__name__)
app_runtime = build_application()


@app.route("/context", methods=["POST"])
def context_route():
    return context()


def context() -> tuple[Response, int] | Response:
    task = request.files.get("original_task")
    blueprint = request.files.get("codesnap")
    overview = request.files.get("overview")

    if task is None or blueprint is None or overview is None:
        return jsonify({
            "error": "Expected multipart fields: original_task, codesnap, overview"
        }), 400

    try:
        ctx = normalize_context(task, blueprint, overview)
        result = app_runtime.analyze_context(ctx)
        return jsonify(asdict(result))

    except (TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid payload: {exc}"}), 400

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
