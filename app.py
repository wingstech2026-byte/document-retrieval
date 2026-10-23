"""Vercel (and other WSGI hosts) entry point: exposes the Flask application as `app`.

Environment variables
    MONGO_URI          required: a MongoDB connection string, e.g. a MongoDB Atlas URI
    DOCRETRIEVAL_DB    database name (default "docretrieval")
    READ_ONLY          "1" (default) = search/read/compare/download only; "0" also allows add and delete.
                       There is no login, so keep the default on any public deployment.
"""
import os

from flask import Flask

import docretrieval as dr
from webapp import create_app


def _setup_problem(message: str) -> Flask:
    """A tiny app that explains what is missing instead of crashing with a bare 500."""
    broken = Flask(__name__)

    @broken.route("/", defaults={"path": ""})
    @broken.route("/<path:path>")
    def explain(path):
        return (f"<!doctype html><title>Setup needed</title><h1>Setup needed</h1><p>{message}</p>", 503)
    return broken


def build():
    uri = os.environ.get("MONGO_URI")
    if not uri:
        return _setup_problem("Set the <code>MONGO_URI</code> environment variable to a MongoDB connection string "
                              "(for example from MongoDB Atlas) and redeploy.")
    read_only = os.environ.get("READ_ONLY", "1").strip().lower() not in ("0", "false", "no")
    try:
        # Indexes are created by whoever loads the data (the CLI does it); a read-only
        # deployment does not need permission to create them.
        store = dr.DocumentStore(uri, os.environ.get("DOCRETRIEVAL_DB", dr.DEFAULT_DB), create_indexes=not read_only)
    except Exception as e:  # unreachable server, bad credentials, IP not allow-listed, ...
        return _setup_problem(f"Could not connect to MongoDB ({type(e).__name__}). Check the connection string, "
                              "the database user's password, and that Atlas allows connections from anywhere "
                              "(Network Access: 0.0.0.0/0), since Vercel's IP addresses change.")
    return create_app(store, read_only=read_only)


app = build()
