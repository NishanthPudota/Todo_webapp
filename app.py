"""
Flask frontend for the Todo Backend API.

Token strategy
──────────────
After login the Flask session stores two tokens server-side:

  session["access_token"]  — short-lived (60 min)
                             sent as  Authorization: Bearer <token>
                             on every API call to FastAPI

  session["refresh_token"] — long-lived (7 days)
                             NEVER sent to FastAPI for data calls
                             used ONLY to silently renew the access token

What happens when the access token expires (60 min):
  1. Flask makes an API call → FastAPI returns 401 "Access token has expired"
  2. api_call() intercepts the 401 and calls _try_refresh() automatically
  3. Flask posts the refresh_token to POST /api/v1/auth/refresh
  4. FastAPI returns a fresh access_token + refresh_token pair
  5. Flask updates the session silently and retries the original request
  6. The user sees their data — no interruption, no forced re-login

What happens when the refresh token also expires (7 days):
  1. POST /api/v1/auth/refresh → FastAPI returns 401
  2. _try_refresh() clears the session and returns None
  3. api_call() raises SessionExpired
  4. @login_required catches it, clears the session, redirects to /login
  5. User sees: "Your session has expired. Please log in again."
"""

import os
from functools import wraps

import requests
from dotenv import load_dotenv
from flask import Flask, flash, g, redirect, render_template, request, session, url_for

# Load .env file so FLASK_SECRET_KEY and API_BASE_URL are available
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "flask-dev-secret-change-in-production")

# Sessions last 8 days — slightly longer than the 7-day refresh token
# so the cookie doesn't expire before the refresh token does
from datetime import timedelta
app.permanent_session_lifetime = timedelta(days=8)

# Base URL of the FastAPI backend — override via .env in production
API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")


# ══════════════════════════════════════════════════════════════════════════════
# Template context processor — injects current_user into every template
# ══════════════════════════════════════════════════════════════════════════════

@app.context_processor
def inject_user():
    """Makes {{ current_user }} available in every Jinja2 template."""
    return {"current_user": session.get("username")}


# ══════════════════════════════════════════════════════════════════════════════
# Custom exception
# ══════════════════════════════════════════════════════════════════════════════

class SessionExpired(Exception):
    """Raised by api_call() when both access and refresh tokens are invalid."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Token refresh helper
# ══════════════════════════════════════════════════════════════════════════════

def _try_refresh() -> dict | None:
    """
    Attempt to get a new token pair using the stored refresh token.

    Returns the new token dict on success.
    Returns None (and clears the session) on failure.
    """
    refresh_token = session.get("refresh_token")
    if not refresh_token:
        session.clear()
        return None

    try:
        resp = requests.post(
            f"{API_BASE}/auth/refresh",
            json={"refresh_token": refresh_token},
            timeout=10,
        )
    except requests.RequestException:
        # Backend unreachable — don't clear session, maybe a transient network issue
        return None

    if resp.status_code == 200:
        data = resp.json()
        # Rotate both tokens in the session
        session["access_token"] = data["access_token"]
        session["refresh_token"] = data["refresh_token"]
        return data

    # Refresh token is invalid or expired — force re-login
    session.clear()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Authenticated API call helper
# ══════════════════════════════════════════════════════════════════════════════

def api_call(method: str, path: str, retry: bool = True, **kwargs) -> requests.Response:
    """
    Make an authenticated HTTP request to the FastAPI backend.

    - Automatically attaches the Bearer token from the Flask session.
    - On 401 (access token expired): silently refreshes tokens and retries once.
    - On retry failure: raises SessionExpired so @login_required can redirect.

    Args:
        method:  HTTP verb — "get", "post", "patch", "delete"
        path:    API path e.g.  "/tasks/"  or  "/tasks/{id}/done"
        retry:   Internal flag — prevents infinite recursion on the retry pass
        **kwargs: Passed directly to requests.request (json=, timeout=, etc.)

    Usage:
        resp = api_call("get",   "/tasks/")
        resp = api_call("post",  "/tasks/",           json={...})
        resp = api_call("patch", f"/tasks/{id}/done")
        resp = api_call("delete",f"/tasks/{id}")
    """
    access_token = session.get("access_token")

    # Pull out any caller-supplied headers and add Authorization on top
    headers = kwargs.pop("headers", {})
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    # Default timeout so a hung backend doesn't freeze Flask
    kwargs.setdefault("timeout", 10)

    response = requests.request(
        method.upper(),
        f"{API_BASE}{path}",
        headers=headers,
        **kwargs,
    )

    # ── Transparent token refresh on 401 ──────────────────────────────────────
    if response.status_code == 401 and retry:
        new_tokens = _try_refresh()
        if new_tokens:
            # One silent retry with the fresh access token
            headers["Authorization"] = f"Bearer {new_tokens['access_token']}"
            response = requests.request(
                method.upper(),
                f"{API_BASE}{path}",
                headers=headers,
                **kwargs,
            )
        else:
            raise SessionExpired()

    return response


# ══════════════════════════════════════════════════════════════════════════════
# @login_required decorator
# ══════════════════════════════════════════════════════════════════════════════

def login_required(f):
    """
    Route decorator: redirects to /login if no session token is present.
    Also handles SessionExpired (both tokens gone) with a friendly message.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("access_token"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        try:
            return f(*args, **kwargs)
        except SessionExpired:
            session.clear()
            flash("Your session has expired. Please log in again.", "warning")
            return redirect(url_for("login"))
    return decorated


# ══════════════════════════════════════════════════════════════════════════════
# Auth routes
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Root — send logged-in users to tasks, others to login."""
    if session.get("access_token"):
        return redirect(url_for("tasks"))
    return redirect(url_for("login"))


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("access_token"):
        return redirect(url_for("tasks"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        try:
            resp = requests.post(
                f"{API_BASE}/auth/signup",
                json={"username": username, "password": password},
                timeout=10,
            )
        except requests.RequestException:
            flash("Cannot reach the backend server. Is it running?", "danger")
            return render_template("signup.html")

        if resp.status_code == 201:
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        elif resp.status_code == 409:
            flash("That username is already taken.", "danger")
        elif resp.status_code == 422:
            # FastAPI / Pydantic validation errors — each is a dict with a "msg" key
            errors = resp.json().get("detail", [])
            if isinstance(errors, list):
                for err in errors:
                    flash(err.get("msg", "Validation error."), "danger")
            else:
                flash(str(errors), "danger")
        else:
            detail = resp.json().get("detail", "Signup failed.")
            flash(detail, "danger")

    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("access_token"):
        return redirect(url_for("tasks"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()

        try:
            resp = requests.post(
                f"{API_BASE}/auth/login",
                json={"username": username, "password": password},
                timeout=10,
            )
        except requests.RequestException:
            flash("Cannot reach the backend server. Is it running?", "danger")
            return render_template("login.html")

        if resp.status_code == 200:
            data = resp.json()
            # Both tokens go into the server-side Flask session.
            # The browser only ever sees an encrypted session cookie — never the raw JWTs.
            session.permanent       = True   # honour permanent_session_lifetime (8 days)
            session["access_token"]  = data["access_token"]
            session["refresh_token"] = data["refresh_token"]
            session["username"]      = username
            return redirect(url_for("tasks"))

        elif resp.status_code == 401:
            flash("Incorrect username or password.", "danger")
        else:
            flash("Login failed. Please try again.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# ══════════════════════════════════════════════════════════════════════════════
# Task routes  (all protected by @login_required)
# ══════════════════════════════════════════════════════════════════════════════

@app.route("/tasks")
@login_required
def tasks():
    """List all active tasks for the logged-in user."""
    resp = api_call("get", "/tasks/")
    if resp.status_code == 200:
        task_list = resp.json()
    else:
        task_list = []
        flash("Could not load tasks from the server.", "danger")
    return render_template("tasks.html", tasks=task_list, username=session.get("username"))


@app.route("/tasks/create", methods=["GET", "POST"])
@login_required
def create_task():
    """Show and handle the new-task form."""
    if request.method == "POST":
        payload = {
            "title":       request.form.get("title", "").strip(),
            "description": request.form.get("description", "").strip() or None,
            "priority":    request.form.get("priority", "P3"),
        }
        est = request.form.get("estimated_minutes", "").strip()
        if est.isdigit() and int(est) > 0:
            payload["estimated_minutes"] = int(est)

        resp = api_call("post", "/tasks/", json=payload)

        if resp.status_code == 201:
            flash("Task created successfully!", "success")
            return redirect(url_for("tasks"))
        elif resp.status_code == 422:
            errors = resp.json().get("detail", [])
            if isinstance(errors, list):
                for err in errors:
                    flash(err.get("msg", "Validation error."), "danger")
            else:
                flash(str(errors), "danger")
        else:
            flash(resp.json().get("detail", "Could not create task."), "danger")

    return render_template("create_task.html")


@app.route("/tasks/<task_id>/done", methods=["POST"])
@login_required
def mark_done(task_id):
    """Mark a task as completed."""
    resp = api_call("patch", f"/tasks/{task_id}/done")
    if resp.status_code == 200:
        flash("Task marked as done!", "success")
    elif resp.status_code == 404:
        flash("Task not found.", "warning")
    else:
        flash("Could not update task.", "danger")
    return redirect(url_for("tasks"))


@app.route("/tasks/<task_id>/delete", methods=["POST"])
@login_required
def delete_task(task_id):
    """Soft-delete a task (hidden from list, kept in DB)."""
    resp = api_call("delete", f"/tasks/{task_id}")
    if resp.status_code == 204:
        flash("Task deleted.", "info")
    elif resp.status_code == 404:
        flash("Task not found.", "warning")
    else:
        flash("Could not delete task.", "danger")
    return redirect(url_for("tasks"))


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
