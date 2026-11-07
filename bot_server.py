"""Serve one strategy through the participant bundle's referee protocol."""

import os
import traceback
from threading import Lock

from game import Action, AgentConfig, HandResult, Observation


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None):
    """Report an exception from your code to the judge (with traceback)."""
    from starlette.responses import JSONResponse
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def create_app(strategy: dict):
    """Expose the referee routes without importing the HTTP stack into training workers."""
    strategy.setdefault("MEMORY", {})
    strategy.setdefault("STATUS", {"started": False})
    strategy.setdefault("TOKEN", os.environ.get("AGENT_TOKEN") or None)
    return _LazyApp(strategy)


class _LazyApp:
    def __init__(self, strategy):
        self.strategy = strategy
        self.app = None
        self.lock = Lock()

    def _load(self):
        if self.app is None:
            with self.lock:
                if self.app is None:
                    self.app = _referee_app(self.strategy)
        return self.app

    async def __call__(self, scope, receive, send):
        # Uvicorn sends lifespan.startup before accepting any referee requests.
        await self._load()(scope, receive, send)

    def __getattr__(self, name):
        return getattr(self._load(), name)


def _referee_app(strategy):
    # The four referee routes exchange plain JSON. Using Starlette directly
    # avoids importing the validation and documentation stack at every startup.
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    memory = strategy["MEMORY"]
    status = strategy["STATUS"]

    async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
        """Check the token (and that the half has started) BEFORE reading the body.

        Checking first means a request without the right token costs your bot
        almost nothing, so nobody else can slow it down by sending junk.
        Returns (body, None) when the request is fine, or ({}, error response).
        """
        if strategy["TOKEN"] and request.headers.get("x-agent-token") != strategy["TOKEN"]:
            return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
        if not status["started"] and path != "/new-half":
            return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
        try:
            body = await request.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
        return body, None


    def _deliver_results(raw_results) -> list:
        """Call on_hand_end() for each finished hand the judge sent.
        Returns a list of error reports (one per hand that crashed)."""
        errors = []
        for raw in raw_results or []:
            try:
                strategy["on_hand_end"](HandResult.from_dict(raw), memory)
            except Exception as exc:
                errors.append(_error_info(exc))
        return errors


    async def info(request: Request):
        """Health check: tells the judge who this bot is."""
        return JSONResponse({"participant": strategy["PARTICIPANT"], "agent_name": strategy["AGENT_NAME"]})


    async def route_new_half(request: Request):
        """Wipe memory and call your new_half(). The body is {"config": {...}}."""
        body, refused = await _read_request(request, "/new-half")
        if refused:
            return refused
        memory.clear()
        status["started"] = True
        try:
            strategy["new_half"](AgentConfig.from_dict(body.get("config", {})), memory)
        except Exception as exc:
            return _crash(exc)
        return JSONResponse({"ok": True})


    async def route_act(request: Request):
        """Deliver finished hands, then turn the JSON into an Observation, call
        choose_action(), and send the answer back."""
        body, refused = await _read_request(request, "/act")
        if refused:
            return refused
        hand_end_errors = _deliver_results(body.get("hand_results"))
        try:
            obs = Observation.from_dict(body["observation"])
            action = strategy["choose_action"](obs, memory)
        except Exception as exc:
            return _crash(exc, hand_end_errors)
        # Send the action's name; anything strange is passed on for the judge to reject.
        if isinstance(action, Action):
            answer = action.name
        elif action is None or isinstance(action, (str, int, float, bool)):
            answer = action
        else:
            answer = repr(action)
        return JSONResponse({"action": answer, "hand_end_errors": hand_end_errors})


    async def route_hand_end(request: Request):
        """Call on_hand_end() for the last finished hands of a half."""
        body, refused = await _read_request(request, "/hand-end")
        if refused:
            return refused
        return JSONResponse({"ok": True, "hand_end_errors": _deliver_results(body.get("results"))})

    return Starlette(routes=[
        Route("/", info, methods=["GET"]),
        Route("/new-half", route_new_half, methods=["POST"]),
        Route("/act", route_act, methods=["POST"]),
        Route("/hand-end", route_hand_end, methods=["POST"]),
    ])
