# fastapi-vss, Apache-2.0 license
# Filename: tests/test_websocket.py
# Description: Tests for WebSocket job status endpoint
import json
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from starlette.testclient import TestClient

import sys

sys.path.append("src")


def _make_job(finished=False, failed=False, return_val=None):
    job = MagicMock()
    job.is_finished = finished
    job.is_failed = failed
    job.return_value.return_value = return_val
    return job


@pytest.fixture()
def client():
    """Return a TestClient with mocked Redis / RQ internals."""
    with (
        patch("app.main.config", {"testproject": {"redis_host": "localhost", "redis_port": 6379, "device": "cpu"}}),
        patch("app.main.connections", {"testproject": MagicMock()}),
        patch("app.main.queues", {"testproject": MagicMock()}),
        patch("app.main.DEFAULT_PROJECT", "testproject"),
    ):
        from app.main import app  # import after patches are applied

        yield TestClient(app)


class TestWebSocketJobResult:
    def test_invalid_project(self, client):
        with client.websocket_connect("/ws/predict/job/some-job-id/bad-project") as ws:
            msg = json.loads(ws.receive_text())
        assert msg["status"] == "error"
        assert "Invalid project" in msg["message"]

    def test_job_not_found(self, client):
        with (
            patch("app.main.Job.exists", return_value=False),
        ):
            with client.websocket_connect("/ws/predict/job/missing-id/testproject") as ws:
                msg = json.loads(ws.receive_text())
        assert msg["status"] == "error"
        assert "does not exist" in msg["message"]

    def test_job_finished_immediately(self, client):
        finished_job = _make_job(finished=True, return_val={"scores": [0.9]})
        with (
            patch("app.main.Job.exists", return_value=True),
            patch("app.main.Job.fetch", return_value=finished_job),
        ):
            with client.websocket_connect("/ws/predict/job/done-id/testproject") as ws:
                msg = json.loads(ws.receive_text())
        assert msg["status"] == "done"
        assert msg["result"] == {"scores": [0.9]}

    def test_job_failed(self, client):
        failed_job = _make_job(failed=True)
        with (
            patch("app.main.Job.exists", return_value=True),
            patch("app.main.Job.fetch", return_value=failed_job),
        ):
            with client.websocket_connect("/ws/predict/job/failed-id/testproject") as ws:
                msg = json.loads(ws.receive_text())
        assert msg["status"] == "failed"
        assert "failed-id" in msg["message"]

    def test_job_pending_then_done(self, client):
        """Job starts pending, then becomes finished on the second poll."""
        pending_job = _make_job(finished=False, failed=False)
        finished_job = _make_job(finished=True, return_val={"scores": [0.8]})

        fetch_side_effects = [pending_job, finished_job]

        with (
            patch("app.main.Job.exists", return_value=True),
            patch("app.main.Job.fetch", side_effect=fetch_side_effects),
            patch("app.main.WS_POLL_INTERVAL", 0),  # no sleep in tests
        ):
            with client.websocket_connect("/ws/predict/job/slow-id/testproject") as ws:
                first = json.loads(ws.receive_text())
                second = json.loads(ws.receive_text())

        assert first["status"] == "pending"
        assert second["status"] == "done"
        assert second["result"] == {"scores": [0.8]}

    def test_embedding_job_result(self, client):
        """Test that embedding results are properly returned."""
        embedding_result = {"filenames": ["test.jpg"], "embeddings": [[0.1] * 768]}
        finished_job = _make_job(finished=True, return_val=embedding_result)

        with (
            patch("app.main.Job.exists", return_value=True),
            patch("app.main.Job.fetch", return_value=finished_job),
        ):
            with client.websocket_connect("/ws/predict/job/embed-id/testproject") as ws:
                msg = json.loads(ws.receive_text())

        assert msg["status"] == "done"
        assert msg["result"]["filenames"] == ["test.jpg"]
        assert len(msg["result"]["embeddings"]) == 1
        assert len(msg["result"]["embeddings"][0]) == 768

    def test_multiple_pending_updates(self, client):
        """Job sends multiple pending updates before completing."""
        pending_job = _make_job(finished=False, failed=False)
        finished_job = _make_job(finished=True, return_val={"predictions": [["A", "B"]]})

        fetch_side_effects = [pending_job, pending_job, pending_job, finished_job]

        with (
            patch("app.main.Job.exists", return_value=True),
            patch("app.main.Job.fetch", side_effect=fetch_side_effects),
            patch("app.main.WS_POLL_INTERVAL", 0),
        ):
            with client.websocket_connect("/ws/predict/job/multi-id/testproject") as ws:
                messages = []
                for _ in range(4):
                    messages.append(json.loads(ws.receive_text()))

        assert messages[0]["status"] == "pending"
        assert messages[1]["status"] == "pending"
        assert messages[2]["status"] == "pending"
        assert messages[3]["status"] == "done"
        assert messages[3]["result"] == {"predictions": [["A", "B"]]}


class TestWebSocketTimeoutIsWallClock:
    """WS_MAX_WAIT must be a real elapsed-time limit, not a count of loop iterations."""

    def test_timeout_message_reports_elapsed_seconds(self, client):
        pending_job = _make_job(finished=False, failed=False)

        with (
            patch("app.main.Job.exists", return_value=True),
            patch("app.main.Job.fetch", return_value=pending_job),
            patch("app.main.WS_MAX_WAIT", 0),  # expire on the first check
        ):
            with client.websocket_connect("/ws/predict/job/slow-id/testproject") as ws:
                msg = json.loads(ws.receive_text())

        assert msg["status"] == "error"
        assert "Timed out waiting for job after" in msg["message"]

    def test_slow_redis_still_counts_against_the_budget(self, client):
        """
        Regression test for tracking elapsed time as `elapsed += WS_POLL_INTERVAL`.

        That counted sleeps only and ignored how long each Redis round-trip took, so with a
        slow Redis (exactly when a bound matters) the real limit drifted arbitrarily far past
        WS_MAX_WAIT -- and with WS_POLL_INTERVAL patched to 0, as below, it never advanced at
        all and the loop ran forever.
        """
        pending_job = _make_job(finished=False, failed=False)

        def slow_fetch(*args, **kwargs):
            time.sleep(0.15)
            return pending_job

        with (
            patch("app.main.Job.exists", return_value=True),
            patch("app.main.Job.fetch", side_effect=slow_fetch),
            patch("app.main.WS_POLL_INTERVAL", 0),
            patch("app.main.WS_MAX_WAIT", 0.2),
        ):
            with client.websocket_connect("/ws/predict/job/slow-redis/testproject") as ws:
                statuses = []
                for _ in range(50):  # bounded so a regression fails rather than hangs
                    statuses.append(json.loads(ws.receive_text())["status"])
                    if statuses[-1] != "pending":
                        break

        assert statuses[-1] == "error", f"never timed out; got {len(statuses)} frames"


class TestBlockingRedisCallsRunOffTheEventLoop:
    """
    Job.exists / Job.fetch / job.return_value() are blocking redis-py calls.

    Running them directly in the async endpoint stalls the single event loop shared by every
    other connection this process serves -- including the heartbeat frames those clients use
    to distinguish a slow job from a dead connection. They must be dispatched to a thread.
    """

    def test_job_calls_are_dispatched_to_a_worker_thread(self, client):
        seen = {}
        finished_job = MagicMock()
        finished_job.is_finished = True
        finished_job.is_failed = False

        def record(name, result):
            def _inner(*args, **kwargs):
                seen[name] = threading.current_thread().name
                return result

            return _inner

        finished_job.return_value = record("return_value", {"embeddings": [[0.5]]})

        with (
            patch("app.main.Job.exists", side_effect=record("exists", True)),
            patch("app.main.Job.fetch", side_effect=record("fetch", finished_job)),
        ):
            with client.websocket_connect("/ws/predict/job/thread-id/testproject") as ws:
                msg = json.loads(ws.receive_text())

        assert msg["status"] == "done"
        assert msg["result"] == {"embeddings": [[0.5]]}
        # asyncio.to_thread runs work on the default executor, whose threads are named
        # "asyncio_N". Anything left inline would report the event loop's own thread.
        for name in ("exists", "fetch", "return_value"):
            assert name in seen, f"{name} was never called"
            assert seen[name].startswith("asyncio_"), (
                f"{name} ran on {seen[name]!r}, i.e. inline on the event loop"
            )


if __name__ == "__main__":
    pytest.main()
