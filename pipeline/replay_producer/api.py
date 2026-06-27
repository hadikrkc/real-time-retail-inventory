"""
FastAPI control wrapper around the M5 replay producer.

Endpoints:
  POST /start        — start replay (params: speed_days_per_min, start_day, end_day)
  POST /stop         — stop replay
  GET  /status       — current state (running, progress, events/sec)
  POST /reset        — clean-slate reset (level=soft|hard)
                       soft: truncate analytics tables, reset consumer offsets
                       hard: + truncate sales_events, delete/recreate Kafka topics
"""

import os
import json
import threading
import time

import psycopg2
from confluent_kafka.admin import AdminClient, NewTopic
from fastapi import FastAPI
from producer import delivery_report, iter_day_events, load_data

KAFKA_BROKERS = os.environ.get("KAFKA_BROKERS", "localhost:9092")
TOPIC         = "retail.sales.events"

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "retail")
DB_USER = os.environ.get("DB_USER", "retail")
DB_PASS = os.environ.get("DB_PASS", "retail")

KAFKA_TOPICS      = ["retail.sales.events", "retail.anomaly.alerts", "retail.forecast.results"]
CONSUMER_GROUPS   = ["timescale-sink"]

app = FastAPI(title="Replay Producer API")

_state: dict = {
    "running": False,
    "current_date": None,
    "total_events": 0,
    "events_per_sec": 0.0,
    "error": None,
}
_stop_event = threading.Event()

_reset_state: dict = {
    "running": False,
    "done": False,
    "level": None,
    "steps": [],
    "error": None,
}


def _replay_worker(speed_sec: float, start_day: int, end_day: int) -> None:
    from confluent_kafka import Producer

    conf = {
        "bootstrap.servers": KAFKA_BROKERS,
        "compression.type": "lz4",
        "batch.size": 256 * 1024,
        "linger.ms": 20,
        "acks": "1",
        "queue.buffering.max.messages": 100_000,
    }
    producer = Producer(conf)
    print(f"[producer] connected to {KAFKA_BROKERS}")

    try:
        sales, calendar = load_data()
        total = 0
        t0 = time.perf_counter()

        for event_date, events in iter_day_events(sales, calendar, start_day, end_day):
            if _stop_event.is_set():
                print("[producer] stop requested")
                break

            for event in events:
                producer.produce(
                    topic=TOPIC,
                    key=f"{event['store_id']}_{event['item_id']}",
                    value=json.dumps(event).encode(),
                    on_delivery=delivery_report,
                )
            producer.flush()
            total += len(events)

            elapsed = time.perf_counter() - t0
            _state["current_date"] = str(event_date.date())
            _state["total_events"] = total
            _state["events_per_sec"] = round(total / elapsed, 1) if elapsed > 0 else 0.0

            if speed_sec > 0:
                time.sleep(speed_sec)

        producer.flush()
        print(f"[producer] done — {total:,} events")

    except Exception as exc:
        _state["error"] = str(exc)
        print(f"[producer] error: {exc}")
    finally:
        _state["running"] = False


@app.post("/start")
def start(speed_days_per_min: float = 10.0, start_day: int = 1, end_day: int = 1941):
    if _state["running"]:
        return {"status": "already_running", **_state}

    speed_sec = 60.0 / speed_days_per_min if speed_days_per_min > 0 else 0.0

    _stop_event.clear()
    _state.update({"running": True, "total_events": 0,
                   "events_per_sec": 0.0, "error": None, "current_date": None})

    threading.Thread(
        target=_replay_worker,
        args=(speed_sec, start_day, end_day),
        daemon=True,
    ).start()

    return {"status": "started", "speed_sec_per_day": speed_sec}


@app.post("/stop")
def stop():
    _stop_event.set()
    _state["running"] = False
    return {"status": "stopped"}


@app.get("/status")
def status():
    return _state


def _reset_worker(level: str) -> None:
    _reset_state.update({"running": True, "done": False, "level": level,
                          "steps": [], "error": None})
    steps = _reset_state["steps"]

    # ── 1. Truncate DB tables ─────────────────────────────────────────────────
    tables = ["anomaly_alerts", "sales_features", "forecast_results", "pipeline_metrics"]
    if level == "hard":
        tables.append("sales_events")

    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASS,
            connect_timeout=10,
        )
        # SET lock_timeout so TRUNCATE doesn't wait forever for locks held by pollers.
        # If a table is locked, wait up to 15s then raise; client retries on re-run.
        # Use DELETE (ROW EXCLUSIVE) instead of TRUNCATE (ACCESS EXCLUSIVE).
        # DELETE is compatible with concurrent SELECT queries from the streaming
        # services — no lock contention. Runs async, may take 30-60s for large tables.
        conn.autocommit = True
        with conn.cursor() as cur:
            for tbl in tables:
                cur.execute(f"DELETE FROM {tbl}")
                print(f"[reset] Cleared: {tbl}")
        conn.close()
        steps.append({"db_cleared": tables, "status": "ok"})
        print(f"[reset] DB clear complete: {tables}")
    except Exception as exc:
        steps.append({"db_truncate": "failed", "error": str(exc)})
        print(f"[reset] DB truncate error: {exc}")

    # ── 2. Kafka topic delete+recreate (hard only) ────────────────────────────
    if level == "hard":
        try:
            admin = AdminClient({
                "bootstrap.servers": KAFKA_BROKERS,
                "socket.timeout.ms": 10000,
                "request.timeout.ms": 15000,
            })

            del_fs = admin.delete_topics(KAFKA_TOPICS, operation_timeout=15)
            for topic, fut in del_fs.items():
                try:
                    fut.result(timeout=15)
                    print(f"[reset] deleted: {topic}")
                except Exception as ex:
                    print(f"[reset] delete {topic}: {ex}")

            time.sleep(3)

            new_topics = [
                NewTopic(t, num_partitions=10, replication_factor=1)
                for t in KAFKA_TOPICS
            ]
            crt_fs = admin.create_topics(new_topics, operation_timeout=15)
            for topic, fut in crt_fs.items():
                try:
                    fut.result(timeout=15)
                    print(f"[reset] created: {topic}")
                except Exception as ex:
                    print(f"[reset] create {topic}: {ex}")

            steps.append({"kafka_topics": KAFKA_TOPICS, "status": "deleted+recreated"})
        except Exception as exc:
            steps.append({"kafka_topics": "failed", "error": str(exc)})
            print(f"[reset] Kafka error: {exc}")

    _reset_state.update({"running": False, "done": True})
    print(f"[reset] Complete. Steps: {steps}")


@app.post("/reset")
def reset(level: str = "soft"):
    """
    Async reset — returns immediately, runs in background.
    soft: truncate analytics tables only.
    hard: + truncate sales_events + delete/recreate Kafka topics.
    Check /reset-status for progress.
    """
    if _reset_state.get("running"):
        return {"status": "already_running", **_reset_state}

    # Stop producer
    _stop_event.set()
    _state.update({"running": False, "total_events": 0,
                   "events_per_sec": 0.0, "current_date": None, "error": None})

    threading.Thread(target=_reset_worker, args=(level,), daemon=True).start()
    return {"status": "reset_started", "level": level,
            "message": "Reset running in background. Poll /reset-status for completion."}


@app.get("/reset-status")
def reset_status():
    return _reset_state
