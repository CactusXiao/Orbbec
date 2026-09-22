"""Personnel accounting in media seconds. Never scans NAS files on a page request."""
from __future__ import annotations

import math
from collections import defaultdict


METRICS = ("capture_seconds", "qc_seconds", "qc_error_seconds", "qc_false_positive_seconds", "label_seconds")


def union(ranges, limit=None):
    out = []
    clean = []
    for a, b in ranges:
        a, b = max(0, int(a)), int(b)
        if limit is not None:
            b = min(b, limit - 1)
        if b >= a:
            clean.append((a, b))
    for a, b in sorted(clean):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(b, out[-1][1]))
        else:
            out.append((a, b))
    return out


def intersection(left, right):
    left, right = union(left), union(right)
    i = j = 0
    out = []
    while i < len(left) and j < len(right):
        a, b = max(left[i][0], right[j][0]), min(left[i][1], right[j][1])
        if a <= b:
            out.append((a, b))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return out


def frames_to_ranges(frames):
    return union((f, f) for f in frames if type(f) is int and f >= 0)


def segment_ranges(segments):
    pairs = []
    for s in segments or []:
        try:
            a, b = (s.get("start_frame", s.get("start")), s.get("end_frame", s.get("end"))) if isinstance(s, dict) else s[:2]
            if type(a) is int and type(b) is int:
                pairs.append((a, b))
        except (TypeError, ValueError):
            continue
    return union(pairs)


def positive(value):
    try:
        n = float(value)
        return n if math.isfinite(n) and n > 0 and not isinstance(value, bool) else None
    except (ValueError, TypeError):
        return None


def timeline(count, layers):
    """Closed frame ranges; later layers take precedence. O(intervals log intervals)."""
    events = defaultdict(list)
    events[0]
    events[count]
    for priority, (state, ranges) in enumerate(layers):
        for a, b in union(ranges, count):
            events[a].append((priority, state, 1))
            events[b + 1].append((priority, state, -1))
    active, out = {}, []
    positions = sorted(events)
    for index, start in enumerate(positions[:-1]):
        for priority, state, delta in events[start]:
            if delta > 0:
                active[priority] = state
            else:
                active.pop(priority, None)
        end = positions[index + 1] - 1
        state = active[max(active)] if active else "gray"
        if out and out[-1]["state"] == state:
            out[-1]["end_frame"] = end
        elif end >= start:
            out.append(dict(start_frame=start, end_frame=end, state=state))
    return out


def snapshot(store, usernames=(), reservations=()):
    with store.connect() as conn:
        conn.execute("BEGIN")
        episodes = [store._row_to_episode(r) for r in conn.execute("SELECT * FROM episodes")]
        jobs = [store._row_to_job(r) for r in conn.execute("SELECT * FROM jobs WHERE type IN ('qc', 'manual_label') ORDER BY created_at, job_id")]
        segments = [dict(r) for r in conn.execute("SELECT episode_id, start_frame, end_frame FROM segments")]
        decisions = [dict(r) for r in conn.execute("SELECT * FROM label_frame_decisions ORDER BY updated_at, job_id")]
    by_episode = {e["episode_id"]: e for e in episodes}
    for reservation in reservations:
        if reservation.get("status") != "confirmed":
            continue
        eid = reservation["reservation_id"]
        episode = by_episode.setdefault(eid, dict(episode_id=eid, subject_id=reservation.get("subject_id", ""),
            task_name=reservation.get("task_name", ""), episode_index=reservation.get("episode_number"),
            status="captured", metadata={}))
        episode["metadata"] = dict(episode.get("metadata") or {})
        episode["metadata"].setdefault("source", "collection_api")
        for key in ("duration_seconds", "fps"):
            if reservation.get(key) is not None:
                episode["metadata"][key] = reservation[key]
        if not episode["metadata"].get("collection_operator_id"):
            episode["metadata"]["collection_operator_id"] = reservation.get("confirmed_by") or reservation.get("operator_id")
        if not episode.get("frame_count"):
            episode["frame_count"] = reservation.get("frame_count")
    jobs_by_ep, seg_by_ep, dec_by_job = defaultdict(list), defaultdict(list), defaultdict(list)
    for job in jobs:
        jobs_by_ep[job["episode_id"]].append(job)
    for segment in segments:
        seg_by_ep[segment["episode_id"]].append(segment)
    for decision in decisions:
        dec_by_job[decision["job_id"]].append(decision)
    people, records, contributions = {}, [], defaultdict(list)

    def person(name):
        name = str(name or "未归属账号")
        if name not in people:
            people[name] = dict(username=name, **{k: 0.0 for k in METRICS},
                                capture_count=0, qc_count=0, label_count=0, unknown_duration_count=0)
        return name

    for name in usernames:
        person(name)
    unknown = defaultdict(set)
    for eid, episode in by_episode.items():
        meta = episode.get("metadata") or {}
        ep_jobs = jobs_by_ep[eid]
        bad = segment_ranges(seg_by_ep[eid])
        declared_count = int(positive(episode.get("frame_count")) or 0)
        # Development/legacy jobs sometimes stored len(selected_frames) here.
        # Infer a display extent only; never treat a sparse selection as the full duration.
        extents = [b for a, b in bad]
        extents += [f for j in ep_jobs for f in (j.get("payload", {}).get("frames") or []) if type(f) is int]
        minimum_extent = max(extents, default=-1) + 1
        count = max(declared_count, minimum_extent)
        duration = positive(meta.get("duration_seconds"))
        fps = positive(meta.get("fps") or meta.get("frame_rate"))
        authoritative_count = declared_count if declared_count and declared_count >= minimum_extent else None
        seconds_per_frame = duration / authoritative_count if duration and authoritative_count else (1 / fps if fps else None)
        if duration is None and seconds_per_frame and authoritative_count:
            duration = authoritative_count * seconds_per_frame
        full = [(0, count - 1)] if count else []
        latest_qc = next((j for j in reversed(ep_jobs) if j["type"] == "qc" and j["status"] == "succeeded"), None)
        ep_decisions = {}
        for j in ep_jobs:
            if j["type"] == "manual_label" and j["status"] != "canceled":
                for d in dec_by_job[j["job_id"]]:
                    if d["frame"] not in ep_decisions or d["updated_at"] >= ep_decisions[d["frame"]]["updated_at"]:
                        ep_decisions[d["frame"]] = d
        false_ranges = frames_to_ranges(d["frame"] for d in ep_decisions.values() if d["decision"] == "no_error")

        def qc_bad(job):
            result = job.get("result") or {}
            if result.get("bad_episode") or result.get("result_type") in {"bad_episode", "abnormal_episode", "episode_abnormal", "episode_exception"}:
                return full
            for key in ("segments", "failed_segments", "qc_failed_segments", "failure_segments"):
                if result.get(key):
                    return union(segment_ranges(result[key]), count)
            if result.get("passed") or result.get("qc_passed"):
                return []
            return union(bad or full, count)

        def add(name, kind, key, ranges, layers, job=None, extra=None):
            name = person(name)
            people[name][kind + "_count"] += 1
            item = dict(username=name, kind=kind, episode_id=eid, task_name=episode.get("task_name", ""),
                subject_id=episode.get("subject_id", ""), episode_index=episode.get("episode_index"),
                storage_name=episode.get("storage_name") or f"episode_{episode.get('episode_index') or eid}",
                status=(job or episode).get("status"), job_id=(job or {}).get("job_id"),
                updated_at=(job or episode).get("updated_at", ""), frame_count=count,
                duration_seconds=duration, seconds_per_frame=seconds_per_frame,
                layers=layers, no_error_ranges=intersection(false_ranges, bad),
                timing_known=seconds_per_frame is not None)
            item.update(extra or {})
            issues = union([r for state, rs in layers if state in {"orange", "red"} for r in rs], count)
            resolved = union([r for state, rs in layers if state == "green" for r in rs], count) if kind == "label" else []
            item["has_issues"] = sum(b-a+1 for a,b in issues) > sum(b-a+1 for a,b in intersection(issues, resolved))
            records.append(item)
            if (key == "capture_seconds" or (key == "qc_seconds" and job and job["status"] == "succeeded")) and duration is not None:
                contributions[(name, eid, key)] = [(0, 0, duration)]
            elif seconds_per_frame:
                contributions[(name, eid, key)].extend((a, b, seconds_per_frame) for a, b in union(ranges, count))
            elif ranges or kind == "capture":
                unknown[name].add(eid)
            if not seconds_per_frame and count and (kind == "label" or (kind == "qc" and job and job["status"] == "succeeded")):
                unknown[name].add(eid)
            return name

        collector = meta.get("collection_operator_id") or meta.get("collection_confirmed_by")
        if collector or (meta.get("source") == "collection_api" and episode["status"] not in {"planned", "reserved_for_collection"}):
            layers = [("green", full), ("orange", qc_bad(latest_qc))] if latest_qc else []
            add(collector, "capture", "capture_seconds", full, layers)
        for job in ep_jobs:
            result = job.get("result") or {}
            owner = result.get("operator_id") or job.get("lease_owner") or result.get("worker_id")
            if isinstance(owner, str) and owner.startswith("browser:"):
                owner = owner[len("browser:"):].rsplit(":", 1)[0]
            if job["type"] == "qc":
                finished = job["status"] == "succeeded"
                errors = qc_bad(job) if finished else []
                false = intersection(false_ranges, errors)
                name = add(owner, "qc", "qc_seconds", full if finished else [],
                    [("green", full), ("orange", errors), ("red", false)] if finished else [], job,
                    dict(error_ranges=errors, no_error_ranges=false))
                for key, ranges in (("qc_error_seconds", errors), ("qc_false_positive_seconds", false)):
                    if seconds_per_frame:
                        contributions[(name, eid, key)].extend((a, b, seconds_per_frame) for a, b in ranges)
            else:
                ds = dec_by_job[job["job_id"]]
                by_owner = defaultdict(list)
                for d in ds:
                    if d["decision"] != "pending":
                        by_owner[d["operator_id"]].append((d["frame"], d["frame"]))
                if job["status"] == "succeeded":
                    known_frames = {d["frame"] for d in ds}
                    legacy = result.get("frames_completed") or job.get("payload", {}).get("frames") or []
                    by_owner[owner].extend((f, f) for f in legacy if type(f) is int and f not in known_frames)
                    if not legacy and not ds:
                        by_owner[owner].extend(bad)
                by_owner.setdefault(owner, [])
                all_done = union([r for rs in by_owner.values() for r in rs], count)
                for name, done in by_owner.items():
                    add(name, "label", "label_seconds", done,
                        [("orange", bad), ("green", intersection(all_done, bad))], job,
                        dict(operator_completed_ranges=union(done, count)))
    for (name, eid, key), values in contributions.items():
        if values:
            people[name][key] += sum(b - a + 1 for a, b in union((a, b) for a, b, _ in values)) * values[0][2]
    for name, item in people.items():
        item["unknown_duration_count"] = len(unknown[name])
        for key in METRICS:
            item[key] = round(item[key], 6)
    return dict(accounts=sorted(people.values(), key=lambda p: p["username"].casefold()), records=records)


def query_records(data, query):
    def value(key, default=""):
        return (query.get(key) or [default])[0]
    kind = value("kind", "capture")
    username, search, task = value("username"), value("q").casefold(), value("task")
    page = max(1, int(value("page", "1")))
    size = min(50, max(1, int(value("page_size", "20"))))
    rows = [r for r in data["records"] if r["username"] == username and r["kind"] == kind
            and (not search or search in (r["task_name"] + " " + r["subject_id"] + " " + r["storage_name"]).casefold())]
    if value("only_issues") == "1":
        rows = [r for r in rows if r["has_issues"]]
    if "task" in query:
        rows = sorted([r for r in rows if r["task_name"] == task],
                      key=lambda r: (r["subject_id"], r["episode_index"] or 0, r["episode_id"], r["job_id"] or ""))
        selected = []
        for row in rows[(page-1)*size:page*size]:
            item = {k: v for k, v in row.items() if k != "layers"}
            item["timeline"] = timeline(row["frame_count"], row["layers"])
            selected.append(item)
        return dict(total=len(rows), page=page, page_size=size, episodes=selected)
    tasks = {}
    for row in rows:
        group = tasks.setdefault(row["task_name"], dict(task_name=row["task_name"], count=0, issue_count=0))
        group["count"] += 1
        group["issue_count"] += row["has_issues"]
    groups = sorted(tasks.values(), key=lambda t: t["task_name"])
    return dict(total=len(groups), record_count=len(rows), page=page, page_size=size, tasks=groups[(page-1)*size:page*size])


def render_personnel_page():
    from pathlib import Path
    return (Path(__file__).parent / "web" / "personnel.html").read_text(encoding="utf-8")
