import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from task_backend.job_service import JobService
from task_backend.personnel import snapshot, query_records, union, intersection, timeline
from task_backend.server import BackendRuntime, RequestHandler, TaskHTTPServer, TaskInstanceRegistry
from task_backend.workflow_models import WorkflowError
from task_backend.workflow_store import WorkflowStore


class PersonnelTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = WorkflowStore(self.root / "workflow.sqlite3")
        self.service = JobService(self.store)
        self.store.create_or_update_episode(episode_id="ep", subject_id="S01", task_name="pick_cup",
            episode_index=1, status="uploaded", frame_count=100,
            metadata={"collection_operator_id": "collector", "duration_seconds": 10})
        self.store.create_job(job_id="qc", job_type="qc", episode_id="ep", payload={"frames": list(range(100))})
        self.store.complete_job(job_id="qc", result={"operator_id": "reviewer", "passed": False,
            "segments": [{"start_frame": 20, "end_frame": 39}, {"start_frame": 30, "end_frame": 49}]})
        self.store.create_segment(segment_id="seg", episode_id="ep", start_frame=20, end_frame=49)
        self.store.create_job(job_id="label", job_type="manual_label", episode_id="ep", payload={"frames": list(range(20,50))})
        self.store.lease_job(job_type="manual_label", job_id="label", lease_owner="labeler")

    def record(self, frames, decision="no_error", operator="labeler"):
        return self.service.record_label_frames("ep", dict(operator_id=operator, frames=frames, decision=decision))

    def data(self):
        return snapshot(self.store, ["idle"])

    def account(self, name):
        return next(a for a in self.data()["accounts"] if a["username"] == name)

    def rows(self, kind, name):
        return query_records(self.data(), {"username":[name], "kind":[kind], "task":["pick_cup"]})["episodes"]

    def test_overlap_deduplication_and_false_positive(self):
        self.record([25,26,26])
        self.record([35], "corrected")
        self.assertEqual(self.account("collector")["capture_seconds"], 10)
        qc = self.account("reviewer")
        self.assertEqual(qc["qc_seconds"], 10)
        self.assertEqual(qc["qc_error_seconds"], 3)
        self.assertEqual(qc["qc_false_positive_seconds"], .2)
        self.assertEqual(self.account("labeler")["label_seconds"], .3)
        self.assertEqual(self.account("idle")["capture_seconds"], 0)
        row = self.rows("qc", "reviewer")[0]
        self.assertIn(dict(start_frame=25,end_frame=26,state="red"), row["timeline"])
        capture = self.rows("capture", "collector")[0]
        self.assertNotIn("red", [s["state"] for s in capture["timeline"]])
        label = self.rows("label", "labeler")[0]
        self.assertIn(dict(start_frame=25,end_frame=26,state="green"), label["timeline"])
        self.assertEqual(label["timeline"][0],dict(start_frame=0,end_frame=19,state="gray"))

    def test_reconfirm_replaces_decision_and_survives_restart(self):
        self.record([25])
        self.record([25])
        self.record([25], "corrected")
        self.assertEqual(self.account("reviewer")["qc_false_positive_seconds"], 0)
        self.assertEqual(self.account("labeler")["label_seconds"], .1)
        restarted = WorkflowStore(self.store.db_path)
        self.assertEqual(restarted.label_frame_decisions("label")[0]["decision"], "corrected")

    def test_pending_frame_is_not_counted(self):
        self.record([25])
        self.record([25], "pending")
        self.assertEqual(self.account("labeler")["label_seconds"], 0)
        self.assertEqual(self.account("reviewer")["qc_false_positive_seconds"], 0)

    def test_invalid_frames_owner_and_expired_lease(self):
        for frames in ([True], [-1], [100], [19], [1.5], [], "25"):
            with self.subTest(frames=frames), self.assertRaises(WorkflowError):
                self.record(frames)
        with self.assertRaises(WorkflowError):
            self.record([25], operator="someone_else")
        with self.store.connect() as conn:
            conn.execute("UPDATE jobs SET lease_until='2000-01-01T00:00:00Z' WHERE job_id='label'")
        with self.assertRaises(WorkflowError):
            self.record([25])
        self.assertEqual(self.store.label_frame_decisions("label"), [])

    def test_completed_job_cannot_change_decisions(self):
        self.record([25])
        self.store.complete_job(job_id="label", result={"operator_id":"labeler", "frames_completed":list(range(20,50))})
        with self.assertRaises(WorkflowError):
            self.record([25], "corrected")
        self.assertEqual(self.account("labeler")["label_seconds"],3)
        self.assertEqual(self.account("reviewer")["qc_false_positive_seconds"],.1)

    def test_unknown_duration_is_visible_not_estimated(self):
        self.store.create_or_update_episode(episode_id="unknown", subject_id="S02", task_name="unknown",
            status="captured", frame_count=300, metadata={"collection_operator_id":"collector"})
        self.assertEqual(self.account("collector")["capture_seconds"],10)
        self.assertEqual(self.account("collector")["unknown_duration_count"],1)

    def test_sparse_frame_count_is_not_used_as_episode_length(self):
        with self.store.connect() as conn:
            conn.execute("UPDATE episodes SET frame_count=3 WHERE episode_id='ep'")
        row=self.rows("qc","reviewer")[0]
        self.assertEqual(row["frame_count"],100)
        self.assertIsNone(row["seconds_per_frame"])
        self.assertEqual(self.account("reviewer")["qc_seconds"],10)
        self.assertEqual(self.account("reviewer")["qc_error_seconds"],0)
        self.assertEqual(self.account("reviewer")["unknown_duration_count"],1)

    def test_all_registered_instance_history_is_available_before_start(self):
        tasks=self.root/"tasks.json"
        tasks.write_text(json.dumps({"pick_cup":{"repeat_times":3}}))
        registry=TaskInstanceRegistry(self.root/"state",seed_task_files=[(tasks,None)])
        task=registry.snapshot()["task_files"][0]
        from task_backend.server import state_path_from_instance
        instance=registry.add_instance(task["id"],"previous")
        for index, item in enumerate([task["instances"][0],instance]):
            path=state_path_from_instance(registry.data_root,item)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps({"subjects":{"S":{"reservations":{f"history-{index}":{
                "status":"confirmed","operator_id":"historical","task_name":"pick_cup","duration_seconds":8}}}}}))
        runtime=BackendRuntime(registry,self.service)
        rows=runtime.personnel_reservations()
        self.assertEqual(len(rows),2)
        data=snapshot(self.store,[],rows)
        self.assertEqual(next(a for a in data["accounts"] if a["username"]=="historical")["capture_seconds"],16)

    def test_legacy_reservation_duration_backfill_and_zero_activity(self):
        reservations=[dict(reservation_id="legacy", status="confirmed", subject_id="S02", task_name="legacy",
            operator_id="old", duration_seconds=15, frame_count=450)]
        data=snapshot(self.store,["never_worked"],reservations)
        self.assertEqual(next(a for a in data["accounts"] if a["username"]=="old")["capture_seconds"],15)
        self.assertEqual(next(a for a in data["accounts"] if a["username"]=="never_worked")["qc_count"],0)

    def test_full_episode_failure_and_repeated_qc_union(self):
        self.store.create_job(job_id="qc2",job_type="qc",episode_id="ep",payload={})
        self.store.complete_job(job_id="qc2",result={"operator_id":"reviewer","bad_episode":True})
        self.assertEqual(self.account("reviewer")["qc_seconds"],10)
        self.assertEqual(self.account("reviewer")["qc_error_seconds"],10)
        self.assertEqual(self.account("reviewer")["qc_count"],2)

    def test_two_labelers_keep_frame_attribution(self):
        self.record([25])
        self.store.release_job(job_id="label",reason="handoff")
        self.store.lease_job(job_type="manual_label",job_id="label",lease_owner="second")
        self.record([26],"corrected",operator="second")
        self.assertEqual(self.account("labeler")["label_seconds"],.1)
        self.assertEqual(self.account("second")["label_seconds"],.1)

    def test_task_and_episode_pagination_and_filters(self):
        for i in range(53):
            self.store.create_or_update_episode(episode_id=f"e{i}",subject_id="S",task_name="many",
                episode_index=i,status="captured",frame_count=10,metadata={"collection_operator_id":"collector","fps":10})
        data=self.data()
        args={"username":["collector"],"kind":["capture"],"task":["many"],"page":["2"],"page_size":["20"]}
        result=query_records(data,args)
        self.assertEqual(result["total"],53)
        self.assertEqual(len(result["episodes"]),20)
        self.assertEqual(result["episodes"][0]["episode_index"],20)
        filtered=query_records(data,{"username":["collector"],"kind":["capture"],"only_issues":["1"]})
        self.assertEqual(filtered["total"],1)

    def test_http_stats_and_frame_decisions_before_setup(self):
        runtime=BackendRuntime(TaskInstanceRegistry(self.root/"state",seed_task_files=[]),self.service)
        runtime.accounts.register({"username":"registered", "password":"secret", "password_repeat":"secret"})
        server=TaskHTTPServer(("127.0.0.1",0),RequestHandler,runtime)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            base=f"http://127.0.0.1:{server.server_port}"
            with urlopen(base+"/people") as response:
                self.assertIn("人员工作统计",response.read().decode())
            with urlopen(base+"/episodes/ep") as response:
                html=response.read().decode()
                self.assertIn("pick_cup",html)
                self.assertNotIn("Delete Episode From Backend",html)
            with urlopen(base+"/api/v1/personnel") as response:
                text=response.read().decode();data=json.loads(text)
                self.assertIn("registered",[a["username"] for a in data["accounts"]])
                self.assertNotIn("password",text);self.assertNotIn("secret",text)
            request=Request(base+"/api/v1/label/episodes/ep/frames",data=json.dumps(dict(operator_id="labeler",frames=[25],decision="no_error")).encode(),headers={"Content-Type":"application/json"})
            with urlopen(request) as response:
                self.assertTrue(json.load(response)["saved"])
            with self.assertRaises(HTTPError) as error:
                urlopen(base+"/api/v1/personnel/records?page=bad")
            self.assertEqual(error.exception.code,400)
        finally:
            server.shutdown();server.server_close();thread.join(5)


class IntervalTest(unittest.TestCase):
    def test_closed_intervals_and_large_extents_without_frame_expansion(self):
        self.assertEqual(union([(0,0),(1,4),(3,8),(12,15)],10),[(0,8)])
        self.assertEqual(intersection([(0,4),(8,12)],[(3,9)]),[(3,4),(8,9)])
        self.assertEqual(timeline(100000000,[("green",[(0,99999999)]),("red",[(25,25)])]),[
            dict(start_frame=0,end_frame=24,state="green"),dict(start_frame=25,end_frame=25,state="red"),
            dict(start_frame=26,end_frame=99999999,state="green")])


if __name__=="__main__":
    unittest.main()
