"""Compare browser transitions with the actual desktop methods, not duplicates."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from types import SimpleNamespace
from src.qc.app import QcPage
from src.qc.state_store import normalize_ranges

ROOT=Path(__file__).resolve().parents[1]
NODE=shutil.which('node') or '/Users/cactusxiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
class WorkflowParityTest(unittest.TestCase):
    def js(self, code, value):
        run=subprocess.run([NODE,'--input-type=module','-e',code],input=json.dumps(value),text=True,capture_output=True,cwd=ROOT,check=True)
        return json.loads(run.stdout)
    def test_qc_bad_range_transitions_match_desktop(self):
        progress=SimpleNamespace(frames=list(range(180)),current_frame=50,bad_frame_ranges=[],ego_bad_frame_ranges=[],bad_frame_segments=[],ego_bad_frame_segments=[])
        qc=object.__new__(QcPage);qc.progress=progress;qc.mode='playback';qc._playing=False
        qc.app=SimpleNamespace(config=SimpleNamespace(range_merge_gap_frames=5))
        qc._refresh=lambda:None;qc._persist_and_refresh=lambda:None
        actions=[['enter'],['step',-10],['start'],['step',30],['end'],['confirm','egopose'],
                 ['enter'],['step',-100],['start'],['step',12],['end'],['confirm','hand_pose'],
                 ['enter'],['step',30],['cancel'],['step',1],['enter'],['start'],['end'],['confirm','hand_pose']]
        expected=[]
        for op,*args in actions:
            if op=='enter':
                qc.enter_bad_range()
                qc.primary_camera='00'
            elif op=='step':qc.move_bad_cursor(*args) if qc.mode=='bad_range' else qc.step_frames(*args)
            elif op=='start':qc.set_bad_start()
            elif op=='end':qc.set_bad_end()
            elif op=='confirm':qc.confirm_bad_range(*args)
            elif op=='cancel':qc.cancel_bad_range()
            expected.append(dict(mode=qc.mode,frame=qc._display_frame(),start=qc.bad_start,end=qc.bad_end,
                hand=[list(p)for p in progress.bad_frame_ranges],ego=[list(p)for p in progress.ego_bad_frame_ranges]))
        actual=self.js('''import {QcWorkflow} from './remote_frontend/web/workflow.js';
            let input='';for await(const c of process.stdin)input+=c;
            const result={bad_ranges:[],ego_ranges:[]},q=new QcWorkflow(Array.from({length:180},(_,i)=>i),result,50),out=[];
            for(const [op,...args]of JSON.parse(input)){
              if(op==='enter'){q.enterBadRange();q.primaryCamera='00';}else if(op==='step')q.step(...args);else if(op==='start'||op==='end')q.boundary(op);else if(op==='confirm')q.confirmBadRange(...args);else q.cancelBadRange();
              out.push(structuredClone({mode:q.mode,frame:q.frames[q.position],start:q.start,end:q.end,hand:result.bad_ranges,ego:result.ego_ranges}));
            }console.log(JSON.stringify(out));''',actions)
        self.assertEqual(actual,expected)
    def test_gap_merge_matches_native_including_boundary(self):
        cases=[[[1,3],[8,10]],[[1,3],[9,10]],[[10,8],[1,3]],[[0,0],[1,1]],[[2,8],[4,6]],[]]
        expected=[[list(pair)for pair in normalize_ranges(case,max_gap_frames=5)]for case in cases]
        actual=self.js("import {normalizeRanges} from './remote_frontend/web/workflow.js';let s='';for await(const c of process.stdin)s+=c;console.log(JSON.stringify(JSON.parse(s).map(r=>normalizeRanges(r,5))));",cases)
        self.assertEqual(actual,expected)
    def test_playback_completion_does_not_become_per_frame_checklist(self):
        actual=self.js('''import {QcWorkflow} from './remote_frontend/web/workflow.js';const r={},q=new QcWorkflow([0,1,2],r);q.seek(2);const seek=!!r.playback_complete;q.step(-1);q.play();q.displayed(2);q.seek(0);const back=!!r.playback_complete;q.seek(2);q.play();console.log(JSON.stringify({seek,back,replay:q.position}));''',None)
        self.assertEqual(actual,dict(seek=False,back=True,replay=0))

    def test_canvas_mouse_double_click_keeps_visibility_and_toggles_tracking(self):
        actual=self.js("""import {LabelCanvas} from './remote_frontend/web/label-canvas.js';
          globalThis.ResizeObserver=class{observe(){}};
          const listeners={},canvas={addEventListener:(k,f)=>listeners[k]=f,focus(){},setPointerCapture(){},getBoundingClientRect:()=>({left:0,top:0})};
          let widget,tracks=[];
          widget=new LabelCanvas(canvas,{track:(h,j)=>{tracks.push([h,j]);widget.tracked=tracks;},cancelled:()=>{tracks=[];widget.tracked=tracks;}});
          widget.sample={points:[[[10,10]]],visible:[[true]]};widget.readOnly=false;
          widget.hitSchematic=()=>({h:0,j:0,count:false});
          const event={button:0,clientX:0,clientY:0,detail:1};
          const results=[];
          for(let i=0;i<2;i++){
            listeners.pointerdown({...event,detail:0,pointerId:1});listeners.mousedown(event);
            listeners.pointerdown({...event,detail:0,pointerId:1});listeners.mousedown({...event,detail:2});
            widget.double(event);results.push({visible:widget.sample.visible[0][0],tracks:tracks.length,history:widget.history.length});
          }
          console.log(JSON.stringify(results));""",None)
        self.assertEqual(actual,[dict(visible=True,tracks=1,history=0),dict(visible=True,tracks=0,history=0)])

if __name__=='__main__':unittest.main()
