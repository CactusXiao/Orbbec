import assert from 'node:assert/strict';
import {labelSegments, activeLabelSegment, labelStep, enterLabelSegment} from '../remote_frontend/web/workflow.js';
import {LabelCanvas} from '../remote_frontend/web/label-canvas.js';
const manifest = {frames:[0,1,2,3,4,10,11,12,20], qc_segments:[
  {segment_id:'a',start_frame:0,end_frame:2},
  {segment_id:'b',start_frame:3,end_frame:4},
  {segment_id:'c',start_frame:10,end_frame:12},
  {segment_id:'d',start_frame:11,end_frame:12},
]};
const draft = {manifest,activeSegment:'a'};
assert.equal(labelSegments(manifest).length,5);
assert.equal(labelStep(draft,2,1),2); // Adjacent intervals must not join.
assert.equal(labelStep(draft,0,-1),0);
assert.equal(labelStep(draft,0,10),2);
draft.activeSegment='d';
assert.equal(activeLabelSegment(draft,6).key,'d');
assert.equal(labelStep(draft,6,-1),6); // Preserve chosen overlapping segment.
const overlapDraft = {manifest:{cameras:['00','ego'],frames:[0,1,2,3],qc_segments:[
  {segment_id:'fixed',start_frame:0,end_frame:3,primary_camera:'00'},
  {segment_id:'ego',start_frame:1,end_frame:2,primary_camera:'ego'}]}, activeSegment:'fixed'};
assert.equal(enterLabelSegment(overlapDraft,0,activeLabelSegment(overlapDraft,0)),'00');
assert.equal(enterLabelSegment(overlapDraft,1,activeLabelSegment(overlapDraft,1)),null);
assert.deepEqual(overlapDraft.visitedSegments,['fixed']);
let observer;
globalThis.ResizeObserver=class{constructor(cb){observer=cb;}observe(){}};
globalThis.Image=class{naturalWidth=640;naturalHeight=480;async decode(){}};
globalThis.devicePixelRatio=1;
let size={width:800,height:600};
const canvas={addEventListener(){},getBoundingClientRect:()=>size,getContext:()=>({setTransform(){}})};
const widget=new LabelCanvas(canvas);widget.draw=()=>{};
await widget.setImage('first','episode:0:ego','episode:ego');
widget.view={scale:2.3,x:-130,y:17};
await widget.setImage('next','episode:1:ego','episode:ego');
assert.deepEqual(widget.view,{scale:2.3,x:-130,y:17});
observer();assert.deepEqual(widget.view,{scale:2.3,x:-130,y:17});
await widget.setImage('source','episode:1:ego:mano','episode:ego');
assert.deepEqual(widget.view,{scale:2.3,x:-130,y:17});
await widget.setImage('other','episode:1:00','episode:00');
assert.equal(widget.view.scale,1.25);
console.log('PASS: independent/overlapping intervals; confirmation view persistence; resize; camera reset.');
