import assert from 'node:assert/strict';
import {labelLookahead} from '../remote_frontend/web/frame-cache.js';
const frames=Array.from({length:30},(_,i)=>i+500), cameras=['00','02','03','05'];
const plan=labelLookahead(frames,cameras,5,'00',true);
for(const camera of [...cameras,'ego'])
 for(let d=1;d<=4;d++) assert.ok(plan.some(p=>p.camera===camera&&p.frame===frames[5+d]));
assert.equal(plan.length,new Set(plan.map(p=>`${p.camera}:${p.frame}`)).size);
assert.ok(labelLookahead(frames,cameras,29,'00',true).every(p=>frames.includes(p.frame)));
const single=labelLookahead(frames,cameras,5,'02',false);
assert.equal(single[0].camera,'02');
assert.ok(single.some(p=>p.camera==='02'&&p.frame===525));
console.log('Label overview prefetch includes Ego, respects boundaries and preserves selected-view lookahead');
