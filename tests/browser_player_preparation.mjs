import assert from 'node:assert/strict';
import {EpisodePlayer} from '../remote_frontend/web/player.js';
const make=(prepared,frames,seconds)=>Object.assign(Object.create(EpisodePlayer.prototype),{
 total:3000,fps:30,video:{currentTime:0},state:{complete:false,prepared},
 preparationSamples:[{time:0,frames:prepared-frames},{time:seconds*1000,frames:prepared}]
});
// Ten seconds cached cannot sustain 100 seconds when preparation is half real time.
assert.equal(make(300,150,10).preparationReady(10000),false);
// Enough preparation accumulated to cover the slow producer's remaining work.
assert.equal(make(2400,150,10).preparationReady(10000),true);
// A producer faster than playback can start before rendering the full episode.
assert.equal(make(600,450,10).preparationReady(10000),true);
// A stalled producer must not use its last, now stale, high speed estimate.
assert.equal(make(600,450,10).preparationReady(60000),false);
const done=make(3000,0,1);assert.equal(done.preparationReady(1000),true);
const unknown=make(300,0,0);unknown.preparationSamples=[];
assert.equal(unknown.preparationReady(10000),false);
console.log('Preparation headroom: slow, fast, stopped, complete and unknown scenarios passed');
