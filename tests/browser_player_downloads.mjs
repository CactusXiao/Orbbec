import assert from 'node:assert/strict';
import {ChunkDownloads, EpisodePlayer} from '../remote_frontend/web/player.js';
const flush=async()=>{for(let i=0;i<30;i++)await Promise.resolve();};
const chunks=Array.from({length:4},(_,index)=>({index,start:index*90,count:90}));
// Native browser fetch rejects a receiver other than Window.
const nativeFetch = globalThis.fetch;
try {
  globalThis.fetch = function () {
    assert.equal(this, globalThis, 'Default downloader must preserve native fetch receiver');
    return Promise.resolve({ok:true,arrayBuffer:async()=>new ArrayBuffer(8)});
  };
  const browserDownloads = new ChunkDownloads(i=>String(i));
  browserDownloads.plan(chunks.slice(0,1));
  const result = await browserDownloads.entries.get(0).promise;
  assert.equal(result.error, undefined);
  assert.equal(result.bytes.byteLength,8);
  browserDownloads.close();
} finally { globalThis.fetch = nativeFetch; }
const requests=new Map(), appended=[];
const downloads=new ChunkDownloads(i=>String(i),(url,{signal})=>new Promise((resolve,reject)=>{
  requests.set(+url,{resolve:()=>resolve({ok:true,arrayBuffer:async()=>new Uint8Array([+url])}),signal});
  signal.addEventListener('abort',()=>reject(new DOMException('cancelled','AbortError')));
}));
const player=Object.assign(Object.create(EpisodePlayer.prototype),{
  incremental:true,closed:false,running:false,open:Promise.resolve(),video:{currentTime:0},
  downloads,state:{chunks,complete:false},fps:30,total:360,loaded:new Set(),seekSerial:0,
  buffer:{appendBuffer(bytes){appended.push([bytes[0],this.timestampOffset]);}},
  change:async fn=>fn(),onStatus(){},resume(){},
  ahead(t){const c=chunks.find(c=>t>=c.start/30&&t<(c.start+c.count)/30);return c&&this.loaded.has(c.index)?(c.start+c.count)/30-t:0;}
});
const pumping=player.pump();await flush();
assert.deepEqual([...requests.keys()],[0,1,2], 'Three requests start before any response');
requests.get(2).resolve();requests.get(1).resolve();await flush();
assert.equal(appended.length,0,'Out-of-order downloads must not reorder video frames');
requests.get(0).resolve();await flush();
assert.deepEqual(appended,[[0,0],[1,3],[2,6]]);
assert(requests.has(3));assert(downloads.entries.size<=3);
requests.get(3).resolve();await pumping;
assert.deepEqual(appended,[[0,0],[1,3],[2,6],[3,9]]);
assert.equal(downloads.entries.size,0,'Appended compressed buffers are released');
const seekQueue=new ChunkDownloads(i=>String(i),(url,{signal})=>new Promise((_,reject)=>{
 requests.set(+url,{signal});signal.addEventListener('abort',()=>reject(new DOMException('cancelled','AbortError')));
}));
seekQueue.plan(chunks);const old=[...seekQueue.entries.values()];
seekQueue.plan([{index:20},{index:21},{index:22},{index:23}]);
assert(old.every(e=>e.controller.signal.aborted),'A distant seek cancels old network work');
assert.deepEqual([...seekQueue.entries.keys()],[20,21,22]);
await Promise.all(old.map(e=>e.promise));
const active=[...seekQueue.entries.values()];seekQueue.close();
assert(active.every(e=>e.controller.signal.aborted));
await Promise.all(active.map(e=>e.promise));
console.log('QC parallel downloads, ordered append, seek cancellation and buffer bounds passed');
const {LabelCanvas}=await import('../remote_frontend/web/label-canvas.js');
for(const alpha of [0,0.5,1]) {
 const calls=[],ctx={clearRect(){},fillRect(){},save(){},restore(){},drawImage(...args){calls.push(args);}};
 const canvas=Object.assign(Object.create(LabelCanvas.prototype),{ctx,width:512,height:408,image:{},crop:[0,0,512,408],renderCrop:[0,864,512,408],overlayOpacity:alpha,view:{x:0,y:0,scale:1},imageWidth:512,imageHeight:408});
 canvas.draw();assert.equal(calls.length,alpha===0.5?2:1);
 assert.equal(calls[0][2],alpha===1?864:0);
}
const fs=await import('node:fs'),vm=await import('node:vm');
const source=fs.readFileSync(new URL('../remote_frontend/web/app.js',import.meta.url),'utf8');
let clock=0,updates=0;const displayed=[];
const video={paused:false,seeking:false,requestVideoFrameCallback(){}};
const ctx=vm.createContext({owner:true,draft:{manifest:{frames:Array.from({length:60},(_,i)=>i),fps:30}},qcFlow:{displayed:i=>displayed.push(i)},player:{seeking:false},$:id=>id==='video'?video:{},paintQC(){},updateProgress(){updates++;},position:0,frameReady:false,reviewTimer:1,playbackUiAt:0,performance:{now:()=>clock}});
vm.runInContext(source.slice(source.indexOf('function videoFrame('),source.indexOf('if ("requestVideoFrameCallback"')),ctx);
for(let i=0;i<30;i++){clock=i*1000/30;ctx.videoFrame(null,{mediaTime:i/30});}
assert.deepEqual(displayed,Array.from({length:30},(_,i)=>i));assert(updates<=10);
assert.equal(ctx.position,29);
console.log('QC opacity endpoints and per-frame accounting with throttled UI passed');
