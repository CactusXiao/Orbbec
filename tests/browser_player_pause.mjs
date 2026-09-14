import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import {EpisodePlayer} from '../remote_frontend/web/player.js';
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});return {promise,resolve,reject};};
const flush=async()=>{for(let i=0;i<8;i++) await Promise.resolve();};
function fixture() {
  const video={paused:true,currentTime:0,readyState:4,playCalls:0,pauseCalls:0,
    play(){this.paused=false;this.playCalls++;return Promise.resolve();},pause(){this.paused=true;this.pauseCalls++;}};
  const player=Object.assign(Object.create(EpisodePlayer.prototype),{video,total:900,fps:30,playbackSerial:0,playPending:null,wantsPlay:false,closed:false,seeking:false,onStatus(){},ahead(){return 20;},seek:async()=>true});
  return {player,video};
}
// Real app handler: a pending seek must expose Pause immediately and remain
// cancelled after the original seek finishes (old renderFrame->play order fails).
{
  const {player,video}=fixture(),seek=deferred();player.seek=()=>seek.promise;
  const elements={video,play:{},highQuality:{}},flow={position:0,playing:false,play(){this.playing=true;return true;}};
  const context=vm.createContext({player,qcFlow:flow,owner:'test',busy:false,frameReady:true,position:0,highQuality:false,
    $:id=>elements[id],paintQC(){},renderFrame:()=>seek.promise,notice:()=>assert.fail('Unexpected error'),updateProgress(){elements.play.textContent=player.wantsPlay?'暂停':'播放';}});
  player.onStatus=context.updateProgress;
  const source=fs.readFileSync(new URL('../remote_frontend/web/app.js',import.meta.url),'utf8');
  const begin=source.indexOf('$("play").onclick = async () => {'),end=source.indexOf('function selectBoundary',begin);
  vm.runInContext(source.slice(begin,end),context);
  const start=elements.play.onclick();
  assert.equal(player.wantsPlay,true,'Pending start must be cancellable');
  assert.equal(elements.play.textContent,'暂停');
  await elements.play.onclick();
  assert.equal(player.wantsPlay,false);
  seek.resolve(true);await start;player.resume();await flush();
  assert.equal(video.playCalls,0,'Delayed seek must not restart paused video');
}
// Automatic buffering resume must respect explicit pause even after data arrives.
{
  const {player,video}=fixture();let seconds=0;player.ahead=()=>seconds;
  await player.play(0);assert.equal(video.playCalls,0);
  player.pause();seconds=20;player.resume();await flush();
  assert.equal(video.paused,true);assert.equal(video.playCalls,0);
}
// AbortError from an old native play() must not clear a new play intent.
{
  const {player,video}=fixture(),old=deferred(),next=deferred();
  video.play=function(){this.playCalls++;this.paused=false;return this.playCalls===1?old.promise:next.promise;};
  await player.play(0);player.pause();await player.play(0);
  old.reject(new Error('old AbortError'));await flush();
  assert.equal(player.wantsPlay,true);assert.equal(video.playCalls,2);
  next.resolve();await flush();player.pause();assert.equal(video.paused,true);
}
// A late successful native start must also yield to a user's final pause.
{
  const {player,video}=fixture(),pending=deferred();
  video.play=function(){this.playCalls++;return pending.promise;};
  await player.play(0);player.resume();assert.equal(video.playCalls,1,'Avoid parallel play promises');
  player.pause();video.paused=false;pending.resolve();await flush();
  assert.equal(video.paused,true);assert.equal(player.wantsPlay,false);
}
console.log('4 QC pause regression scenarios passed');
