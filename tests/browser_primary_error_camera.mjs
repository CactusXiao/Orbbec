import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createServer} from 'node:http';
import {createRequire} from 'node:module';
const require = createRequire(process.env.PLAYWRIGHT_PACKAGE || '/Users/cactusxiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/package.json');
const {chromium} = require('playwright');
const root = new URL('../remote_frontend/web/', import.meta.url);
// Test-only hooks are appended to an in-memory copy of the actual app module.
const hooks = `
save = async () => {};
imageURL = async () => 'data:image/svg+xml,' + encodeURIComponent('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480"><rect width="640" height="480" fill="#344b61"/></svg>');
frameCache.prefetch = () => {};
window.cameraTest = {
  async setup(role, segments = []) {
    owner = 'test'; busy = false; durable = true; highQuality = true;
    position = 0; camera = '00'; overview = false; overlay = null;
    draft = {id: role, manifest: {role, cameras:['00','02','03'], frames:[0,1,2,10,11,20,21,30,31],
      qc_segments:segments, task_name:'Test', episode_index:1, skeleton_edges:[]},
      result:{samples:{}, confirmed:[], reviewed:[], bad_ranges:[], ego_ranges:[]}};
    for (const f of draft.manifest.frames) for (const c of draft.manifest.cameras)
      draft.result.samples[f+':'+c] = {points:Array.from({length:2},()=>Array.from({length:21},()=>[30,30])),visible:Array.from({length:2},()=>Array(21).fill(true))};
    qcFlow = role === 'qc' ? new QcWorkflow(draft.manifest.frames, draft.result) : null;
    for (const id of ['loginPanel','picker','preparing']) $(id).hidden = true;
    for (const id of ['workspace','content']) $(id).hidden = false;
    $('labelPanel').hidden = role !== 'label'; $('qcPanel').hidden = role !== 'qc';
    $('labelToolbar').hidden = role !== 'label'; $('qcControlBar').hidden = role !== 'qc';
    $('editorLayout').hidden = false; $('overviewGrid').hidden = true;
    await renderFrame();
  },
  async jump(p) { position=p; await renderFrame(); },
  notify(text) { notice(text); },
  state() {return {camera,overview,mode:qcFlow?.mode,selected:qcFlow?.primaryCamera,result:draft.result,visited:draft.visitedSegments};}
};`;
const server = createServer(async (req,res) => {
  try {
    const name = req.url === '/' ? 'index.html' : req.url.slice(1).split('?')[0];
    if (name.includes('..') || name === 'sw.js') {res.writeHead(404);res.end();return;}
    let content = await readFile(new URL(name,root),'utf8');
    if (name === 'app.js') content = content.replace('await auth.start();','') + hooks;
    res.setHeader('Content-Type', name.endsWith('.js') ? 'text/javascript' : name.endsWith('.css') ? 'text/css' : 'text/html');
    res.end(content);
  } catch {res.writeHead(404);res.end();}
});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const browser = await chromium.launch({executablePath: process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
try {
  const page = await browser.newPage({viewport:{width:1440,height:1000}}), errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(()=>window.cameraTest);
  await page.evaluate(()=>cameraTest.setup('qc'));
  await page.locator('#enterRange').click();
  await page.locator('#setStart').click();
  await page.locator('#setEnd').click();
  await page.locator('#badRange').click();
  assert.equal((await page.evaluate(()=>cameraTest.state())).mode,'bad_range');
  assert.match(await page.locator('#notice').textContent(), /请双击/);
  await page.locator('#nativeQcGrid canvas[aria-label="Camera 02"]').dblclick();
  assert.equal(await page.locator('#notice').textContent(),'');
  assert.equal(await page.locator('.primaryErrorCamera').count(),1);
  assert.equal(await page.locator('.primaryErrorCamera').evaluate(e=>getComputedStyle(e).outlineColor),'rgb(229, 72, 77)');
  await page.locator('#nativeQcGrid canvas[aria-label="Camera 03"]').dblclick();
  assert.equal(await page.locator('.primaryErrorCamera').count(),1);
  assert.equal((await page.evaluate(()=>cameraTest.state())).selected,'03');
  await page.locator('#badRange').click();
  await page.waitForFunction(()=>cameraTest.state().mode==='playback');
  assert.equal((await page.evaluate(()=>cameraTest.state())).result.bad_segments[0].primary_camera,'03');
  assert.equal(await page.locator('.primaryErrorCamera').count(),0);
  await page.evaluate(()=>cameraTest.setup('label',[
    {segment_id:'a',start_frame:0,end_frame:2,primary_camera:'02'},
    {segment_id:'b',start_frame:10,end_frame:11,primary_camera:'03'},
    {segment_id:'c',start_frame:20,end_frame:21,primary_camera:'ego'}]));
  assert.equal((await page.evaluate(()=>cameraTest.state())).camera,'02');
  await page.keyboard.press('1');
  await page.waitForFunction(()=>cameraTest.state().camera==='00');
  await page.evaluate(()=>cameraTest.jump(1));
  assert.equal((await page.evaluate(()=>cameraTest.state())).camera,'00');
  await page.evaluate(()=>cameraTest.jump(3));
  assert.equal((await page.evaluate(()=>cameraTest.state())).camera,'03');
  await page.keyboard.press('1');
  await page.evaluate(()=>cameraTest.jump(0));
  assert.equal((await page.evaluate(()=>cameraTest.state())).camera,'00');
  await page.evaluate(()=>cameraTest.jump(5));
  assert.equal((await page.evaluate(()=>cameraTest.state())).camera,'ego');
  assert.equal(await page.locator('#overviewGrid .nativeCameraCell:visible').count(),1);
  assert.equal(await page.locator('#editorLayout').isVisible(),false);
  await page.locator('#overview').click();
  await page.waitForFunction(()=>cameraTest.state().camera==='00');
  await page.evaluate(()=>cameraTest.jump(6));
  assert.equal((await page.evaluate(()=>cameraTest.state())).camera,'00');
  await page.evaluate(()=>cameraTest.notify('计算完成，结果尚未提交。'));
  await page.keyboard.press('1');
  assert.equal(await page.locator('#notice').textContent(),'');
  await page.clock.install();
  await page.clock.pauseAt(new Date());
  await page.evaluate(()=>cameraTest.notify('计算完成，结果尚未提交。'));
  await page.clock.fastForward(2000);
  assert.equal(await page.locator('#notice').isVisible(),true);
  await page.evaluate(()=>cameraTest.notify('新的操作提醒'));
  await page.clock.fastForward(1000);
  assert.equal(await page.locator('#notice').textContent(),'新的操作提醒');
  await page.clock.fastForward(2000);
  assert.equal(await page.locator('#notice').textContent(),'');
  assert.equal(await page.locator('#notice').isVisible(),false);
  assert.deepEqual(errors,[]);
  console.log('Browser UI passed: QC/Label camera handoff, reminder expiry, replacement timer and clearing on next action.');
} finally {await browser.close();await new Promise(r=>server.close(r));}
