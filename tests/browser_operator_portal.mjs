import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
import {mkdir,writeFile} from 'node:fs/promises';
const require=createRequire(process.env.PLAYWRIGHT_PACKAGE || '/Users/cactusxiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright/package.json');
const {chromium}=require('playwright');
const base=process.env.ORBBEC_OPERATOR_TEST_URL;
assert(base,'Set ORBBEC_OPERATOR_TEST_URL to an isolated backend seeded with 整理杯子 (2), 摆放积木 (1), 叠放毛巾 (1), in that order.');
const output=process.env.TEST_OUTPUT||'/tmp/orbbec-operator-ui';await mkdir(output,{recursive:true});
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
const alice=await browser.newContext({viewport:{width:1440,height:1000}}),bob=await browser.newContext();
const page=await alice.newPage(),other=await bob.newPage(),errors=[],checks=[];
for(const p of [page,other])p.on('pageerror',e=>errors.push(e.message));
const check=name=>{checks.push(name);console.log('PASS:',name)};
async function post(path,data){const r=await page.request.post(base+path,{data});assert(r.ok(),await r.text());return r.json()}
async function login(p,name,password='portal-test-password'){await p.goto(base+'/operator');await p.locator('#name').fill(name);await p.locator('#password').fill(password);await p.locator('#loginButton').click()}
try{
 await post('/api/v1/auth/register',{username:'portal.alice',password:'portal-test-password',password_repeat:'portal-test-password'});
 await post('/api/v1/auth/register',{username:'portal.bob',password:'portal-test-password',password_repeat:'portal-test-password'});
 const registry=await page.request.post(base+'/setup/start',{form:{selection:'tasks::default'}});assert(registry.ok());
 await page.goto(base+'/operator');await page.screenshot({path:output+'/operator-login.png',fullPage:true});
 assert.equal((await page.request.get(base+'/api/v1/operator/task')).status(),401);
 await login(page,'portal.alice','wrong');await page.getByText('账号或密码不正确，请重新输入。').waitFor();check('wrong password rejected');
 await login(page,'portal.alice');await page.waitForFunction(()=>document.querySelector('#taskName').textContent==='整理杯子');
 assert.equal(await page.locator('#progress').textContent(),'0 / 2');assert.match(await page.locator('#description').textContent(),/拿起杯子/);
 assert.equal(await page.locator('select').count(),0);check('login automatically shows only the assigned task');
 await page.locator('#video').evaluate(v=>v.play());await page.waitForFunction(()=>document.querySelector('#video').currentTime>0);
 await page.locator('#video').evaluate(v=>{v.pause();v.currentTime=.5});await page.locator('#refresh').click();await page.waitForFunction(()=>!document.querySelector('#refresh').disabled);
 assert.equal(await page.locator('#video').evaluate(v=>v.currentTime),.5);check('demo plays; background refresh preserves playback position');
 await page.screenshot({path:output+'/operator-current-task.png',fullPage:true});
 await page.reload();await page.waitForFunction(()=>document.querySelector('#taskName').textContent==='整理杯子');check('session and assignment survive page reload');
 await page.setViewportSize({width:390,height:844});assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));await page.screenshot({path:output+'/operator-mobile.png',fullPage:true});check('mobile layout fits 390px width');
 await login(other,'portal.bob');await other.waitForFunction(()=>document.querySelector('#taskName').textContent==='摆放积木');check('second operator receives a different entire task');
 const assignment=await post('/api/v1/collection/assignment',{subject_id:'portal.alice'});assert.equal(assignment.task.task_name,'整理杯子');
 const rejected=await page.request.post(base+'/api/v1/episodes/reserve',{data:{subject_id:'portal.alice',client_id:'browser-test',task_name:'摆放积木'}});assert.equal(rejected.status(),409);check('capture API rejects arbitrary task selection');
 for(let i=1;i<=2;i++){
  const reservation=await post('/api/v1/episodes/reserve',{subject_id:'portal.alice',client_id:'browser-test',task_name:'整理杯子'});
  await post('/api/v1/episodes/confirm',{...reservation,subject_id:'portal.alice',idempotency_key:reservation.reservation_id,frame_count:20,duration_seconds:2});
  if(i===1){await page.waitForFunction(()=>document.querySelector('#progress').textContent==='1 / 2');assert.equal(await page.locator('#taskName').textContent(),'整理杯子');check('first episode keeps same task');}
 }
 await page.waitForFunction(()=>document.querySelector('#taskName').textContent==='叠放毛巾');assert.equal(await page.locator('#noVideo').isVisible(),true);assert.equal(await page.locator('#video').isVisible(),false);check('all episodes complete: next task and description appear automatically');
 assert.equal(await other.locator('#taskName').textContent(),'摆放积木');
 const last=await post('/api/v1/episodes/reserve',{subject_id:'portal.alice',client_id:'browser-test',task_name:'叠放毛巾'});await post('/api/v1/episodes/confirm',{...last,subject_id:'portal.alice',idempotency_key:last.reservation_id});
 await page.locator('#waiting').waitFor();check('no available tasks shows waiting state without stealing another operator task');
 await page.locator('#logout').click();await page.locator('#login').waitFor();assert.equal((await page.request.get(base+'/api/v1/operator/task')).status(),401);check('logout revokes access');
 assert.deepEqual(errors,[]);check('no browser JavaScript errors');
}finally{await writeFile(output+'/browser-results.json',JSON.stringify({checks,errors},null,2));await browser.close()}
