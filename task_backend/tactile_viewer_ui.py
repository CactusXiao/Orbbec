"""Self-contained tactile viewer assets, embedded in the existing viewer page."""

TACTILE_CSS = r"""
#stage.pico-tactile{display:grid;grid-template-columns:minmax(0,1fr) minmax(330px,36%);gap:14px}
#tactilePanel{display:none;min-width:0;min-height:0;overflow:auto;background:#101a22;border:1px solid #263541;border-radius:12px;padding:18px}
#stage.pico-tactile #tactilePanel{display:block}
.touch-heading{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:8px}.touch-heading strong{font-size:15px}
.touch-toggle{display:flex;padding:3px;background:#080f15;border-radius:8px}.touch-toggle button{border:0;background:transparent;color:#93a8b7;border-radius:5px;padding:6px 9px;cursor:pointer}.touch-toggle button.active{background:#22574c;color:#e0fff5}
.touch-hands-toggle{display:flex;gap:6px;margin-top:12px}.touch-hands-toggle button{flex:1;padding:8px;border:1px solid #2a414d;border-radius:7px;color:#8faab9;background:#0b151d;cursor:pointer}.touch-hands-toggle button.active{color:#c2f5e4;background:#173a34;border-color:#326453}
.touch-caption,.touch-note{color:#93a8b7;font-size:11px;line-height:1.6}.touch-note{margin-top:12px}
.touch-hand{padding:13px 0;border-bottom:1px solid #263541}.touch-hand:last-child{border-bottom:0}.touch-hand-head{display:flex;justify-content:space-between;align-items:center;gap:8px}.touch-hand-head strong{font-size:13px}.touch-sync{font-size:11px;color:#83d6bd}.touch-sync.warning{color:#ffbc74}
.touch-stat{margin-top:8px;display:flex;align-items:baseline;gap:8px}.touch-stat b{font-size:24px;font-weight:600;font-variant-numeric:tabular-nums}.touch-stat span{font-size:11px;color:#93a8b7}
.touch-map{display:block;width:100%;height:auto;max-height:290px}.touch-map text{fill:#9db0be;font:13px sans-serif}.touch-cell{stroke:#364b59;stroke-width:1;cursor:crosshair}.touch-cell.calibrated-region{stroke:#62d9b1;stroke-width:2.5}.touch-cell:hover,.touch-cell:focus{stroke:#fff;stroke-width:2.5;outline:none}
.touch-detail{min-height:18px;font-size:11px;color:#b4c6d1}.touch-scale{display:flex;align-items:center;gap:8px;font-size:11px;color:#b4c6d1;margin-top:5px}.touch-gradient{height:6px;flex:1;border-radius:3px;background:linear-gradient(90deg,#153849,#28b69d,#f4bc55,#ef6850)}
.touch-empty{padding:20px 8px;color:#93a8b7;font-size:12px;text-align:center}.touch-invalid{color:#ffbc74;font-size:11px;margin-top:5px;line-height:1.5}
.touch-bends{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:5px;margin:8px 0}.touch-bend{background:#14232d;border-radius:5px;padding:5px;text-align:center;font-size:11px;color:#b4c6d1}.touch-bend meter{display:block;width:100%;height:6px;margin-top:4px}
@media(max-width:1050px){#stage.pico-tactile{grid-template-columns:minmax(0,1fr);grid-template-rows:minmax(150px,1fr) minmax(180px,42%)}#tactilePanel{padding:12px}#touchHands{display:block}.touch-hand{border-bottom:0}.touch-map{max-height:220px}}
"""

TACTILE_HTML = r"""
<section id="tactilePanel" aria-label="同步触觉可视化">
  <div class="touch-heading"><strong>触觉 · Tactile</strong><div class="touch-toggle" aria-label="触觉显示单位"><button data-touch-unit="N" aria-pressed="false">区域力 N</button><button class="active" data-touch-unit="ADC" aria-pressed="true">压力 ADC</button></div></div>
  <div class="touch-caption">与 Pico 画面同步 · 色阶在整段采集中保持固定</div>
  <div id="touchHandTabs" class="touch-hands-toggle" aria-label="选择手套"></div>
  <div id="touchHands"></div>
  <div class="touch-note">布局依据触觉手套说明书第 11–13 页：每手 132 个压力点、5 个弯曲通道。N 仅为右手中指标定区域总力；没有逐点力标定。弯曲保留 ADC，不换算角度或 N。</div>
</section>
"""

TACTILE_JS = r"""
let touchUnit='ADC',touchData=null,touchSignature='',touchSelected='';
const touchCards=new Map();
function touchElement(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text)e.textContent=text;return e}
function touchSvg(tag,attrs){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);return e}
function buildTouchCard(hand){
 const card=touchElement('div','touch-hand'),head=touchElement('div','touch-hand-head'),label=hand.side==='right'?'右手 · 掌面':hand.side==='left'?'左手 · 掌面':hand.id+' · 通道矩阵';
 head.append(touchElement('strong','',label));const sync=touchElement('span','touch-sync');head.append(sync);card.append(head);
 const stat=touchElement('div','touch-stat'),total=touchElement('b'),statLabel=touchElement('span');stat.append(total,statLabel);card.append(stat);
 const warning=touchElement('div','touch-invalid');card.append(warning);
 const svg=touchSvg('svg',{viewBox:'0 0 410 345',class:'touch-map',role:'group','aria-label':label+'触觉分布'}),cells=[];
 function group(ids,x,y,columns,label){ids.forEach((id,index)=>{const cell=touchSvg('rect',{x:x+(index%columns)*18,y:y+Math.floor(index/columns)*18,width:15,height:15,rx:4,class:'touch-cell',tabindex:0});const title=touchSvg('title',{});cell.append(title);svg.append(cell);cells.push({cell,title,index:id-1,id});const select=()=>{detail.textContent=cell.getAttribute('aria-label')};cell.onmouseenter=select;cell.onfocus=select});if(label){const text=touchSvg('text',{x:x+columns*9,y:y-10,'text-anchor':'middle'});text.textContent=label;svg.append(text)}}
 const layout=hand.layout;
 if(layout){
  const positions=[[24,153],[104,65],[174,30],[244,60],[314,104]];
  layout.fingers.forEach((f,i)=>{const [x,y]=positions[i];group(f.ids,hand.side==='left'?356-x:x,y,3,f.label)});
  layout.palm_rows.forEach((row,i)=>group(row,70+(i===0?layout.palm_first_offset*18:0),235+i*18,row.length,i===0?'手掌':''));
 }else{group(Array.from({length:256},(_,i)=>i+1),62,32,16,'原始通道 #1–256')}
 card.append(svg);const empty=touchElement('div','touch-empty');card.append(empty);const detail=touchElement('div','touch-detail','悬停或聚焦测点查看数值');card.append(detail);
 const scale=touchElement('div','touch-scale'),maxLabel=touchElement('span');scale.append(touchElement('span','','0'),touchElement('div','touch-gradient'),maxLabel);card.append(scale);
 const mapNote=touchElement('div','touch-caption');card.append(mapNote);
 const bendTitle=touchElement('div','touch-caption','五指弯曲 · 原始 ADC'),bends=touchElement('div','touch-bends'),bendCells=[];
 if(layout){card.append(bendTitle,bends);layout.fingers.forEach(f=>{const box=touchElement('div','touch-bend'),value=touchElement('span'),meter=document.createElement('meter');meter.min=0;meter.max=255;meter.setAttribute('aria-label',f.label+'弯曲 ADC');box.append(touchElement('div','',f.label),value,meter);box.title=f.label+'弯曲 · 传感器 #'+f.bend_id;bends.append(box);bendCells.push({id:f.bend_id,value,meter})})}
 $('#touchHands').append(card);return {card,sync,total,statLabel,warning,svg,cells,detail,empty,maxLabel,scale,mapNote,bendCells};
}
function touchColor(value,max){if(!Number.isFinite(value))return '#202d37';const stops=[[21,56,73],[40,182,157],[244,188,85],[239,104,80]],t=Math.min(1,Math.max(0,value/Math.max(max,1e-9)))*3,i=Math.min(2,Math.floor(t)),f=t-i;return `rgb(${stops[i].map((v,j)=>Math.round(v+(stops[i+1][j]-v)*f)).join(',')})`}
function renderTactile(data){
 touchData=data;const hands=data&&data.hands||[],signature=JSON.stringify(hands.map(h=>[h.id,h.side,h.layout]));
 if(signature!==touchSignature){touchSignature=signature;touchCards.clear();$('#touchHands').replaceChildren();$('#touchHandTabs').replaceChildren();if(!hands.some(h=>h.id===touchSelected))touchSelected=(hands.find(h=>h.side==='right')||hands[0]||{}).id;hands.forEach(h=>{touchCards.set(h.id,buildTouchCard(h));const b=touchElement('button','',h.side==='right'?'右手':h.side==='left'?'左手':h.id);b.dataset.touchHand=h.id;b.onclick=()=>{touchSelected=h.id;renderTactile(touchData)};$('#touchHandTabs').append(b)})}
 document.querySelectorAll('[data-touch-hand]').forEach(b=>{const active=b.dataset.touchHand===touchSelected;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active))});
 if(!hands.length){$('#touchHands').textContent=data&&data.error||'此 Episode 没有触觉数据';touchSignature='';return}
 for(const hand of hands){
 const c=touchCards.get(hand.id),sample=hand.sample,ready=hand.status==='ready'&&sample,force=touchUnit==='N';
 const values=ready?sample.adc:[],region=new Set(hand.calibrated_sensor_ids||[]);
 c.card.style.display=hand.id===touchSelected?'block':'none';
 c.sync.textContent=ready?(hand.delta_ms===null?'已按采集索引对齐':`同步差 ${hand.delta_ms>=0?'+':''}${hand.delta_ms.toFixed(1)} ms`):hand.message;c.sync.classList.toggle('warning',!ready||sample.out_of_range);
 c.total.textContent=ready&&Number.isFinite(sample.total_n)?sample.total_n.toFixed(2)+' N':'—';
 c.statLabel.textContent=hand.side==='right'?'右手中指区域总力':'该手尚无力值标定';
 const warnings=[];
 if(ready){
  if(hand.side!=='right')warnings.push('左手尚无独立力值标定，压力与弯曲均保留 ADC');
  else if(!hand.has_force)warnings.push('此段数据仅保存 ADC，尚无区域力 N');
  if(sample.out_of_range)warnings.push('超出标定范围；公式无有效结果时不显示力值');
  if(sample.quality!=='ok')warnings.push('采样质量：'+sample.quality);
 }
 c.warning.textContent=warnings.join('；');
 c.svg.style.display=ready?'block':'none';c.empty.style.display=ready?'none':'block';c.empty.textContent=hand.message;
 c.scale.style.display=ready&&!force?'flex':'none';c.maxLabel.textContent='255 ADC';c.detail.textContent=ready?'悬停或聚焦测点查看数值':'';
 c.mapNote.textContent=force?'绿色边框仅标出总力对应的区域；灰色测点没有逐点 N 值。':'颜色表示测点原始 ADC，色阶在整段采集中保持固定；不包含弯曲通道。';
 for(const {cell,title,index,id} of c.cells){
  const value=values[index],inRegion=region.has(id);
  const text=force?`传感器 #${id} · ${inRegion?'右手中指标定区域；总力见上方，非逐点力':'未标定逐点力'}`:`传感器 #${id} · ${Number.isFinite(value)?value.toFixed(0)+' ADC':'无数据'}`;
  cell.classList.toggle('calibrated-region',force&&inRegion);
  cell.setAttribute('fill',force?'#202d37':touchColor(value,255));cell.setAttribute('aria-label',text);title.textContent=text;
 }
 for(const bend of c.bendCells){const value=values[bend.id-1];bend.value.textContent=Number.isFinite(value)?value.toFixed(0):'—';bend.meter.value=Number.isFinite(value)?value:0;bend.meter.style.visibility=Number.isFinite(value)?'visible':'hidden'}

 }
}
for(const b of document.querySelectorAll('[data-touch-unit]'))b.onclick=()=>{touchUnit=b.dataset.touchUnit;document.querySelectorAll('[data-touch-unit]').forEach(e=>{const active=e.dataset.touchUnit===touchUnit;e.classList.toggle('active',active);e.setAttribute('aria-pressed',String(active))});renderTactile(touchData)};
function cachedTactile(url){let promise=frameCache.get(url);if(!promise){promise=api(url).catch(error=>{frameCache.delete(url);return {hands:[],error:'触觉数据加载失败：'+error.message}});frameCache.set(url,promise)}return promise}
function cachedFrameAsset(url){return url.includes('/media/pico/tactile/')?cachedTactile(url):mode==='pointcloud'?cachedCloud(url):cachedImage(url)}
"""
