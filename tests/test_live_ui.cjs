const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const {test} = require('node:test');
const path = require('node:path');
const root = process.env.FACE_MASK_ROOT || path.resolve(__dirname, '..');
const source = fs.readFileSync(process.env.TEST_WEB_APP || path.join(root, 'web/app.js'), 'utf8').replace(/\ninit\(\);\s*$/, '');
const html = fs.readFileSync(path.join(root, 'web/index.html'), 'utf8');
function setup() {
  const nodes = new Map();
  const make = () => ({textContent:'', hidden:false, disabled:false, className:'', style:{}, children:[],
    attrs:{}, click(){this.clicked=true}, classList:{toggle(){}}, append(...items){this.children.push(...items)},
    replaceChildren(...items){this.children=items}, setAttribute(k,v){this.attrs[k]=String(v)},
    removeAttribute(k){delete this.attrs[k]; if(k==='src')this.src=''}, addEventListener(){},
    querySelector(name){return this[name] || (this[name]=make())}});
  for (const id of html.matchAll(/id="([^"]+)"/g)) nodes.set(id[1],make());
  let pending=[], calls=[],requests=[],timers=[];
  const context = vm.createContext({console, document:{getElementById:id=>nodes.get(id),createElement:make,body:{appendChild(){}}},
    location:{origin:'http://127.0.0.1:8000'}, window:{addEventListener(){}},
    URL:{createObjectURL:()=> 'blob:test-result',revokeObjectURL(){}}, Blob,Uint8Array,atob,
    performance:{now:()=> 120}, AbortSignal, setTimeout:(fn,ms)=>{timers.push({fn,ms});return timers.length},clearTimeout(){},
    setInterval(){}, localStorage:{}, fetch(){throw Error('Unexpected test fetch')}});
  vm.runInContext(source,context);
  context.testRequest = (route,options) => {calls.push(route);requests.push({route,options}); return new Promise((resolve,reject)=>pending.push({resolve,reject}))};
  vm.runInContext('cameraBase="http://192.168.137.28";request=testRequest;',context);
  return {nodes,context,calls,requests,timers,resolve:(response,index=0)=>pending[index].resolve(response),reject:(error,index=0)=>pending[index].reject(error),run:code=>vm.runInContext(code,context)};
}
test('one manual shot clears old image, shows pending state and requests YOLO',()=>{
  const s=setup();s.nodes.get('frame').src='data:old-shot';s.nodes.get('captureButton').onclick();
  assert.equal(s.nodes.get('frame').src,'');assert.equal(s.nodes.get('frame').hidden,true);
  assert.match(s.nodes.get('frameSource').textContent,/ĐANG NHẬN DIỆN/);
  assert.equal(s.nodes.get('peopleCount').textContent,'—');
  assert.deepEqual(s.calls,['/camera/predict']);
  assert.equal(JSON.parse(s.requests[0].options.body).send_telegram,true);
});
test('an old delayed result cannot replace pending live after changing view',async()=>{
  const s=setup();
  const work=s.run('renderResult({request_id:"old",source:"esp32",detections:[],detection_count:0,annotated_image_url:"/old.jpg"})');
  s.nodes.get('liveButton').onclick();s.resolve({blob:async()=>({})});await work;
  assert.equal(s.nodes.get('frame').src,'');assert.equal(s.nodes.get('frame').hidden,true);
  assert.match(s.nodes.get('frameSource').textContent,/ĐANG NHẬN DIỆN/);
  assert.equal(s.nodes.get('peopleCount').textContent,'—');
});
test('Xem ảnh freezes the live frame; polling never fetches or replaces latest image',async()=>{
  const s=setup();
  await s.run('liveDetection=true;renderResult({request_id:"held",timestamp:"2026-10-11T00:00:00Z",source:"camera_pull",detections:[],detection_count:0,annotated_image_base64:"eA=="},true)');
  s.nodes.get('resultButton').onclick();const src=s.nodes.get('frame').src;
  s.context.testJson=async route=>{
    s.calls.push(route);
    if(route==='/health')return {model_loaded:true};
    if(route==='/model/info')return {metadata:{epochs_completed:12},device:'0',checkpoint:'candidate.pt'};
    if(route==='/device/status')return {status:'online'};
    if(route==='/telegram/status')return {state:'disabled',alert_mode:'violations',cooldown_seconds:30};
    if(route.startsWith('/metrics/history'))return {items:[]};
    return {};
  };
  await s.run('json=testJson;renderHistory=()=>{};renderSummary=()=>{};poll();');
  assert.equal(s.nodes.get('frame').src,src);
  assert.equal(s.run('liveDetection'),false);
  assert.equal(s.calls.includes('/results/latest'),false);
  assert.match(s.nodes.get('frameSource').textContent,/ẢNH ĐÃ NHẬN DIỆN/);
});
test('flash click gives immediate feedback, yields captures and waits for confirmation',async()=>{
  const s=setup();const work=s.nodes.get('flashButton').onclick();
  assert.match(s.nodes.get('flashButton').textContent,/Đang bật/);
  assert.equal(s.nodes.get('flashButton').disabled,true);
  s.run('liveDetection=true;liveCameraTick(liveGeneration);');
  assert.deepEqual(s.calls,['/camera/flash']);
  s.resolve({json:async()=>({flash:true,confirmed:true})});await work;
  assert.equal(s.nodes.get('flashButton').textContent,'Tắt flash');
  assert.equal(s.nodes.get('flashButton').attrs['aria-pressed'],'true');
  assert.match(s.nodes.get('flashStatus').textContent,/camera xác nhận/);
});
test('failed flash confirmation never becomes an ON state',async()=>{
  const s=setup();const work=s.nodes.get('flashButton').onclick();
  s.resolve({json:async()=>({flash:false,confirmed:true})});await work;
  assert.equal(s.nodes.get('flashButton').attrs['aria-pressed'],'false');
  assert.match(s.nodes.get('flashStatus').textContent,/chưa xác nhận/);
});

test('raw download freezes and saves exact original of displayed request, not the annotation',async()=>{
  const s=setup();let saved;
  const elementFactory=s.context.document.createElement;
  s.context.document.createElement=tag=>tag==='a' ? saved={click(){this.clicked=true},remove(){this.removed=true}} : elementFactory(tag);
  let rawBlob;s.context.URL.createObjectURL=blob=>{rawBlob=blob;return "blob:raw-original"};
  await s.run('liveDetection=true;renderResult({request_id:"raw",timestamp:"2026-10-11T00:00:00Z",source:"camera_pull",detections:[],detection_count:0,annotated_image_base64:"ANNOTATED",original_image_base64:"T1JJR0lOQUw=",original_image_content_type:"image/jpeg",original_image_filename:"cam32_raw.jpg"},true)');
  assert.equal(s.nodes.get('saveRawButton').disabled,false);
  await s.nodes.get('saveRawButton').onclick();
  assert.equal(saved.href,'blob:raw-original');assert.equal(await rawBlob.text(),'ORIGINAL');assert.equal(rawBlob.type,'image/jpeg');assert.equal(saved.removed,true);assert.equal(saved.download,'cam32_raw.jpg');assert.equal(saved.clicked,true);
  assert.equal(s.run('liveDetection'),false);assert.equal(s.nodes.get('frame').src,'data:image/jpeg;base64,ANNOTATED');
  s.run('pendingFrame("pending","working")');assert.equal(s.nodes.get('saveRawButton').disabled,true);
});

test('Live uses automatic Telegram policy rather than forcing every frame',()=>{
  const s=setup();s.nodes.get('liveButton').onclick();
  assert.equal(s.requests[0].route,'/camera/predict');
  assert.notEqual(JSON.parse(s.requests[0].options.body).send_telegram,true);
});


test('slow metrics never block the next Live capture and reporting stays bounded',async()=>{
  const s=setup();s.run('liveDetection=true');
  const first=s.run('liveCameraTick(liveGeneration)');
  const frame={request_id:'f1',report_token:'unit-token',source:'camera_pull',timestamp:'2026-10-12T00:00:00Z',
    detections:[],detection_count:0,annotated_image_base64:'FRAME',camera_fetch_ms:100,inference_ms:20};
  s.resolve({text:async()=>JSON.stringify(frame)},0);await first;
  assert.equal(s.run('liveBusy'),false);assert.deepEqual(s.calls,['/camera/predict','/metrics/report']);
  assert.equal(s.timers.at(-1).ms,50);
  const second=s.run('liveCameraTick(liveGeneration)');
  s.resolve({text:async()=>JSON.stringify({...frame,request_id:'f2'})},2);await second;
  const third=s.run('liveCameraTick(liveGeneration)');
  s.resolve({text:async()=>JSON.stringify({...frame,request_id:'f3'})},3);await third;
  assert.equal(s.calls.filter(x=>x==='/metrics/report').length,1);
  assert.equal(s.run('liveMetricsPending.values.request_id'),'f3');
  assert.equal(s.run('activeFrameResult.request_id'),'f3');
  s.resolve({json:async()=>({})},1);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(s.calls.filter(x=>x==='/metrics/report').length,2);
  assert.equal(JSON.parse(s.requests[4].options.body).request_id,'f3');
  s.resolve({json:async()=>({})},4);await new Promise(resolve=>setImmediate(resolve));
});

test('a camera timeout keeps the last matching image and boxes with a stale-frame label',async()=>{
  const s=setup();
  await s.run('liveDetection=true;renderResult({request_id:"held",source:"camera_pull",detections:[],detection_count:0,annotated_image_base64:"HELD",timestamp:"2026-10-12T00:00:00Z"},true)');
  const work=s.run('liveCameraTick(liveGeneration)');s.reject(new Error('camera timeout'));await work;
  assert.equal(s.nodes.get('frame').hidden,false);assert.equal(s.nodes.get('frame').src,'data:image/jpeg;base64,HELD');
  assert.match(s.nodes.get('frameSource').textContent,/GIỮ ẢNH CŨ/);
  assert.equal(s.run('activeFrameResult.request_id'),'held');assert.equal(s.timers.at(-1).ms,1000);
});

test('a metrics timeout does not mark a good camera frame as a Live failure',async()=>{
  const s=setup();s.run('liveDetection=true');const work=s.run('liveCameraTick(liveGeneration)');
  s.resolve({text:async()=>JSON.stringify({request_id:'ok',report_token:'unit',source:'camera_pull',detections:[],detection_count:0,annotated_image_base64:'FRAME'})});
  await work;s.reject(new Error('metrics timeout'),1);await new Promise(resolve=>setImmediate(resolve));
  assert.equal(s.run('liveFailure'),false);assert.equal(s.nodes.get('frame').hidden,false);
  assert.equal(s.timers.at(-1).ms,50);
});


test('cloud selection stays unavailable without server configuration',()=>{
  const s=setup();s.nodes.get('inferenceTarget').value='cloud';s.nodes.get('inferenceTarget').onchange();
  assert.equal(s.run('inferenceTarget'),'local');assert.equal(s.calls.length,0);
});

test('changing inference target stops Live and rejects the old delayed frame',async()=>{
  const s=setup();s.run('cloudConfigured=true;liveDetection=true');
  const work=s.run('liveCameraTick(liveGeneration)');
  s.nodes.get('inferenceTarget').value='cloud';s.nodes.get('inferenceTarget').onchange();
  s.resolve({text:async()=>JSON.stringify({request_id:'old',source:'camera_pull',detections:[],detection_count:0,annotated_image_base64:'OLD'})});
  await work;assert.equal(s.run('inferenceTarget'),'cloud');assert.equal(s.run('liveDetection'),false);
  assert.equal(s.nodes.get('frame').hidden,true);assert.match(s.nodes.get('frameSource').textContent,/CLOUD/);
});

test('cloud Live uses gateway WAN RTT and marks the real inference location',async()=>{
  const s=setup();let tick=0;s.context.performance.now=()=>++tick*1000;
  s.run('inferenceTarget="cloud";liveDetection=true');const work=s.run('liveCameraTick(liveGeneration)');
  assert.equal(JSON.parse(s.requests[0].options.body).inference_target,'cloud');
  assert.ok(s.requests[0].options.timeout>90000);
  s.resolve({text:async()=>JSON.stringify({request_id:'cloud-frame',report_token:'unit',source:'camera_pull',inference_location:'cloud',
    cloud_roundtrip_ms:500,inference_ms:200,detections:[],detection_count:0,annotated_image_base64:'CLOUD'})});
  await work;assert.equal(s.nodes.get('roundtripMs').textContent,'500.0');
  assert.equal(s.nodes.get('totalMs').textContent,'1000.0');
  assert.match(s.nodes.get('inferenceLocationLabel').textContent,/CLOUD/);
  assert.match(s.nodes.get('frameSource').textContent,/CLOUD/);
  const metrics=JSON.parse(s.requests[1].options.body);assert.equal(metrics.roundtrip_ms,500);assert.equal(metrics.total_ms,1000);
  s.resolve({json:async()=>({})},1);await new Promise(resolve=>setImmediate(resolve));
});


test('recognition history preserves mixed labels, unknown legacy and excludes failures',()=>{
  const s=setup();s.run('drawChart=()=>{}');
  s.context.testRows=[{request_id:'new',source:'camera_pull',detection_count:4,class_counts:{with_mask:2,without_mask:1,mask_weared_incorrect:1}},
                    {request_id:'old',source:'esp32',detection_count:3,class_counts:null},
                    {request_id:'failed',status:'failure',detection_count:null,class_counts:null}];
  s.run('renderHistory(testRows)');
  const body=s.nodes.get('recognitionBody');assert.equal(body.children.length,2);
  const chips=body.children[0].children[3].children[0].children;
  assert.deepEqual(chips.map(chip=>chip.children[1].textContent),[2,1,1]);
  assert.equal(body.children[1].children[3].children[0].textContent,'Chưa lưu nhãn');
});

test('recognition summary separates detected faces from requests and counts all saved history',()=>{
  const s=setup();s.context.testSummary={labeled_requests:4,unlabeled_requests:8,total_faces:7,
    classes:{with_mask:{detections:4,requests:2},without_mask:{detections:2,requests:1},mask_weared_incorrect:{detections:1,requests:1}}};
  s.run('renderRecognitionSummary(testSummary)');
  assert.equal(s.nodes.get('correctTotal').textContent,'4');assert.equal(s.nodes.get('correctRequests').textContent,'2 lượt có nhãn này');
  assert.equal(s.nodes.get('incorrectTotal').textContent,'1');
  assert.match(s.nodes.get('recognitionSummaryNote').textContent,/8 lượt cũ chưa lưu nhãn/);
  assert.match(s.nodes.get('recognitionSummaryNote').textContent,/nhiều frame/);
});

test('changing view clears current label chips and cannot keep stale class counts',async()=>{
  const s=setup();await s.run('renderResult({request_id:"current",source:"dashboard",detection_count:2,detections:[{class_id:2,confidence:.8,bbox:{x1:0,y1:0,x2:10,y2:10}},{class_id:2,confidence:.7,bbox:{x1:20,y1:0,x2:30,y2:10}}],annotated_image_base64:"CURRENT"})');
  assert.equal(s.nodes.get('currentCounts').children[2].children[1].textContent,2);
  s.run('pendingFrame("pending","working")');assert.equal(s.nodes.get('currentCounts').hidden,true);
  assert.equal(s.nodes.get('currentCounts').children.length,0);
});
