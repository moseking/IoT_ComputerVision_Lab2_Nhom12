const {test} = require('node:test');
const assert = require('node:assert/strict');
const {WebcamCapture} = require('../web/webcam.js');
function fixture(options = {}) {
  let stopped = 0, calls = 0, constraints;
  const track = {readyState:'live', stop(){stopped++;this.readyState='ended'}};
  const stream = {getTracks:()=>[track],getVideoTracks:()=>[track]};
  const video = {videoWidth:1920,videoHeight:1080,pause(){},play:async()=>{},srcObject:null};
  let canvas;
  const camera = new WebcamCapture(video,{secureContext:true,mediaDevices:{getUserMedia:async c=>{calls++;constraints=c;return stream}},
    now:()=>20,makeCanvas:()=>canvas={getContext:()=>({drawImage(...args){canvas.drawn=args}}),toBlob(fn,type){fn(new Blob(['JPEG_FRAME'],{type}))}},...options});
  return {camera,stream,video,track,get stopped(){return stopped},get calls(){return calls},get constraints(){return constraints},get canvas(){return canvas}};
}
test('construction never activates hardware; start requests video only and reuses one stream',async()=>{
  const f=fixture();assert.equal(f.calls,0);
  const [one,two]=await Promise.all([f.camera.start(),f.camera.start()]);
  assert.equal(one,two);assert.equal(f.calls,1);assert.equal(f.constraints.audio,false);
  assert.equal(f.camera.ready,true);f.camera.stop();assert.equal(f.stopped,1);assert.equal(f.video.srcObject,null);
});
test('an ignored permission request can be cancelled; late granted tracks are closed',async()=>{
  let resolve;const f=fixture({mediaDevices:{getUserMedia:()=>new Promise(r=>resolve=r)}});
  const started=f.camera.start();f.camera.stop();resolve(f.stream);
  await assert.rejects(started,{name:'AbortError'});assert.equal(f.stopped,1);assert.equal(f.camera.ready,false);
});
test('denied permission is actionable and leaves no active stream',async()=>{
  const f=fixture({mediaDevices:{getUserMedia:async()=>{const e=new Error('denied');e.name='NotAllowedError';throw e}}});
  await assert.rejects(f.camera.start(),e=>e.name==='WebcamError' && /quyền camera/.test(e.message));
  assert.equal(f.camera.ready,false);assert.equal(f.camera.starting,false);
});
test('plain HTTP LAN context never calls getUserMedia and directs the user to localhost',async()=>{
  const f=fixture({secureContext:false});await assert.rejects(f.camera.start(),/127.0.0.1/);assert.equal(f.calls,0);
});
test('JPEG capture bounds dimensions, preserves aspect ratio and returns actual submitted bytes',async()=>{
  const f=fixture();const frame=await f.camera.capture();
  assert.equal(frame.width,1280);assert.equal(frame.height,720);assert.equal(frame.blob.type,'image/jpeg');
  assert.equal(await frame.blob.text(),'JPEG_FRAME');assert.equal(f.canvas.drawn[0],f.video);assert.equal(frame.capture_ms,0);
  f.camera.stop();
});
test('stopping during JPEG encoding rejects the stale frame',async()=>{
  let encoded;const f=fixture({makeCanvas:()=>({getContext:()=>({drawImage(){}}),toBlob:fn=>encoded=fn})});
  const frame=f.camera.capture();await new Promise(r=>setImmediate(r));f.camera.stop();encoded(new Blob(['late'],{type:'image/jpeg'}));
  await assert.rejects(frame,{name:'AbortError'});
});
test('encoder failure never produces an uploadable frame',async()=>{
  const f=fixture({makeCanvas:()=>({getContext:()=>({drawImage(){}}),toBlob:fn=>fn(null)})});
  await assert.rejects(f.camera.capture(),/mã hóa/);f.camera.stop();
});
