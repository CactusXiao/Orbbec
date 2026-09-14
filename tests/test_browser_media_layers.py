"""Frame identity, packing and lookahead bounds for outsourcing-only media."""
import json, shutil, subprocess, tempfile, unittest
from pathlib import Path
from PIL import Image
from remote_frontend.layered_preview import pack_frame, layout, encode_layered, content_layout
ROOT=Path(__file__).resolve().parents[1]
NODE=shutil.which('node') or '/Users/cactusxiao/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node'
class LayeredMediaTest(unittest.TestCase):
 def test_matching_frame_and_letterbox(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)
   for camera in ('00','ego'):
    (root/camera).mkdir();(root/'mesh'/camera).mkdir(parents=True)
    for frame,color in [(3,'red'),(11,'blue')]:
     Image.new('RGB',(640,480),color).save(root/camera/f'{frame:05d}.png')
     Image.new('RGB',(640,480),'green').save(root/'mesh'/camera/f'{frame:05d}.jpg')
   packed=pack_frame(root,['00','ego'],11)
   for part in layout(['00','ego'])['tiles']:
    x,y,w,h=part['raw'];self.assertEqual(packed.getpixel((x+w//2,y+h//2)),(0,0,255))
    x,y,w,h=part['rendered'];r,g,b=packed.getpixel((x+w//2,y+h//2));self.assertGreater(g,100);self.assertLess(r,5)
   Image.new('RGB',(320,240)).save(root/'mesh/ego/00011.jpg')
   with self.assertRaisesRegex(ValueError,'dimensions differ'):pack_frame(root,['ego'],11)
 def test_content_bounds_exclude_padding_and_keep_both_layers_aligned(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);paths={}
   for camera,size in [('00',(640,400)),('ego',(800,800))]:
    (root/camera).mkdir();(root/'mesh'/camera).mkdir(parents=True)
    paths[camera]=root/camera/'00000.jpg'
    Image.new('RGB',size,'white').save(paths[camera]);Image.new('RGB',size,'red').save(root/'mesh'/camera/'00000.jpg')
   packed=pack_frame(root,['00','ego'],0);info=content_layout(['00','ego'],paths)
   self.assertTrue(info['content_bounds'])
   self.assertEqual(info['tiles'][0]['raw'],[0,68,512,320])
   self.assertEqual(info['tiles'][1]['raw'],[564,24,408,408])
   for part in info['tiles']:
    x,y,w,h=part['raw'];rx,ry,rw,rh=part['rendered']
    self.assertEqual((rx,ry,rw,rh),(x,y+864,w,h))
    for px,py in [(x,y),(x+w-1,y),(x,y+h-1),(x+w-1,y+h-1)]:
     self.assertTrue(all(c>245 for c in packed.getpixel((px,py))))
    self.assertNotEqual(packed.getpixel((x,y-1)),(255,255,255))
 def test_existing_video_cache_gets_content_bounds_without_reencoding(self):
  from remote_frontend.browser_media import BrowserMedia
  from types import SimpleNamespace
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);folder=root/'s/layered-v1';(folder/'raw_frames/00').mkdir(parents=True)
   Image.new('RGB',(640,400),'white').save(folder/'raw_frames/00/19.jpg')
   old=dict(ready=True,complete=True,cameras=['00'],layout=layout(['00']))
   text=json.dumps(old);(folder/'ready.json').write_text(text)
   media=BrowserMedia(SimpleNamespace(session=lambda sid:{'role':'qc'}),root,{})
   result=media.status('s')
   self.assertEqual(result['layout']['tiles'][0]['raw'],[0,68,512,320])
   self.assertEqual((folder/'ready.json').read_text(),text)
   self.assertFalse(media.running)
 def test_buffer_window_and_disjoint_label_frames(self):
  code='''import assert from 'node:assert/strict';
   import {chunkWindow} from './remote_frontend/web/player.js';
   import {labelLookahead} from './remote_frontend/web/frame-cache.js';
   const chunks=Array.from({length:100},(_,i)=>({index:i,start:i*90,count:90}));
   for(const t of [0,1,16,99,299]) {
    const p=chunkWindow(chunks,t,30,9000);
    assert.equal(p[0].index,Math.floor(t/3));
    assert(p.every(c=>c.start/30 < Math.min(300,t+60) && (c.start+c.count)/30>Math.max(0,t-15)));
    assert(p.length<=26);
   }
   const p=labelLookahead([2,5,11,40],['00','02'],1,'02',false);
   assert.deepEqual(p.slice(0,2),[{camera:'02',frame:11},{camera:'02',frame:40}]);
   assert(p.every(x=>[2,5,11,40].includes(x.frame)));
   assert.equal(new Set(p.map(x=>x.camera+':'+x.frame)).size,p.length);
  '''
  subprocess.run([NODE,'--input-type=module','-e',code],cwd=ROOT,check=True,capture_output=True,text=True)
 @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'),'ffmpeg required')
 def test_encoded_timestamps_and_dimensions(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'00').mkdir();(root/'mesh/00').mkdir(parents=True)
   for f in (1,8,19):
    Image.new('RGB',(64,48),(f*10,0,0)).save(root/'00'/f'{f:05d}.jpg')
    Image.new('RGB',(64,48),(0,f*10,0)).save(root/'mesh/00'/f'{f:05d}.jpg')
   out=root/'test.mp4';encode_layered(root,out,[1,8,19],['00'])
   value=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_frames','-show_streams','-of','json',str(out)]))
   stream=value['streams'][0];self.assertEqual((stream['width'],stream['height']),(1536,1728));self.assertEqual(stream['has_b_frames'],0)
   self.assertEqual(len(value['frames']),3)
   for i,f in enumerate(value['frames']):self.assertAlmostEqual(float(f['best_effort_timestamp_time']),i/30,places=5)
if __name__=='__main__':unittest.main()
