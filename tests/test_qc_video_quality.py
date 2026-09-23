import json,tempfile,unittest
from pathlib import Path
from PIL import Image
from remote_frontend.layered_preview import layout,pack_frame,content_layout
from remote_frontend.browser_media import qc_encoding_settings
class QualityTest(unittest.TestCase):
 def test_detail_geometry_and_layer_identity(self):
  cameras=['00','02','03','05','ego']
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);paths={}
   for c in cameras:
    (root/c).mkdir();(root/'mesh'/c).mkdir(parents=True)
    size=(2328,1748) if c=='ego' else (640,400)
    paths[c]=root/c/'00003.jpg';Image.new('RGB',size,'white').save(paths[c]);Image.new('RGB',size,'red').save(root/'mesh'/c/'00003.jpg')
   p=pack_frame(root,cameras,3,'detail960');info=content_layout(cameras,paths,'detail960')
   self.assertEqual(p.size,(2240,1600))
   for t in info['tiles']:
    x,y,w,h=t['raw'];xx,yy,ww,hh=t['rendered']
    self.assertEqual((xx,yy,ww,hh),(x,y+800,w,h))
    if t['camera']!='ego':self.assertEqual((w,h),(640,400))
    else:self.assertGreaterEqual(w,958);self.assertEqual(h,720)
    self.assertGreater(min(p.getpixel((x+w//2,y+h//2))),245)
    self.assertGreater(p.getpixel((xx+ww//2,yy+hh//2))[0],245)
 def test_all_camera_counts_fit_nonoverlapping_planes(self):
  for cams in [['ego'],['00'],['00','02','03','05','06','ego']]:
   for profile in ['detail960','detail1280']:
    info=layout(cams,profile);boxes=[]
    for t in info['tiles']:
     for key in ['raw','rendered']:
      x,y,w,h=t[key];self.assertGreaterEqual(x,0);self.assertGreaterEqual(y,0)
      self.assertLessEqual(x+w,info['width']);self.assertLessEqual(y+h,info['height'])
      for a,b,c,d in boxes:self.assertFalse(x<a+c and a<x+w and y<b+d and b<y+h)
      boxes.append((x,y,w,h))
 def test_resume_pins_profile_and_old_cache_is_compatible(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);cfg={'qc_video_profile':'detail960','qc_video_encoder':'h264_nvenc'}
   first=qc_encoding_settings(root,cfg)
   self.assertEqual(first['profile'],'detail960')
   self.assertEqual(qc_encoding_settings(root,{'qc_video_profile':'detail1280'}),first)
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'ready.json').write_text(json.dumps({'complete':False,'chunks':[{'index':0}]}))
   self.assertEqual(qc_encoding_settings(root,cfg)['profile'],'legacy')
if __name__=='__main__':unittest.main()
