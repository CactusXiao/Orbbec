from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task_backend.tactile_viewer import TactileTimeline
from task_backend.viewer_service import ModeState, ViewerSession, ViewerSessionManager, ViewerSource, render_viewer_page


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def measurement(index=7, timestamp=1_000_010, value='2.5', **overrides):
    return dict(sample_index=index, touch_timestamp_us=timestamp, force_007_n=value,
                raw_adc_007=30, force_008_n='nan', raw_adc_008=0,
                calibrated_region_force_n='10', force_out_of_range='0', quality_flag='ok', **overrides)


class TactileViewerTest(unittest.TestCase):
    def test_uses_capture_index_not_csv_row_and_preserves_missing_force(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=42, ref_timestamp_us=1_000_000,
                touch_right_frame_index=7, touch_right_timestamp_us=1_000_010)])
            write_csv(root / 'touch/right_raw.csv', [measurement(index=2, value='1'), measurement(value='2.5')])
            timeline = TactileTimeline(root)
            hand = timeline.frame(42)['hands'][1]
            self.assertEqual(hand['status'], 'ready')
            self.assertEqual(hand['sample']['index'], 7)
            self.assertEqual(hand['sample']['force'][7], 2.5)
            self.assertIsNone(hand['sample']['force'][8])
            self.assertEqual(hand['sample']['adc'][8], 0)
            self.assertEqual(hand['delta_ms'], 0.01)
            json.dumps(timeline.frame(42), allow_nan=False)
            self.assertEqual(timeline.frame(43)['hands'][1]['status'], 'unaligned')
            self.assertEqual(timeline.frame(42)['hands'][0]['status'], 'missing')

    def test_explicit_missing_match_never_falls_back_to_nearby_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, ref_timestamp_us=1_000_000, touch_right_frame_index=-1)])
            write_csv(root / 'touch/right_raw.csv', [measurement()])
            self.assertIsNone(TactileTimeline(root).frame(0)['hands'][1]['sample'])

    def test_nearest_timestamp_respects_tolerance_and_stable_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=i, ref_timestamp_us=t) for i, t in enumerate((1_000_000, 1_030_000, 2_000_000))])
            write_csv(root / 'touch/right_raw.csv', [measurement(), measurement(index=9, timestamp=1_035_000, value='20')])
            timeline = TactileTimeline(root, 60)
            a, b, c = [timeline.frame(i)['hands'][1] for i in range(3)]
            self.assertEqual(a['sample']['index'], 7)
            self.assertEqual(b['sample']['index'], 9)
            self.assertEqual(a['force_max'], b['force_max'])
            self.assertEqual(b['force_max'], 20)
            self.assertEqual(c['status'], 'unaligned')
            self.assertIsNone(c['sample'])

    def test_large_time_difference_rejects_even_an_explicit_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, ref_timestamp_us=2_000_000, touch_right_frame_index=7)])
            write_csv(root / 'touch/right_raw.csv', [measurement()])
            hand = TactileTimeline(root).frame(0)['hands'][1]
            self.assertEqual(hand['status'], 'unaligned')
            self.assertIsNone(hand['sample'])

    def test_legacy_adc_is_not_relabelled_as_newtons(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, ref_timestamp_us=1_000_000)])
            write_csv(root / 'touch/right_raw.csv', [dict(sample_index=7, touch_timestamp_s='1.000010', pressure_007='42')])
            hand = TactileTimeline(root).frame(0)['hands'][1]
            self.assertFalse(hand['has_force'])
            self.assertEqual(hand['sample']['adc'][7], 42)
            self.assertIsNone(hand['sample']['force'][7])
            self.assertIsNone(hand['sample']['total_n'])

    def test_custom_stream_and_saturated_values_remain_json_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, touch_gloveR_frame_index=7)])
            row = measurement(value='nan')
            row.update(calibrated_region_force_n='inf', force_out_of_range='1')
            write_csv(root / 'sensors/record.csv', [row])
            (root / 'sensors/touch_manifest.json').write_text(json.dumps(dict(devices=[dict(id='gloveR', side='right', raw_csv='record.csv')])))
            hand = TactileTimeline(root).frame(0)['hands'][0]
            self.assertEqual(hand['status'], 'ready')
            self.assertIsNone(hand['delta_ms'])
            self.assertTrue(hand['sample']['out_of_range'])
            self.assertIsNone(hand['sample']['total_n'])
            json.dumps(hand, allow_nan=False)

    def test_manifest_cannot_read_outside_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'episode'
            (root / 'touch').mkdir(parents=True)
            outside = Path(tmp) / 'secret.csv'
            write_csv(outside, [measurement()])
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, touch_right_frame_index=7)])
            for name in ('../../secret.csv', str(outside), 'linked.csv'):
                if name == 'linked.csv':
                    (root / 'touch/linked.csv').symlink_to(outside)
                (root / 'touch/touch_manifest.json').write_text(json.dumps(dict(devices=[dict(id='right', raw_csv=name)])))
                self.assertEqual(TactileTimeline(root).frame(0)['hands'][0]['status'], 'missing')

    def test_pico_preparation_writes_synced_json_without_mutating_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = root / 'episode'
            rgb = episode / 'ego/RGB'
            rgb.mkdir(parents=True)
            (rgb / '00003.jpg').write_bytes(b'image')
            write_csv(episode / 'timestamps.csv', [dict(frame_index=42, ref_timestamp_us=1_000_000, touch_right_frame_index=7)])
            write_csv(episode / 'touch/right_raw.csv', [measurement()])
            before = {str(p.relative_to(episode)): p.read_bytes() for p in episode.rglob('*') if p.is_file()}
            session = ViewerSession('test', 'episode', episode, root / 'preview', 'ffmpeg', state='ready', frame_indices=[42],
                sources=[ViewerSource('ego:ego', 'Pico', 'ego', 'ego', rgb_dir=rgb, rgb_frame_map={42: 3})])
            manager = ViewerSessionManager(temp_root=root / 'sessions')
            try:
                manager._sessions[session.session_id] = session
                with patch.object(manager, '_render_pico_frame', side_effect=lambda source, destination, *args: destination.write_bytes(source.read_bytes())):
                    manager._prepare_pico(session, session.modes['pico'])
                session.modes['pico'].status = 'ready'
                path, mime = manager.media_path('test', 'pico', 'tactile', 42)
                self.assertEqual(mime, 'application/json')
                data = json.loads(path.read_text())
                self.assertEqual(data['frame_index'], 42)
                self.assertEqual(data['hands'][1]['sample']['index'], 7)
                self.assertEqual(before, {str(p.relative_to(episode)): p.read_bytes() for p in episode.rglob('*') if p.is_file()})
                media, _ = manager.media_path('test', 'pico', 'ego', 42)
                self.assertEqual(media.read_bytes(), b'image')
            finally:
                manager.shutdown()

    def test_page_includes_atomic_tactile_render_and_prefetch(self):
        page = render_viewer_page('episode')
        self.assertIn('id="tactilePanel"', page)
        self.assertIn('data-touch-unit="N"', page)
        self.assertIn("if(mode==='pico')renderTactile(tactile)", page)
        self.assertIn("cachedTactile(mediaUrl('tactile',frame))", page)
        self.assertIn('未提供左手位置映射', page)
        self.assertIn('色阶在整段采集中保持固定', page)


if __name__ == '__main__':
    unittest.main()
