from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from task_backend.tactile_viewer import TactileTimeline
from task_backend.viewer_service import ModeState, ViewerSession, ViewerSessionManager, ViewerSource, render_viewer_page


def write_manifest(directory, devices=None, schema='orbbec.touch.jq_shroom.v3'):
    directory.mkdir(parents=True, exist_ok=True)
    if devices is None:
        devices = [dict(id=side, side=side, sensor_type=t, raw_csv=side+'_raw.csv')
                   for side, t in (('left', 1), ('right', 2))]
    (directory / 'touch_manifest.json').write_text(json.dumps(dict(schema=schema, devices=devices)))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if path.parent.name == 'touch' and path.name.endswith('_raw.csv'):
        write_manifest(path.parent)


def measurement(index=7, timestamp=1_000_010, value='2.5', **overrides):
    row = dict(sample_index=index, touch_timestamp_us=timestamp,
               calibrated_region_force_n=value, force_out_of_range='0', quality_flag='ok',
               force_calibration_status='right_middle_region')
    row.update({f'raw_adc_{i:03d}': 0 for i in range(256)})
    row['raw_adc_007'] = 30
    row.update(overrides)
    return row


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
            self.assertNotIn('force', hand['sample'])
            self.assertEqual(hand['sample']['total_n'], 2.5)
            self.assertEqual(hand['sample']['force_status'], 'right_middle_region')
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

    def test_rejects_old_and_missing_manifests_and_old_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, ref_timestamp_us=1_000_000)])
            path = root / 'touch/right_raw.csv'
            write_csv(path, [measurement()])
            for schema in ('orbbec.touch.jq_shroom.v1', 'orbbec.touch.jq_shroom.v2'):
                write_manifest(path.parent, schema=schema)
                result = TactileTimeline(root).frame(0)
                self.assertEqual(result['hands'], [])
                self.assertIn('格式不支持', result['error'])
            (path.parent / 'touch_manifest.json').unlink()
            result = TactileTimeline(root).frame(0)
            self.assertEqual(result['hands'], [])
            self.assertIn('缺少', result['error'])
            # A v3 manifest cannot make legacy CSV content acceptable.
            for old in (dict(sample_index=7, touch_timestamp_s='1.000010', pressure_007='42'),
                        dict(sample_index=7, touch_timestamp_us=1_000_010, force_007_n='2.5'),
                        measurement(force_007_n='2.5')):
                write_csv(path, [old])
                hand = TactileTimeline(root).frame(0)['hands'][1]
                self.assertIsNone(hand['sample'])
                self.assertIn('格式不支持', hand['message'])

    def test_rejects_conflicting_hand_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'touch/right_raw.csv', [measurement()])
            write_manifest(root / 'touch', [dict(id='right', side='left', sensor_type=2, raw_csv='right_raw.csv')])
            result = TactileTimeline(root).frame(0)
            self.assertEqual(result['hands'], [])
            self.assertIn('不一致', result['error'])

    def test_custom_stream_and_saturated_values_remain_json_safe(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, touch_gloveR_frame_index=7)])
            row = measurement(value='nan')
            row.update(calibrated_region_force_n='inf', force_out_of_range='1')
            write_csv(root / 'sensors/record.csv', [row])
            write_manifest(root / 'sensors', [dict(id='gloveR', side='right', sensor_type=2, raw_csv='record.csv')])
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
                write_manifest(root / 'touch', [dict(id='right', side='right', sensor_type=2, raw_csv=name)])
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

    def test_manual_layout_and_bend_channels(self):
        from task_backend.tactile_layout import glove_layout
        for side in ("left", "right"):
            layout = glove_layout(side)
            self.assertEqual([len(row) for row in layout['palm_rows']], [12,15,15,15,15])
            fingers = [i for finger in layout['fingers'] for i in finger['ids']]
            pressure = fingers + [i for row in layout['palm_rows'] for i in row]
            bends = [finger['bend_id'] for finger in layout['fingers']]
            self.assertEqual(len(set(pressure)), 132)
            self.assertEqual(len(set(pressure + bends)), 137)
            self.assertTrue(all(1 <= i <= 256 for i in pressure + bends))
        self.assertEqual(glove_layout('right')['palm_first_offset'], 3)
        self.assertEqual(glove_layout('left')['palm_first_offset'], 0)
        self.assertEqual(glove_layout('left')['fingers'][2]['ids'], [25,24,23,9,8,7,249,248,247,233,232,231])
        self.assertIsNone(glove_layout('unknown'))

    def test_v3_region_only_and_left_bend_adc(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_csv(root / 'timestamps.csv', [dict(frame_index=0, ref_timestamp_us=1_000_000)])
            write_csv(root / 'touch/right_raw.csv', [measurement(force_calibration_status='right_middle_region', raw_adc_040=77)])
            write_csv(root / 'touch/left_raw.csv', [measurement(force_calibration_status='uncalibrated_hand', raw_adc_215=66)])
            left, right = TactileTimeline(root).frame(0)['hands']
            self.assertEqual(right['sample']['total_n'], 2.5)
            self.assertNotIn('force', right['sample'])
            self.assertEqual(right['sample']['adc'][40], 77)
            self.assertEqual(len(right['calibrated_sensor_ids']), 12)
            self.assertIsNone(left['sample']['total_n'])
            self.assertFalse(left['has_force'])
            self.assertEqual(left['calibrated_sensor_ids'], [])
            self.assertEqual(left['sample']['force_status'], 'uncalibrated_hand')
            self.assertEqual(left['sample']['adc'][215], 66)
            self.assertEqual(left['layout']['fingers'][2]['bend_id'], 216)
            write_csv(root / 'touch/left_raw.csv', [measurement(force_calibration_status='uncalibrated_hand')])
            left = TactileTimeline(root).frame(0)['hands'][0]
            self.assertEqual(left['sample']['force_status'], 'uncalibrated_hand')
            self.assertIsNone(left['sample']['total_n'])

    def test_page_includes_atomic_tactile_render_and_prefetch(self):
        page = render_viewer_page('episode')
        self.assertIn('id="tactilePanel"', page)
        self.assertIn('data-touch-unit="N"', page)
        self.assertIn("if(mode==='pico')renderTactile(tactile)", page)
        self.assertIn("cachedTactile(mediaUrl('tactile',frame))", page)
        self.assertNotIn('未提供左手位置映射', page)
        self.assertIn('五指弯曲', page)
        self.assertIn('没有逐点力标定', page)
        self.assertIn('色阶在整段采集中保持固定', page)


if __name__ == '__main__':
    unittest.main()
