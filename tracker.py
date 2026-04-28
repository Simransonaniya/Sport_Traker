def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area else 0.0

class Track:
    def __init__(self, track_id, bbox, class_name='person', confidence=0.9):
        self.id = track_id
        self.bbox = bbox
        self.age = 0
        self.missed = 0
        self.class_name = class_name
        self.confidence = confidence

class Tracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        self.next_id = 1
        self.tracks = []
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, detections):
        unmatched_tracks = list(range(len(self.tracks)))
        unmatched_detections = list(range(len(detections)))
        matches = []
        for det_idx, detection in enumerate(detections):
            best_match = None
            best_iou = self.iou_threshold
            for trk_idx in unmatched_tracks:
                overlap = iou(self.tracks[trk_idx].bbox, detection['bbox'])
                if overlap > best_iou:
                    best_iou = overlap
                    best_match = trk_idx
            if best_match is not None:
                matches.append((best_match, det_idx))
                unmatched_tracks.remove(best_match)
                unmatched_detections.remove(det_idx)
        for trk_idx, det_idx in matches:
            self.tracks[trk_idx].bbox = detections[det_idx]['bbox']
            self.tracks[trk_idx].confidence = detections[det_idx]['confidence']
            self.tracks[trk_idx].missed = 0
            self.tracks[trk_idx].age += 1
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            self.tracks.append(Track(self.next_id, det['bbox'], det.get('class', 'person'), det.get('confidence', 0.9)))
            self.next_id += 1
        for trk_idx in sorted(unmatched_tracks, reverse=True):
            self.tracks[trk_idx].missed += 1
            if self.tracks[trk_idx].missed > self.max_age:
                self.tracks.pop(trk_idx)
        return [{'id': t.id, 'bbox': t.bbox, 'confidence': t.confidence, 'class': t.class_name} for t in self.tracks if t.missed == 0]
