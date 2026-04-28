import cv2
import hashlib

def id_to_color(obj_id):
    h = hashlib.md5(str(obj_id).encode()).hexdigest()
    r = max(int(h[0:2], 16), 80)
    g = max(int(h[2:4], 16), 80)
    b = max(int(h[4:6], 16), 80)
    return (b, g, r)

class Visualizer:
    def __init__(self, box_thickness=2, font_scale=0.55, show_confidence=True):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.show_confidence = show_confidence

    def draw(self, frame, tracked_objects):
        for obj in tracked_objects:
            x1, y1, x2, y2 = [int(v) for v in obj['bbox']]
            obj_id = obj['id']
            cls = obj['class']
            conf = obj['confidence']
            color = id_to_color(obj_id)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)
            label = f'ID:{obj_id} {cls} {conf:.2f}' if self.show_confidence else f'ID:{obj_id} {cls}'
            (tw, th), baseline = cv2.getTextSize(label, self.font, self.font_scale, 1)
            label_y = max(y1 - 5, th + 5)
            cv2.rectangle(frame, (x1, label_y - th - baseline - 2), (x1 + tw + 2, label_y + baseline - 2), color, -1)
            cv2.putText(frame, label, (x1 + 1, label_y - 2), self.font, self.font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'Active tracks: {len(tracked_objects)}', (10, 30), self.font, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return frame
