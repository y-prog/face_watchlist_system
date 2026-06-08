import cv2
from insightface.app import FaceAnalysis
import os
import numpy as np
import shutil
import subprocess
import json

class FaceDetector:
    def __init__(self, frame_interval, threshold_similarity,
     model_name="buffalo_l", ctx_id=0, det_size=(640, 640)):
        self.frame_interval = frame_interval
        self.threshold_similarity = threshold_similarity
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        
        
    def load_known_embeddings(self, faces_dir):
        known = {}
        for fn in os.listdir(faces_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(faces_dir, fn)
            img = cv2.imread(path)
            faces = self.app.get(img)
            emb = faces[0].embedding
            norm = np.linalg.norm(emb)
            name = os.path.splitext(fn)[0]
            known[name] = emb / norm
        return known

    def detect_face_rects(self, frame, min_score=0.3):
        results = []
        for t, frame in frames:
            out = frame.copy()
            for f in self.app.get(frame):
                score = getattr(f, "det_score", None) or getattr(f, "score", 0.0)
                if score < min_score:
                    continue
                x1, y1, x2, y2 = f.bbox.astype(int).tolist()
                cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(out, f"{score:.2f}", (x1, max(20, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            results.append((t, out))
        return results

    def extract_frames_every_n_seconds(self, video_path):
        frames = []
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        if ret:
            frames.append((0.0, frame))
        t = float(self.frame_interval)
        while True:
            cap.set(cv2.CAP_PROP_POS_MSEC, int(t * 1000))
            ret, frame = cap.read()
            if not ret:
                break
            frames.append((t, frame))
            t += self.frame_interval
        cap.release()
        return frames

    def identify_faces_in_frames(self, frames, known):
        results = []
        for t, frame in frames:
            out = frame.copy()
            detections = []
            faces = self.app.get(frame)  # list of face objects
            for f in faces:
                emb = f.embedding
                norm = np.linalg.norm(emb)
                if norm <= 0:
                    continue
                emb_n = emb / norm
                best = None
                for name, kemb in known.items():
                    score = float(np.dot(emb_n, kemb))  # cosine similarity
                    if score >= self.threshold_similarity and (best is None or score > best[1]):
                        best = (name, score)
                if best:
                    x1, y1, x2, y2 = f.bbox.astype(int).tolist()
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(out, f"{best[0]} {best[1]:.2f}", (x1, max(20, y1-10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    detections.append((best[0], float(best[1]), (x1, y1, x2, y2)))
            results.append((t, out, detections))
        return results

    def metadata_detection(self, metadata_list):
        detections_per_frame = [m[2] for m in metadata_list]
        out = []
        for i, frame in enumerate(detections_per_frame):
            secs = i * self.frame_interval
            detections = []
            for name, score, bbox in frame:
                detections.append({
                    "name": name,
                    "score": float(score),
                    "bbox": [int(b) for b in bbox],
                    "secs": secs
                })
            out.append(detections)
        out_path = "output/json_files/detections.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        return out_path