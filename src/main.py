from oop_face_detector import FaceDetector
import cv2

fdec = FaceDetector(
    frame_interval=2,
    threshold_similarity=0.45
)

known_embeddings = fdec.load_known_embeddings(
    "data/known_faces"
)

frames = fdec.extract_frames_every_n_seconds(
    "data/videos/trimmed_clip.mp4"
)

faces_in_frames = fdec.identify_faces_in_frames(
    frames,
    known_embeddings
)

metadata_json_path = fdec.metadata_detection(
    faces_in_frames)

# Save annotated images
for i, (t, frame, _) in enumerate(faces_in_frames):
    cv2.imwrite(f"output/saved_frames/frame_{i:04d}.jpg", frame)

print("Metadata saved to:", metadata_json_path)