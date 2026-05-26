import time
import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("archive/normal/normal-1.mp4")
frames = []
for _ in range(32):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

print(f"Loaded {len(frames)} frames.")

# Test CPU
t0 = time.time()
results_cpu = model(frames, verbose=False, device="cpu")
t_cpu = time.time() - t0
print(f"CPU time for {len(frames)} frames: {t_cpu:.3f}s ({len(frames)/t_cpu:.1f} fps)")

# Test MPS
try:
    t0 = time.time()
    results_mps = model(frames, verbose=False, device="mps")
    t_mps = time.time() - t0
    print(f"MPS time for {len(frames)} frames: {t_mps:.3f}s ({len(frames)/t_mps:.1f} fps)")
except Exception as e:
    print("MPS failed:", e)
