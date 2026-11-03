```
===========================================================================
EXTRACTION CODE RECOVERY
===========================================================================
Matching code cells: 88

===========================================================================
CELL INDEX: 60
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)

manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)

# Choose five reproducible sample videos.
sample = df.sample(n=5, random_state=42)

# MediaPipe Tasks API: hand landmark detector.
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "/tmp/hand_landmarker.task"
import urllib.request

model_url = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

urllib.request.urlretrieve(model_url, model_path)

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

results = []

with vision.HandLandmarker.create_from_options(options) as detector:
    for _, row in sample.iterrows():
        video_path = dataset_root / row["video_path"]
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(
            0, max(0, total_frames - 1),
            min(10, total_frames),
            dtype=int
        )

        checked = 0
        detected = 0

        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()

            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(mp_image)
            checked += 1

            if result.hand_landmarks:
                detected += 1

        cap.release()

        results.append({
            "video_path": row["video_path"],
            "frames_checked": checked,
            "frames_with_hands": detected,
            "detection_rate_percent": round(
                100 * detected / checked, 1
            ) if checked else 0
        })

results_df = pd.DataFrame(results)

print("=== MEDIAPIPE LANDMARK TEST ===")
print(results_df.to_string(index=False))
print("\nTotal frames checked:", results_df["frames_checked"].sum())
print("Frames with hand detections:", results_df["frames_with_hands"].sum())

===========================================================================
CELL INDEX: 66
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("1. Loading manifest...")

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)

manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)
video_path = dataset_root / df.iloc[0]["video_path"]

print("2. Selected video:", video_path.name)

model_path = "/tmp/hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

print("3. Initializing MediaPipe...")

with vision.HandLandmarker.create_from_options(options) as detector:
    cap = cv2.VideoCapture(str(video_path))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sorted(set([
        0,
        total_frames // 2,
        max(0, total_frames - 1)
    ]))

    print("4. Video frames:", total_frames)

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            print(f"Frame {frame_idx}: could not read")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(image)
        hand_count = len(result.hand_landmarks)

        print(f"Frame {frame_idx}: detected hands = {hand_count}")

    cap.release()

print("5. Single-video test completed.")

===========================================================================
CELL INDEX: 67
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)
manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)
sample = df.sample(n=5, random_state=42)

model_path = "/tmp/hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

rows = []

with vision.HandLandmarker.create_from_options(options) as detector:
    for _, row in sample.iterrows():
        video_path = dataset_root / row["video_path"]
        cap = cv2.VideoCapture(str(video_path))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(
            0, total_frames - 1, min(5, total_frames), dtype=int
        )

        checked = 0
        detected = 0

        print(f"\nTesting: {row['video_path']}")

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()

            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = detector.detect(image)

            checked += 1
            hands = len(result.hand_landmarks)

            if hands > 0:
                detected += 1

            print(f"  Frame {idx}: {hands} hand(s)")

        cap.release()

        rows.append({
            "video_path": row["video_path"],
            "frames_checked": checked,
            "frames_with_hands": detected,
            "detection_rate_percent": round(
                100 * detected / checked, 1
            ) if checked else 0
        })

results_df = pd.DataFrame(rows)

print("\n=== FIVE-VIDEO SUMMARY ===")
print(results_df.to_string(index=False))
print(
    "\nOverall frame detection rate:",
    round(
        100 * results_df["frames_with_hands"].sum()
        / max(1, results_df["frames_checked"].sum()),
        1
    ),
    "%"
)

===========================================================================
CELL INDEX: 68
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)

manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)
row = df[df["video_path"].str.contains(
    "People/61. Father/MVI_5082.MOV", regex=False
)].iloc[0]

video_path = dataset_root / row["video_path"]
model_path = "/tmp/hand_landmarker.task"

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, total_frames - 1, 5, dtype=int)

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

with vision.HandLandmarker.create_from_options(options) as detector:
    for ax, idx in zip(axes, indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            ax.set_title(f"Frame {idx}: unreadable")
            ax.axis("off")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = detector.detect(image)

        annotated = rgb.copy()

        for hand_landmarks in result.hand_landmarks:
            landmark_list = landmark_pb2.NormalizedLandmarkList(
                landmark=[
                    landmark_pb2.NormalizedLandmark(
                        x=lm.x, y=lm.y, z=lm.z
                    )
                    for lm in hand_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated,
                landmark_list,
                solutions.hands.HAND_CONNECTIONS
            )

        ax.imshow(annotated)
        ax.set_title(
            f"Frame {idx}\nHands: {len(result.hand_landmarks)}"
        )
        ax.axis("off")

cap.release()
plt.tight_layout()
plt.show()

===========================================================================
CELL INDEX: 69
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)

manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)

row = df[
    df["video_path"].str.contains(
        "People/61. Father/MVI_5082.MOV", regex=False
    )
].iloc[0]

video_path = dataset_root / row["video_path"]
model_path = "/tmp/hand_landmarker.task"

# Hand landmark connections (MediaPipe's 21-point hand topology)
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),(19,20),
]

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

cap = cv2.VideoCapture(str(video_path))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, total_frames - 1, 5, dtype=int)

fig, axes = plt.subplots(1, 5, figsize=(18, 5))

with vision.HandLandmarker.create_from_options(options) as detector:
    for ax, idx in zip(axes, indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            ax.set_title(f"Frame {idx}: unreadable")
            ax.axis("off")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = detector.detect(image)

        ax.imshow(rgb)

        for hand in result.hand_landmarks:
            xs = [point.x * rgb.shape[1] for point in hand]
            ys = [point.y * rgb.shape[0] for point in hand]

            for a, b in connections:
                ax.plot(
                    [xs[a], xs[b]], [ys[a], ys[b]],
                    linewidth=2
                )

            ax.scatter(xs, ys, s=10)

        ax.set_title(
            f"Frame {idx}\nHands: {len(result.hand_landmarks)}"
        )
        ax.axis("off")

cap.release()
plt.tight_layout()
plt.show()

===========================================================================
CELL INDEX: 70
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from mediapipe.tasks import python, vision

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)
manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)
row = df[df["video_path"].str.contains(
    "People/61. Father/MVI_5082.MOV", regex=False
)].iloc[0]

cap = cv2.VideoCapture(str(dataset_root / row["video_path"]))
cap.set(cv2.CAP_PROP_POS_FRAMES, 16)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read frame 16.")

# Compare the original frame with a 1.5x resized frame.
resized = cv2.resize(
    frame, None, fx=1.5, fy=1.5,
    interpolation=cv2.INTER_CUBIC
)

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_hands=2
)

with vision.HandLandmarker.create_from_options(options) as detector:
    for name, img in [("Original", frame), ("Resized", resized)]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
        result = detector.detect(mp_image)
        print(f"{name}: {len(result.hand_landmarks)} hand(s) detected")

===========================================================================
CELL INDEX: 71
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
from pathlib import Path
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)

manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)

row = df[
    df["video_path"].str.contains(
        "People/61. Father/MVI_5082.MOV", regex=False
    )
].iloc[0]

video_path = dataset_root / row["video_path"]

cap = cv2.VideoCapture(str(video_path))
cap.set(cv2.CAP_PROP_POS_FRAMES, 16)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read frame 16.")

resized = cv2.resize(
    frame, None, fx=1.5, fy=1.5,
    interpolation=cv2.INTER_CUBIC
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as detector:
    for name, img in [("Original", frame), ("Resized", resized)]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = detector.detect(mp_image)

        print(
            f"{name}: {len(result.hand_landmarks)} hand(s) detected"
        )

===========================================================================
CELL INDEX: 72
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

video_path = (
    "/content/drive/MyDrive/ISL_Sign_Recognition/"
    "dataset/extracted/People/61. Father/MVI_5082.MOV"
)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 16)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read frame 16.")

h, w = frame.shape[:2]

# Crop the central 60% of the image.
x1, x2 = int(w * 0.2), int(w * 0.8)
y1, y2 = int(h * 0.1), int(h * 0.9)
cropped = frame[y1:y2, x1:x2]

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
)

with HandLandmarker.create_from_options(options) as detector:
    for name, img in [
        ("Original", frame),
        ("Central crop", cropped),
    ]:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = detector.detect(mp_image)
        print(f"{name}: {len(result.hand_landmarks)} hand(s) detected")

print("Original dimensions:", frame.shape[1], "x", frame.shape[0])
print("Crop dimensions:", cropped.shape[1], "x", cropped.shape[0])

===========================================================================
CELL INDEX: 73
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

video_path = (
    "/content/drive/MyDrive/ISL_Sign_Recognition/"
    "dataset/extracted/People/61. Father/MVI_5082.MOV"
)

model_path = "/tmp/hand_landmarker.task"

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
indices = np.linspace(0, total_frames - 1, 5, dtype=int)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
)

results = []

with HandLandmarker.create_from_options(options) as detector:
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()

        if not ok:
            print(f"Frame {idx}: could not read")
            continue

        h, w = frame.shape[:2]

        # Crop the central 60% horizontally and 80% vertically.
        cropped = frame[
            int(h * 0.1):int(h * 0.9),
            int(w * 0.2):int(w * 0.8)
        ]

        counts = {}

        for name, img in [
            ("original", frame),
            ("crop", cropped)
        ]:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = detector.detect(mp_image)
            counts[name] = len(result.hand_landmarks)

        results.append((int(idx), counts["original"], counts["crop"]))

cap.release()

print("Frame | Original hands | Cropped hands")
print("-" * 42)

for idx, original, crop in results:
    print(f"{idx:5d} | {original:14d} | {crop:13d}")

print("\nFrames with at least one hand detected:")
print("Original:", sum(o > 0 for _, o, c in results), "/", len(results))
print("Cropped:", sum(c > 0 for _, o, c in results), "/", len(results))

===========================================================================
CELL INDEX: 74
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from pathlib import Path
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)

manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)
sample = df.sample(n=10, random_state=123)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
)

rows = []

with HandLandmarker.create_from_options(options) as detector:
    for _, row in sample.iterrows():
        video_path = dataset_root / row["video_path"]
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(
            0, total_frames - 1, min(5, total_frames), dtype=int
        )

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()

            if not ok:
                continue

            h, w = frame.shape[:2]
            cropped = frame[
                int(h * 0.1):int(h * 0.9),
                int(w * 0.2):int(w * 0.8)
            ]

            counts = {}

            for name, img in [
                ("original", frame),
                ("crop", cropped)
            ]:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
                result = detector.detect(mp_image)
                counts[name] = len(result.hand_landmarks)

            rows.append({
                "video_path": row["video_path"],
                "frame": int(idx),
                "original_hands": counts["original"],
                "cropped_hands": counts["crop"],
                "original_detected": counts["original"] > 0,
                "cropped_detected": counts["crop"] > 0,
            })

        cap.release()

results = pd.DataFrame(rows)

print("Videos tested:", results["video_path"].nunique())
print("Frames compared:", len(results))

print("\n=== DETECTION RATES ===")
print(
    "Original:",
    round(100 * results["original_detected"].mean(), 1), "%"
)
print(
    "Cropped:",
    round(100 * results["cropped_detected"].mean(), 1), "%"
)

print("\n=== VIDEO-LEVEL RESULTS ===")
print(
    results.groupby("video_path")[
        ["original_detected", "cropped_detected"]
    ].mean().mul(100).round(1).to_string()
)

===========================================================================
CELL INDEX: 75
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

from pathlib import Path
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

dataset_root = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
)

manifest_path = Path(
    "/content/drive/MyDrive/ISL_Sign_Recognition/audit/"
    "full_include_working_manifest.csv"
)

df = pd.read_csv(manifest_path)
sample = df.sample(n=10, random_state=123)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2,
)

records = []

with HandLandmarker.create_from_options(options) as detector:
    for _, row in sample.iterrows():
        cap = cv2.VideoCapture(
            str(dataset_root / row["video_path"])
        )
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(
            0, total_frames - 1, min(5, total_frames), dtype=int
        )

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()

            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = detector.detect(image)
            count = len(result.hand_landmarks)

            records.append({
                "video_path": row["video_path"],
                "hands_detected": count
            })

        cap.release()

results = pd.DataFrame(records)

print("Frames analyzed:", len(results))
print("\n=== FRAME-LEVEL DETECTION ===")
print(
    "No hands:",
    round(100 * (results["hands_detected"] == 0).mean(), 1), "%"
)
print(
    "One hand:",
    round(100 * (results["hands_detected"] == 1).mean(), 1), "%"
)
print(
    "Two hands:",
    round(100 * (results["hands_detected"] == 2).mean(), 1), "%"
)

print("\n=== VIDEO-LEVEL DETECTION RATE ===")
print(
    results.assign(
        detected=results["hands_detected"] > 0
    ).groupby("video_path")["detected"].mean().mul(100).round(1).to_string()
)

===========================================================================
CELL INDEX: 76
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

video_path = (
    "/content/drive/MyDrive/ISL_Sign_Recognition/"
    "dataset/extracted/Clothes/38. Dress/MVI_3693.MOV"
)

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, 29)
ok, frame = cap.read()
cap.release()

if not ok:
    raise RuntimeError("Could not read the selected video frame.")

rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as detector:
    result = detector.detect(image)

print("Hands detected:", len(result.hand_landmarks))

for i, hand in enumerate(result.hand_landmarks):
    coords = np.array([
        [point.x, point.y, point.z]
        for point in hand
    ])

    print(f"\nHand {i + 1}")
    print("Landmark array shape:", coords.shape)
    print("X range:", round(coords[:, 0].min(), 4),
          "to", round(coords[:, 0].max(), 4))
    print("Y range:", round(coords[:, 1].min(), 4),
          "to", round(coords[:, 1].max(), 4))
    print("Z range:", round(coords[:, 2].min(), 4),
          "to", round(coords[:, 2].max(), 4))

===========================================================================
CELL INDEX: 79
MATCHED KEYWORDS: HandLandmarker, handedness
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "pilot")
MODEL_PATH = "/tmp/hand_landmarker.task"

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(MODEL_PATH), (
    "MediaPipe model missing. Download hand_landmarker.task first."
)
assert len(pilot) == 10, "Expected exactly 10 pilot videos."

# Shape: videos x frames x hands x landmarks x coordinates
N_FRAMES = 16
N_HANDS = 2
N_LANDMARKS = 21
N_COORDS = 3

all_features = []
all_detection_flags = []
results_rows = []

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=N_HANDS
)

with HandLandmarker.create_from_options(options) as landmarker:
    for video_index, row in pilot.iterrows():
        video_path = os.path.join(VIDEO_ROOT, row["video_path"])

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or total_frames < 1:
            cap.release()
            raise RuntimeError(f"Could not read video: {video_path}")

        # 16 evenly spaced, chronological frame indices
        frame_indices = np.linspace(
            0, total_frames - 1, N_FRAMES
        ).round().astype(int)

        sequence = np.zeros(
            (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
            dtype=np.float32
        )
        flags = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)

        for frame_pos, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()

            if not ok:
                cap.release()
                raise RuntimeError(
                    f"Failed to read frame {frame_idx} from {video_path}"
                )

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb
            )
            detection = landmarker.detect(mp_image)

            # Assign slots by handedness label rather than detection order.
            # These are MediaPipe's image-based handedness labels.
            for hand_idx, landmarks in enumerate(
                detection.hand_landmarks
            ):
                handedness = detection.handedness[hand_idx]
                hand_label = handedness[0].category_name.lower()

                if hand_label == "left":
                    slot = 0
                elif hand_label == "right":
                    slot = 1
                else:
                    continue

                sequence[frame_pos, slot] = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32
                )
                flags[frame_pos, slot] = 1

        cap.release()

        all_features.append(sequence)
        all_detection_flags.append(flags)

        results_rows.append({
            "video_path": row["video_path"],
            "category": row["category"],
            "label": row["label"],
            "split": row["split"],
            "total_video_frames": total_frames,
            "sampled_frames": N_FRAMES,
            "left_hand_detected_frames": int(flags[:, 0].sum()),
            "right_hand_detected_frames": int(flags[:, 1].sum()),
            "any_hand_detected_frames": int(flags.any(axis=1).sum())
        })

        print(
            f"{video_index + 1}/10 processed: "
            f"{row['video_path']}"
        )

features = np.stack(all_features)
detection_flags = np.stack(all_detection_flags)
results_df = pd.DataFrame(results_rows)

# Save pilot outputs separately from the source videos.
np.save(os.path.join(OUTPUT_DIR, "pilot_landmarks.npy"), features)
np.save(os.path.join(OUTPUT_DIR, "pilot_detection_flags.npy"),
        detection_flags)
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "pilot_results.csv"), index=False
)
pilot.to_csv(
    os.path.join(OUTPUT_DIR, "pilot_manifest.csv"), index=False
)

print("\nPILOT COMPLETE")
print("Landmarks shape:", features.shape)
print("Detection flags shape:", detection_flags.shape)
print("Expected landmarks shape: (10, 16, 2, 21, 3)")
print("Expected flags shape: (10, 16, 2)")
display(results_df)


===========================================================================
CELL INDEX: 81
MATCHED KEYWORDS: HandLandmarker, handedness
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "pilot")
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

pilot_check = pd.read_csv(
    os.path.join(OUTPUT_DIR, "pilot_manifest.csv")
)
saved_features = np.load(
    os.path.join(OUTPUT_DIR, "pilot_landmarks.npy")
)
saved_flags = np.load(
    os.path.join(OUTPUT_DIR, "pilot_detection_flags.npy")
)

# Recheck handedness assignments on the first selected video.
video_path = os.path.join(VIDEO_ROOT, pilot_check.iloc[0]["video_path"])
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = np.linspace(0, total_frames - 1, 16).round().astype(int)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

observed_labels = []

with HandLandmarker.create_from_options(options) as landmarker:
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()

        if not ok:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )

        observed_labels.append([
            hand[0].category_name
            for hand in result.handedness
        ])

cap.release()

print("First video:", pilot_check.iloc[0]["video_path"])
print("MediaPipe handedness labels for sampled frames:")
for i, labels in enumerate(observed_labels):
    print(f"Frame {i + 1:02d}: {labels}")

print("\nCoordinate range across detected slots:")
detected = saved_flags.astype(bool)
if detected.any():
    coords = saved_features[detected]
    print("Minimum:", coords.min(axis=(0, 1, 2)))
    print("Maximum:", coords.max(axis=(0, 1, 2)))
else:
    print("No detected hands to inspect.")

print("\nPilot split counts:")
print(pilot_check["split"].value_counts())


===========================================================================
CELL INDEX: 82
MATCHED KEYWORDS: HandLandmarker, handedness
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
INPUT_DIR = os.path.join(BASE_DIR, "features", "pilot")
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "pilot_v2")
MODEL_PATH = "/tmp/hand_landmarker.task"

os.makedirs(OUTPUT_DIR, exist_ok=True)

pilot = pd.read_csv(
    os.path.join(INPUT_DIR, "pilot_manifest.csv")
)

assert len(pilot) == 10
assert os.path.exists(MODEL_PATH), (
    "MediaPipe model missing. Download hand_landmarker.task first."
)

N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS = 16, 2, 21, 3

all_features = []
all_flags = []
all_results = []

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=N_HANDS
)

with HandLandmarker.create_from_options(options) as landmarker:
    for i, row in pilot.iterrows():
        path = os.path.join(VIDEO_ROOT, row["video_path"])
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            cap.release()
            raise RuntimeError(f"No frames found: {path}")

        indices = np.linspace(
            0, total_frames - 1, N_FRAMES
        ).round().astype(int)

        sequence = np.zeros(
            (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
            dtype=np.float32
        )
        flags = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)

        for t, frame_idx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()

            if not ok:
                cap.release()
                raise RuntimeError(
                    f"Could not read frame {frame_idx}: {path}"
                )

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            result = landmarker.detect(image)

            # Preserve both detections without relying on handedness labels.
            # Slot 0 = first detection; slot 1 = second detection.
            for slot, landmarks in enumerate(result.hand_landmarks[:N_HANDS]):
                sequence[t, slot] = np.asarray(
                    [[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32
                )
                flags[t, slot] = 1

        cap.release()

        all_features.append(sequence)
        all_flags.append(flags)

        all_results.append({
            "video_path": row["video_path"],
            "category": row["category"],
            "label": row["label"],
            "split": row["split"],
            "total_video_frames": total_frames,
            "sampled_frames": N_FRAMES,
            "first_slot_detections": int(flags[:, 0].sum()),
            "second_slot_detections": int(flags[:, 1].sum()),
            "frames_with_any_hand": int(flags.any(axis=1).sum()),
            "frames_with_two_hands": int(flags.all(axis=1).sum())
        })

        print(f"{i + 1}/10 processed: {row['video_path']}")

features_v2 = np.stack(all_features)
flags_v2 = np.stack(all_flags)
results_v2 = pd.DataFrame(all_results)

np.save(os.path.join(OUTPUT_DIR, "pilot_landmarks.npy"), features_v2)
np.save(os.path.join(OUTPUT_DIR, "pilot_detection_flags.npy"), flags_v2)
results_v2.to_csv(
    os.path.join(OUTPUT_DIR, "pilot_results.csv"), index=False
)
pilot.to_csv(
    os.path.join(OUTPUT_DIR, "pilot_manifest.csv"), index=False
)

# Validate saved outputs.
reloaded_features = np.load(
    os.path.join(OUTPUT_DIR, "pilot_landmarks.npy")
)
reloaded_flags = np.load(
    os.path.join(OUTPUT_DIR, "pilot_detection_flags.npy")
)

assert reloaded_features.shape == (10, 16, 2, 21, 3)
assert reloaded_flags.shape == (10, 16, 2)
assert np.isfinite(reloaded_features).all()
assert np.isin(reloaded_flags, [0, 1]).all()
assert np.all(reloaded_features[reloaded_flags == 0] == 0)

print("\nPILOT V2 COMPLETE")
print("Landmarks:", reloaded_features.shape)
print("Detection flags:", reloaded_flags.shape)
display(results_v2)


===========================================================================
CELL INDEX: 83
MATCHED KEYWORDS: HandLandmarker
===========================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)
from mediapipe.tasks.python.vision.drawing_utils import (
    draw_landmarks,
    DrawingSpec
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

inspect_videos = [
    "Clothes/38. Dress/MVI_8601.MP4",
    "Society/6. Marriage/MVI_4817.MOV"
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    for relative_path in inspect_videos:
        path = os.path.join(VIDEO_ROOT, relative_path)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 16).round().astype(int)

        # Show 4 representative frames: beginning, early-middle,
        # late-middle and end.
        selected = [indices[i] for i in [0, 5, 10, 15]]
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        for ax, frame_idx in zip(axes, selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()

            if not ok:
                ax.set_title(f"Frame {frame_idx}: read failed")
                ax.axis("off")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            annotated = rgb.copy()
            for hand in result.hand_landmarks:
                draw_landmarks(
                    annotated,
                    hand,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    DrawingSpec(color=(255, 0, 0), thickness=2)
                )

            ax.imshow(annotated)
            ax.set_title(
                f"Frame {frame_idx}\n"
                f"Detected hands: {len(result.hand_landmarks)}"
            )
            ax.axis("off")

        cap.release()
        fig.suptitle(relative_path, fontsize=12)
        plt.tight_layout()
        plt.show()


===========================================================================
CELL INDEX: 84
MATCHED KEYWORDS: HandLandmarker
===========================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

# Connections between the 21 hand landmarks.
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(0,17),(17,18),(18,19),
    (19,20)
]

inspect_videos = [
    "Clothes/38. Dress/MVI_8601.MP4",
    "Society/6. Marriage/MVI_4817.MOV"
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    for relative_path in inspect_videos:
        path = os.path.join(VIDEO_ROOT, relative_path)
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 16).round().astype(int)
        selected = [indices[i] for i in [0, 5, 10, 15]]

        fig, axes = plt.subplots(1, 4, figsize=(16, 5))

        for ax, frame_idx in zip(axes, selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame = cap.read()

            if not ok:
                ax.set_title(f"Frame {frame_idx}: read failed")
                ax.axis("off")
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            h, w = rgb.shape[:2]

            for hand in result.hand_landmarks:
                points = [
                    (int(lm.x * w), int(lm.y * h))
                    for lm in hand
                ]

                for a, b in HAND_CONNECTIONS:
                    cv2.line(rgb, points[a], points[b], (0, 255, 0), 3)

                for x, y in points:
                    cv2.circle(rgb, (x, y), 4, (255, 0, 0), -1)

            ax.imshow(rgb)
            ax.set_title(
                f"Frame {frame_idx}\n"
                f"Detected hands: {len(result.hand_landmarks)}"
            )
            ax.axis("off")

        cap.release()
        fig.suptitle(relative_path, fontsize=12)
        plt.tight_layout()
        plt.show()


===========================================================================
CELL INDEX: 85
MATCHED KEYWORDS: HandLandmarker
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

video_path = os.path.join(
    VIDEO_ROOT, "Clothes/38. Dress/MVI_8601.MP4"
)

assert os.path.exists(video_path), f"Video not found: {video_path}"
assert os.path.exists(MODEL_PATH), "MediaPipe model file is missing."

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Same representative frames as our previous inspection.
frame_indices = np.linspace(0, total_frames - 1, 16).round().astype(int)
selected_indices = [int(frame_indices[i]) for i in [0, 5, 10, 15]]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

rows = []

with HandLandmarker.create_from_options(options) as landmarker:
    for frame_idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()

        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx}")

        original_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Method A: original image.
        result_original = landmarker.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=original_rgb
            )
        )

        # Method B: enlarge the entire frame by 1.5x.
        enlarged = cv2.resize(
            original_rgb,
            None,
            fx=1.5,
            fy=1.5,
            interpolation=cv2.INTER_CUBIC
        )

        result_enlarged = landmarker.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=enlarged
            )
        )

        rows.append({
            "frame": frame_idx,
            "original_hands": len(result_original.hand_landmarks),
            "enlarged_hands": len(result_enlarged.hand_landmarks)
        })

cap.release()

comparison = pd.DataFrame(rows)

print("Detection comparison:")
display(comparison)

print("Original frames with at least one hand:",
      int((comparison["original_hands"] > 0).sum()), "/", len(comparison))
print("Enlarged frames with at least one hand:",
      int((comparison["enlarged_hands"] > 0).sum()), "/", len(comparison))



===========================================================================
CELL INDEX: 86
MATCHED KEYWORDS: HandLandmarker
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

manifest = pd.read_csv(
    os.path.join(BASE_DIR, "audit", "full_include_working_manifest.csv")
)

# Reproducible sample across labels and dataset splits.
sample = manifest.sample(
    n=min(50, len(manifest)), random_state=2026
).reset_index(drop=True)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

records = []

with HandLandmarker.create_from_options(options) as landmarker:
    for i, row in sample.iterrows():
        path = os.path.join(VIDEO_ROOT, row["video_path"])
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            records.append({
                "video_path": row["video_path"],
                "split": row["split"],
                "status": "OPEN_FAILED",
                "frames_checked": 0,
                "frames_with_hands": 0
            })
            continue

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.unique(
            np.linspace(0, max(total - 1, 0), 8).round().astype(int)
        )

        detected_frames = 0
        checked_frames = 0

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()

            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )

            checked_frames += 1
            if len(result.hand_landmarks) > 0:
                detected_frames += 1

        cap.release()

        records.append({
            "video_path": row["video_path"],
            "split": row["split"],
            "status": "OK",
            "frames_checked": checked_frames,
            "frames_with_hands": detected_frames
        })

        print(f"{i + 1}/50 checked")

audit = pd.DataFrame(records)
audit_path = os.path.join(BASE_DIR, "features", "pilot_v2", "detection_screening_50.csv")
os.makedirs(os.path.dirname(audit_path), exist_ok=True)
audit.to_csv(audit_path, index=False)

valid = audit[audit["status"] == "OK"]

print("\nSCREENING SUMMARY")
print("Videos sampled:", len(sample))
print("Videos opened successfully:", len(valid))
print("Videos that failed to open:", (audit["status"] != "OK").sum())

if len(valid):
    total_checked = valid["frames_checked"].sum()
    total_detected = valid["frames_with_hands"].sum()
    print("Frames with detected hands:", total_detected, "/", total_checked)
    print("Detection rate:", round(100 * total_detected / total_checked, 2), "%")

print("\nResults saved to:", audit_path)
display(audit.head(10))


===========================================================================
CELL INDEX: 87
MATCHED KEYWORDS: HandLandmarker, handedness
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "pilot_100")

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(MANIFEST_PATH), "Working manifest not found."
assert os.path.exists(MODEL_PATH), (
    "MediaPipe model missing. Download hand_landmarker.task first."
)

manifest = pd.read_csv(MANIFEST_PATH)

# Reproducible 100-video sample, retaining original split assignments.
sample = (
    manifest.sample(n=100, random_state=2026)
    .reset_index(drop=True)
)

# Save the sample manifest before extraction.
sample.to_csv(
    os.path.join(OUTPUT_DIR, "pilot_manifest.csv"), index=False
)

CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.npz")
RESULTS_PATH = os.path.join(OUTPUT_DIR, "pilot_results.csv")

# Resume only from a valid checkpoint belonging to this exact sample.
completed = {}
if os.path.exists(CHECKPOINT_PATH):
    saved = np.load(CHECKPOINT_PATH, allow_pickle=False)
    if (
        "sample_paths" in saved
        and saved["sample_paths"].tolist() == sample["video_path"].tolist()
    ):
        for i in saved["completed_indices"].tolist():
            completed[int(i)] = (
                saved["features"][int(i)],
                saved["flags"][int(i)]
            )
        print("Resuming:", len(completed), "videos already completed.")
    else:
        print("Existing checkpoint belongs to another sample; starting fresh.")

N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS = 16, 2, 21, 3
features = np.zeros(
    (100, N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
    dtype=np.float32
)
flags = np.zeros((100, N_FRAMES, N_HANDS), dtype=np.uint8)

for i, (seq, det) in completed.items():
    features[i] = seq
    flags[i] = det

def save_checkpoint():
    completed_indices = np.array(sorted(completed), dtype=np.int32)
    np.savez_compressed(
        CHECKPOINT_PATH,
        sample_paths=sample["video_path"].to_numpy(dtype=str),
        completed_indices=completed_indices,
        features=features,
        flags=flags
    )

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

try:
    with HandLandmarker.create_from_options(options) as landmarker:
        for i, row in sample.iterrows():
            if i in completed:
                continue

            path = os.path.join(VIDEO_ROOT, row["video_path"])
            cap = cv2.VideoCapture(path)

            if not cap.isOpened():
                cap.release()
                print("SKIPPED: could not open", row["video_path"])
                continue

            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 1:
                cap.release()
                print("SKIPPED: no frames", row["video_path"])
                continue

            indices = np.linspace(
                0, total - 1, N_FRAMES
            ).round().astype(int)

            seq = np.zeros(
                (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                dtype=np.float32
            )
            det = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)

            valid_video = True

            for t, idx in enumerate(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, frame = cap.read()

                if not ok:
                    valid_video = False
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )
                )

                # Detection-order slots prevent duplicate-handedness overwrite.
                for slot, hand in enumerate(result.hand_landmarks[:N_HANDS]):
                    seq[t, slot] = np.asarray(
                        [[lm.x, lm.y, lm.z] for lm in hand],
                        dtype=np.float32
                    )
                    det[t, slot] = 1

            cap.release()

            if not valid_video:
                print("SKIPPED: frame read failed", row["video_path"])
                continue

            features[i] = seq
            flags[i] = det
            completed[i] = (seq, det)

            if len(completed) % 10 == 0:
                save_checkpoint()
                print(f"Checkpoint saved: {len(completed)}/100")

except Exception:
    save_checkpoint()
    raise

save_checkpoint()

# Create a result table for successfully processed videos.
result_rows = []
for i in sorted(completed):
    row = sample.iloc[i]
    result_rows.append({
        "sample_index": i,
        "video_path": row["video_path"],
        "category": row["category"],
        "label": row["label"],
        "split": row["split"],
        "frames_with_any_hand": int(flags[i].any(axis=1).sum()),
        "frames_with_two_hands": int(flags[i].all(axis=1).sum())
    })

results = pd.DataFrame(result_rows)
results.to_csv(RESULTS_PATH, index=False)

print("\nEXTRACTION SUMMARY")
print("Videos successfully processed:", len(completed), "/ 100")
print("Feature array shape:", features.shape)
print("Detection flag shape:", flags.shape)
print("Frames with at least one hand:",
      int(flags[list(completed)].any(axis=2).sum()),
      "/", len(completed) * N_FRAMES)
print("Results saved:", RESULTS_PATH)
display(results.head(10))


===========================================================================
CELL INDEX: 89
MATCHED KEYWORDS: HandLandmarker, handedness
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

manifest = pd.read_csv(
    os.path.join(BASE_DIR, "features", "pilot_100", "pilot_manifest.csv")
)

# Check 20 videos and 8 evenly spaced frames per video.
sample = manifest.sample(n=20, random_state=7).reset_index(drop=True)
duplicate_frames = 0
two_hand_frames = 0
total_frames = 0

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    for _, row in sample.iterrows():
        path = os.path.join(VIDEO_ROOT, row["video_path"])
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            cap.release()
            continue

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.unique(np.linspace(0, max(n - 1, 0), 8).round().astype(int))

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            )
            total_frames += 1

            if len(result.hand_landmarks) == 2:
                two_hand_frames += 1
                labels = [
                    h[0].category_name.lower()
                    for h in result.handedness
                ]
                if len(set(labels)) < 2:
                    duplicate_frames += 1

        cap.release()

print("Frames checked:", total_frames)
print("Frames with two detected hands:", two_hand_frames)
print("Two-hand frames with duplicate handedness labels:", duplicate_frames)
if two_hand_frames:
    print("Duplicate-label rate among two-hand frames:",
          round(100 * duplicate_frames / two_hand_frames, 2), "%")


===========================================================================
CELL INDEX: 90
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================

import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(MANIFEST_PATH), "Working manifest not found."
assert os.path.exists(MODEL_PATH), "MediaPipe model missing."

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)
assert len(manifest) == 4257
assert manifest["video_path"].is_unique

N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS = 16, 2, 21, 3
BATCH_SIZE = 50

# A status CSV is the resume index. Each completed video is stored
# in its own batch file, so a runtime restart does not lose prior batches.
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")
status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

if os.path.exists(STATUS_PATH):
    status_df = pd.read_csv(STATUS_PATH)
else:
    status_df = pd.DataFrame(columns=status_columns)

completed = set(
    status_df.loc[status_df["status"] == "success", "index"]
    .astype(int).tolist()
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

def save_status(rows):
    global status_df
    if rows:
        status_df = pd.concat(
            [status_df, pd.DataFrame(rows)],
            ignore_index=True
        )
        status_df = status_df.drop_duplicates(
            subset=["index"], keep="last"
        )
        status_df.to_csv(STATUS_PATH, index=False)

remaining = [i for i in range(len(manifest)) if i not in completed]
print("Total videos:", len(manifest))
print("Already completed:", len(completed))
print("Remaining:", len(remaining))

with HandLandmarker.create_from_options(options) as landmarker:
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch_indices = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_rows = []
        batch_features = []
        batch_flags = []
        batch_meta = []

        for idx in batch_indices:
            row = manifest.iloc[idx]
            path = os.path.join(VIDEO_ROOT, row["video_path"])

            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError("Could not open video")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    cap.release()
                    raise RuntimeError("Video contains no readable frames")

                frame_indices = np.linspace(
                    0, total_frames - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )
                flags = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)
                ambiguous_frames = 0

                for t, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Failed to read frame {frame_idx}")

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]
                    labels = [
                        h[0].category_name.lower()
                        if h and result.handedness[j]
                        else "unknown"
                        for j, h in enumerate(hands)
                    ]

                    # If two labels are distinct, put hands into left/right
                    # slots. Otherwise use detection order for this frame
                    # and record the ambiguity.
                    if len(hands) == 2:
                        distinct = (
                            len(labels) == 2
                            and set(labels) == {"left", "right"}
                        )
                        if not distinct:
                            ambiguous_frames += 1
                            slots = [(0, hands[0]), (1, hands[1])]
                        else:
                            slots = [
                                (0 if labels[j] == "left" else 1, hands[j])
                                for j in range(2)
                            ]
                    elif len(hands) == 1:
                        label = labels[0] if labels else "unknown"
                        slot = (
                            0 if label == "left"
                            else 1 if label == "right"
                            else 0
                        )
                        slots = [(slot, hands[0])]
                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in hand],
                            dtype=np.float32
                        )
                        flags[t, slot] = 1

                cap.release()

                batch_features.append(sequence)
                batch_flags.append(flags)
                batch_meta.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total_frames,
                    "frames_with_any_hand": int(flags.any(axis=1).sum()),
                    "frames_with_two_hands": int(flags.all(axis=1).sum()),
                    "ambiguous_handedness_frames": ambiguous_frames
                })

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "success",
                    "batch_file": "",
                    "frames_with_any_hand": int(flags.any(axis=1).sum()),
                    "frames_with_two_hands": int(flags.all(axis=1).sum()),
                    "ambiguous_handedness_frames": ambiguous_frames,
                    "error": ""
                })

            except Exception as e:
                try:
                    cap.release()
                except Exception:
                    pass

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "failed",
                    "batch_file": "",
                    "frames_with_any_hand": 0,
                    "frames_with_two_hands": 0,
                    "ambiguous_handedness_frames": 0,
                    "error": str(e)[:300]
                })

        if batch_features:
            batch_number = batch_start // BATCH_SIZE
            batch_file = f"batch_{batch_number:04d}.npz"
            batch_path = os.path.join(OUTPUT_DIR, batch_file)

            np.savez_compressed(
                batch_path,
                indices=np.array(
                    [m["index"] for m in batch_meta], dtype=np.int32
                ),
                features=np.stack(batch_features),
                flags=np.stack(batch_flags),
                metadata=pd.DataFrame(batch_meta).to_json(
                    orient="records"
                )
            )

            # Link successful status rows to the saved batch.
            for record in batch_rows:
                if record["status"] == "success":
                    record["batch_file"] = batch_file

        save_status(batch_rows)

        print(
            f"Processed {min(batch_start + BATCH_SIZE, len(remaining))}"
            f"/{len(remaining)} remaining videos; "
            f"successful this batch: "
            f"{sum(r['status'] == 'success' for r in batch_rows)}"
        )

print("\nEXTRACTION RUN FINISHED")
print("Successful videos:", int((status_df["status"] == "success").sum()))
print("Failed videos:", int((status_df["status"] == "failed").sum()))
print("Status file:", STATUS_PATH)
print("Batch directory:", OUTPUT_DIR)


===========================================================================
CELL INDEX: 95
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)
assert len(manifest) == 4257
assert manifest["video_path"].is_unique
assert os.path.exists(MODEL_PATH), "MediaPipe model missing."

N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS = 16, 2, 21, 3
BATCH_SIZE = 50

status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

if os.path.exists(STATUS_PATH):
    status_df = pd.read_csv(STATUS_PATH)
else:
    status_df = pd.DataFrame(columns=status_columns)

# Only successful records are skipped.
completed = set(
    status_df.loc[status_df["status"] == "success", "index"]
    .astype(int).tolist()
)

remaining = [i for i in range(len(manifest)) if i not in completed]

print("Total videos:", len(manifest))
print("Already successful:", len(completed))
print("Remaining to process:", len(remaining))

def update_status(rows):
    global status_df
    if rows:
        status_df = pd.concat(
            [status_df, pd.DataFrame(rows)], ignore_index=True
        )
        status_df = status_df.drop_duplicates(
            subset=["index"], keep="last"
        )
        status_df.to_csv(STATUS_PATH, index=False)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    for start in range(0, len(remaining), BATCH_SIZE):
        batch_indices = remaining[start:start + BATCH_SIZE]
        batch_features, batch_flags, batch_meta, batch_rows = [], [], [], []

        for idx in batch_indices:
            row = manifest.iloc[idx]
            path = os.path.join(VIDEO_ROOT, row["video_path"])
            cap = None

            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video")

                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total < 1:
                    raise RuntimeError("Video contains no frames")

                frame_indices = np.linspace(
                    0, total - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )
                flags = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)
                ambiguous = 0

                for t, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Failed to read frame {frame_idx}")

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]

                    # Correct API: handedness labels come from result.handedness.
                    labels = []
                    for j in range(len(hands)):
                        categories = (
                            result.handedness[j]
                            if j < len(result.handedness) else []
                        )
                        labels.append(
                            categories[0].category_name.lower()
                            if categories else "unknown"
                        )

                    if len(hands) == 2:
                        if set(labels) == {"left", "right"}:
                            slots = [
                                (0 if labels[j] == "left" else 1, hands[j])
                                for j in range(2)
                            ]
                        else:
                            ambiguous += 1
                            slots = [(0, hands[0]), (1, hands[1])]
                    elif len(hands) == 1:
                        label = labels[0] if labels else "unknown"
                        slot = (
                            0 if label == "left"
                            else 1 if label == "right"
                            else 0
                        )
                        slots = [(slot, hands[0])]
                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in hand],
                            dtype=np.float32
                        )
                        flags[t, slot] = 1

                batch_features.append(sequence)
                batch_flags.append(flags)
                batch_meta.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total,
                    "frames_with_any_hand": int(flags.any(axis=1).sum()),
                    "frames_with_two_hands": int(flags.all(axis=1).sum()),
                    "ambiguous_handedness_frames": ambiguous
                })

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "success",
                    "batch_file": "",
                    "frames_with_any_hand": int(flags.any(axis=1).sum()),
                    "frames_with_two_hands": int(flags.all(axis=1).sum()),
                    "ambiguous_handedness_frames": ambiguous,
                    "error": ""
                })

            except Exception as e:
                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "failed",
                    "batch_file": "",
                    "frames_with_any_hand": 0,
                    "frames_with_two_hands": 0,
                    "ambiguous_handedness_frames": 0,
                    "error": str(e)[:300]
                })
            finally:
                if cap is not None:
                    cap.release()

        if batch_features:
            # Unique batch filename prevents overwriting earlier saved batches.
            batch_number = (start // BATCH_SIZE)
            batch_file = f"resume_{len(completed):04d}_{batch_number:04d}.npz"
            np.savez_compressed(
                os.path.join(OUTPUT_DIR, batch_file),
                indices=np.array(
                    [m["index"] for m in batch_meta], dtype=np.int32
                ),
                features=np.stack(batch_features),
                flags=np.stack(batch_flags),
                metadata=pd.DataFrame(batch_meta).to_json(orient="records")
            )

            for record in batch_rows:
                if record["status"] == "success":
                    record["batch_file"] = batch_file

        update_status(batch_rows)
        completed = set(
            status_df.loc[status_df["status"] == "success", "index"]
            .astype(int).tolist()
        )

        print(
            f"Processed batch: {len(batch_indices)} videos | "
            f"total successful: {len(completed)} | "
            f"failed records: {(status_df['status'] == 'failed').sum()}"
        )

print("\nEXTRACTION RUN FINISHED")
print("Successful:", int((status_df["status"] == "success").sum()))
print("Failed:", int((status_df["status"] == "failed").sum()))
print("Status:", STATUS_PATH)


===========================================================================
CELL INDEX: 97
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import mediapipe as mp
print("MediaPipe version:", mp.__version__)

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

print("MediaPipe Tasks API imported successfully!")

===========================================================================
CELL INDEX: 102
MATCHED KEYWORDS: HandLandmarker
===========================================================================
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    print("MediaPipe hand detector initialized successfully!")

===========================================================================
CELL INDEX: 103
MATCHED KEYWORDS: HandLandmarker
===========================================================================
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    print("MediaPipe hand detector initialized successfully!")

===========================================================================
CELL INDEX: 104
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")

N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS = 16, 2, 21, 3
BATCH_SIZE = 50

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)
assert len(manifest) == 4257
assert manifest["video_path"].is_unique
assert os.path.exists(MODEL_PATH)

status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

if os.path.exists(STATUS_PATH):
    status_df = pd.read_csv(STATUS_PATH)
else:
    status_df = pd.DataFrame(columns=status_columns)

# Preserve successful entries only when their batch really exists
# and contains the corresponding index.
valid_success = set()
for _, record in status_df[status_df["status"] == "success"].iterrows():
    batch_file = str(record.get("batch_file", ""))
    batch_path = os.path.join(OUTPUT_DIR, batch_file)

    try:
        if not batch_file or not os.path.isfile(batch_path):
            continue
        with np.load(batch_path, allow_pickle=False) as batch:
            if int(record["index"]) in batch["indices"].tolist():
                valid_success.add(int(record["index"]))
    except Exception:
        continue

# Keep valid successes; reset failed/invalid entries for retry.
status_df = status_df[
    status_df["index"].astype(int).isin(valid_success)
].copy()
status_df.to_csv(STATUS_PATH, index=False)

remaining = [
    i for i in range(len(manifest)) if i not in valid_success
]

print("Total videos:", len(manifest))
print("Valid successful videos preserved:", len(valid_success))
print("Videos remaining:", len(remaining))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch_indices = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_features, batch_flags, batch_meta = [], [], []
        batch_rows = []

        for idx in batch_indices:
            row = manifest.iloc[idx]
            path = os.path.join(VIDEO_ROOT, row["video_path"])
            cap = None

            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    raise RuntimeError("Video contains no frames")

                frame_indices = np.linspace(
                    0, total_frames - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )
                flags = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)
                ambiguous_frames = 0

                for t, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(f"Failed reading frame {frame_idx}")

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]

                    # Correct API: handedness comes from result.handedness,
                    # not from the landmark objects.
                    labels = []
                    for j in range(len(hands)):
                        categories = (
                            result.handedness[j]
                            if j < len(result.handedness) else []
                        )
                        label = (
                            categories[0].category_name.lower()
                            if categories else "unknown"
                        )
                        labels.append(label)

                    if len(hands) == 2:
                        distinct = set(labels) == {"left", "right"}
                        if distinct:
                            slots = [
                                (0 if labels[j] == "left" else 1, hands[j])
                                for j in range(2)
                            ]
                        else:
                            ambiguous_frames += 1
                            slots = [(0, hands[0]), (1, hands[1])]
                    elif len(hands) == 1:
                        label = labels[0] if labels else "unknown"
                        slot = 0 if label == "left" else (
                            1 if label == "right" else 0
                        )
                        slots = [(slot, hands[0])]
                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in hand],
                            dtype=np.float32
                        )
                        flags[t, slot] = 1

                batch_features.append(sequence)
                batch_flags.append(flags)
                batch_meta.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total_frames,
                    "frames_with_any_hand": int(flags.any(axis=1).sum()),
                    "frames_with_two_hands": int(flags.all(axis=1).sum()),
                    "ambiguous_handedness_frames": ambiguous_frames
                })

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "success",
                    "batch_file": "",
                    "frames_with_any_hand": int(flags.any(axis=1).sum()),
                    "frames_with_two_hands": int(flags.all(axis=1).sum()),
                    "ambiguous_handedness_frames": ambiguous_frames,
                    "error": ""
                })

            except Exception as e:
                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "failed",
                    "batch_file": "",
                    "frames_with_any_hand": 0,
                    "frames_with_two_hands": 0,
                    "ambiguous_handedness_frames": 0,
                    "error": str(e)[:300]
                })
            finally:
                if cap is not None:
                    cap.release()

        if batch_features:
            # Unique names prevent overwriting any previous batch.
            batch_number = max(
                [
                    int(f[6:10]) for f in os.listdir(OUTPUT_DIR)
                    if f.startswith("batch_")
                    and f.endswith(".npz")
                    and f[6:10].isdigit()
                ] or [-1]
            ) + 1

            batch_file = f"batch_{batch_number:04d}.npz"
            batch_path = os.path.join(OUTPUT_DIR, batch_file)

            np.savez_compressed(
                batch_path,
                indices=np.array(
                    [m["index"] for m in batch_meta], dtype=np.int32
                ),
                features=np.stack(batch_features),
                flags=np.stack(batch_flags),
                metadata=pd.DataFrame(batch_meta).to_json(orient="records")
            )

            for record in batch_rows:
                if record["status"] == "success":
                    record["batch_file"] = batch_file

        status_df = pd.concat(
            [status_df, pd.DataFrame(batch_rows)], ignore_index=True
        ).drop_duplicates(subset=["index"], keep="last")
        status_df.to_csv(STATUS_PATH, index=False)

        print(
            f"Run progress: {min(batch_start + BATCH_SIZE, len(remaining))}"
            f"/{len(remaining)} videos attempted; "
            f"{sum(r['status'] == 'success' for r in batch_rows)} succeeded "
            f"in this batch."
        )

print("\nRUN FINISHED")
print(status_df["status"].value_counts())
print("Status file:", STATUS_PATH)


===========================================================================
CELL INDEX: 110
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================
import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(MANIFEST_PATH), "Working manifest not found."
assert os.path.exists(MODEL_PATH), "MediaPipe model missing."

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)
assert len(manifest) == 4257
assert manifest["video_path"].is_unique

N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS = 16, 2, 21, 3
BATCH_SIZE = 10

status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

if os.path.exists(STATUS_PATH):
    status_df = pd.read_csv(STATUS_PATH)
else:
    status_df = pd.DataFrame(columns=status_columns)

# Only skip successful videos whose saved batch actually contains their index.
valid_success = set()
for _, record in status_df[status_df["status"] == "success"].iterrows():
    batch_file = record["batch_file"]
    if not isinstance(batch_file, str) or not batch_file:
        continue
    batch_path = os.path.join(OUTPUT_DIR, batch_file)
    if not os.path.exists(batch_path):
        continue
    try:
        with np.load(batch_path, allow_pickle=True) as data:
            if int(record["index"]) in data["indices"].astype(int):
                valid_success.add(int(record["index"]))
    except Exception:
        pass

# Remove stale or failed records; retry them.
status_df = status_df[
    status_df["index"].astype(int).isin(valid_success)
].copy()
status_df.to_csv(STATUS_PATH, index=False)

remaining = [i for i in range(len(manifest)) if i not in valid_success]

print("Total videos:", len(manifest))
print("Already completed and verified:", len(valid_success))
print("Remaining:", len(remaining))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

def save_status(rows):
    global status_df
    if rows:
        status_df = pd.concat(
            [status_df, pd.DataFrame(rows)],
            ignore_index=True
        ).drop_duplicates(subset=["index"], keep="last")
        status_df.to_csv(STATUS_PATH, index=False)

with HandLandmarker.create_from_options(options) as landmarker:
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch_indices = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_rows, batch_features, batch_flags, batch_meta = [], [], [], []
        batch_started = time.time()

        for idx in batch_indices:
            row = manifest.iloc[idx]
            path = os.path.join(VIDEO_ROOT, row["video_path"])
            cap = None

            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    raise RuntimeError("Video contains no readable frames")

                frame_indices = np.linspace(
                    0, total_frames - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )
                flags = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)
                ambiguous_frames = 0

                for t, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"Failed to read frame {frame_idx}"
                        )

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]

                    # Correct API: handedness is separate from landmarks.
                    labels = []
                    for j in range(len(hands)):
                        categories = (
                            result.handedness[j]
                            if j < len(result.handedness) else []
                        )
                        labels.append(
                            categories[0].category_name.lower()
                            if categories else "unknown"
                        )

                    if len(hands) == 2:
                        distinct = set(labels) == {"left", "right"}
                        if distinct:
                            slots = [
                                (0 if labels[j] == "left" else 1, hands[j])
                                for j in range(2)
                            ]
                        else:
                            ambiguous_frames += 1
                            slots = [(0, hands[0]), (1, hands[1])]
                    elif len(hands) == 1:
                        label = labels[0] if labels else "unknown"
                        slot = 0 if label == "left" else (
                            1 if label == "right" else 0
                        )
                        slots = [(slot, hands[0])]
                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in hand],
                            dtype=np.float32
                        )
                        flags[t, slot] = 1

                cap.release()
                cap = None

                any_hand = int(flags.any(axis=1).sum())
                two_hands = int(flags.all(axis=1).sum())

                batch_features.append(sequence)
                batch_flags.append(flags)
                batch_meta.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total_frames,
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous_frames
                })

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "success",
                    "batch_file": "",
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous_frames,
                    "error": ""
                })

            except Exception as e:
                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "failed",
                    "batch_file": "",
                    "frames_with_any_hand": 0,
                    "frames_with_two_hands": 0,
                    "ambiguous_handedness_frames": 0,
                    "error": str(e)[:300]
                })
            finally:
                if cap is not None:
                    cap.release()

        if batch_features:
            # Unique, non-overwriting batch filename.
            existing_numbers = [
                int(f[6:10]) for f in os.listdir(OUTPUT_DIR)
                if f.startswith("batch_") and f.endswith(".npz")
                and f[6:10].isdigit()
            ]
            batch_number = max(existing_numbers, default=-1) + 1
            batch_file = f"batch_{batch_number:04d}.npz"
            batch_path = os.path.join(OUTPUT_DIR, batch_file)

            np.savez_compressed(
                batch_path,
                indices=np.array(
                    [m["index"] for m in batch_meta], dtype=np.int32
                ),
                features=np.stack(batch_features),
                flags=np.stack(batch_flags),
                metadata=pd.DataFrame(batch_meta).to_json(orient="records")
            )

            for record in batch_rows:
                if record["status"] == "success":
                    record["batch_file"] = batch_file

        save_status(batch_rows)

        print(
            f"Finished {min(batch_start + len(batch_indices), len(remaining))}"
            f"/{len(remaining)} remaining | "
            f"success: {sum(r['status'] == 'success' for r in batch_rows)} | "
            f"failed: {sum(r['status'] == 'failed' for r in batch_rows)} | "
            f"seconds: {time.time() - batch_started:.1f}"
        )

print("\nEXTRACTION RUN FINISHED")
print("Successful:", int((status_df["status"] == "success").sum()))
print("Failed:", int((status_df["status"] == "failed").sum()))
print("Status:", STATUS_PATH)
print("Output:", OUTPUT_DIR)

===========================================================================
CELL INDEX: 112
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import mediapipe as mp
print("MediaPipe version:", mp.__version__)

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

print("MediaPipe imports successful.")

===========================================================================
CELL INDEX: 114
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================
import os
import cv2
import time
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

BASE_DIR = "/content/drive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(MANIFEST_PATH), "Working manifest not found."
assert os.path.exists(MODEL_PATH), "MediaPipe model missing."

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)
assert len(manifest) == 4257
assert manifest["video_path"].is_unique

N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS = 16, 2, 21, 3
BATCH_SIZE = 10

status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

if os.path.exists(STATUS_PATH):
    status_df = pd.read_csv(STATUS_PATH)
else:
    status_df = pd.DataFrame(columns=status_columns)

# Only skip successful videos whose saved batch actually contains their index.
valid_success = set()
for _, record in status_df[status_df["status"] == "success"].iterrows():
    batch_file = record["batch_file"]
    if not isinstance(batch_file, str) or not batch_file:
        continue
    batch_path = os.path.join(OUTPUT_DIR, batch_file)
    if not os.path.exists(batch_path):
        continue
    try:
        with np.load(batch_path, allow_pickle=True) as data:
            if int(record["index"]) in data["indices"].astype(int):
                valid_success.add(int(record["index"]))
    except Exception:
        pass

# Remove stale or failed records; retry them.
status_df = status_df[
    status_df["index"].astype(int).isin(valid_success)
].copy()
status_df.to_csv(STATUS_PATH, index=False)

remaining = [i for i in range(len(manifest)) if i not in valid_success]

print("Total videos:", len(manifest))
print("Already completed and verified:", len(valid_success))
print("Remaining:", len(remaining))

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

def save_status(rows):
    global status_df
    if rows:
        status_df = pd.concat(
            [status_df, pd.DataFrame(rows)],
            ignore_index=True
        ).drop_duplicates(subset=["index"], keep="last")
        status_df.to_csv(STATUS_PATH, index=False)

with HandLandmarker.create_from_options(options) as landmarker:
    for batch_start in range(0, len(remaining), BATCH_SIZE):
        batch_indices = remaining[batch_start:batch_start + BATCH_SIZE]
        batch_rows, batch_features, batch_flags, batch_meta = [], [], [], []
        batch_started = time.time()

        for idx in batch_indices:
            row = manifest.iloc[idx]
            path = os.path.join(VIDEO_ROOT, row["video_path"])
            cap = None

            try:
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    raise RuntimeError("Video contains no readable frames")

                frame_indices = np.linspace(
                    0, total_frames - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )
                flags = np.zeros((N_FRAMES, N_HANDS), dtype=np.uint8)
                ambiguous_frames = 0

                for t, frame_idx in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame = cap.read()
                    if not ok:
                        raise RuntimeError(
                            f"Failed to read frame {frame_idx}"
                        )

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]

                    # Correct API: handedness is separate from landmarks.
                    labels = []
                    for j in range(len(hands)):
                        categories = (
                            result.handedness[j]
                            if j < len(result.handedness) else []
                        )
                        labels.append(
                            categories[0].category_name.lower()
                            if categories else "unknown"
                        )

                    if len(hands) == 2:
                        distinct = set(labels) == {"left", "right"}
                        if distinct:
                            slots = [
                                (0 if labels[j] == "left" else 1, hands[j])
                                for j in range(2)
                            ]
                        else:
                            ambiguous_frames += 1
                            slots = [(0, hands[0]), (1, hands[1])]
                    elif len(hands) == 1:
                        label = labels[0] if labels else "unknown"
                        slot = 0 if label == "left" else (
                            1 if label == "right" else 0
                        )
                        slots = [(slot, hands[0])]
                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in hand],
                            dtype=np.float32
                        )
                        flags[t, slot] = 1

                cap.release()
                cap = None

                any_hand = int(flags.any(axis=1).sum())
                two_hands = int(flags.all(axis=1).sum())

                batch_features.append(sequence)
                batch_flags.append(flags)
                batch_meta.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total_frames,
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous_frames
                })

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "success",
                    "batch_file": "",
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous_frames,
                    "error": ""
                })

            except Exception as e:
                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "failed",
                    "batch_file": "",
                    "frames_with_any_hand": 0,
                    "frames_with_two_hands": 0,
                    "ambiguous_handedness_frames": 0,
                    "error": str(e)[:300]
                })
            finally:
                if cap is not None:
                    cap.release()

        if batch_features:
            # Unique, non-overwriting batch filename.
            existing_numbers = [
                int(f[6:10]) for f in os.listdir(OUTPUT_DIR)
                if f.startswith("batch_") and f.endswith(".npz")
                and f[6:10].isdigit()
            ]
            batch_number = max(existing_numbers, default=-1) + 1
            batch_file = f"batch_{batch_number:04d}.npz"
            batch_path = os.path.join(OUTPUT_DIR, batch_file)

            np.savez_compressed(
                batch_path,
                indices=np.array(
                    [m["index"] for m in batch_meta], dtype=np.int32
                ),
                features=np.stack(batch_features),
                flags=np.stack(batch_flags),
                metadata=pd.DataFrame(batch_meta).to_json(orient="records")
            )

            for record in batch_rows:
                if record["status"] == "success":
                    record["batch_file"] = batch_file

        save_status(batch_rows)

        print(
            f"Finished {min(batch_start + len(batch_indices), len(remaining))}"
            f"/{len(remaining)} remaining | "
            f"success: {sum(r['status'] == 'success' for r in batch_rows)} | "
            f"failed: {sum(r['status'] == 'failed' for r in batch_rows)} | "
            f"seconds: {time.time() - batch_started:.1f}"
        )

print("\nEXTRACTION RUN FINISHED")
print("Successful:", int((status_df["status"] == "success").sum()))
print("Failed:", int((status_df["status"] == "failed").sum()))
print("Status:", STATUS_PATH)
print("Output:", OUTPUT_DIR)

===========================================================================
CELL INDEX: 117
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================

import os
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FRAMES = 16
N_HANDS = 2
N_LANDMARKS = 21
N_COORDS = 3
BATCH_SIZE = 10

assert os.path.exists(MANIFEST_PATH), (
    f"Manifest not found: {MANIFEST_PATH}"
)
assert os.path.exists(MODEL_PATH), (
    f"MediaPipe model not found: {MODEL_PATH}"
)
assert os.path.isdir(VIDEO_ROOT), (
    f"Video folder not found: {VIDEO_ROOT}"
)

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)

assert len(manifest) == 4257, (
    f"Expected 4257 manifest rows, found {len(manifest)}"
)
assert manifest["video_path"].is_unique, (
    "Manifest contains duplicate video paths"
)

print("Manifest loaded:", len(manifest), "videos")
print("Video root:", VIDEO_ROOT)
print("Output folder:", OUTPUT_DIR)

# ============================================================
# 2. RESUME SUPPORT
# ============================================================

status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

if os.path.exists(STATUS_PATH):
    status_df = pd.read_csv(STATUS_PATH)
else:
    status_df = pd.DataFrame(columns=status_columns)

# A video counts as completed only if its index exists in its
# recorded batch file. Failed and incomplete videos are retried.
valid_success = set()

for _, record in status_df.iterrows():
    if record["status"] != "success":
        continue

    try:
        idx = int(record["index"])
        batch_file = record["batch_file"]

        if not isinstance(batch_file, str) or not batch_file:
            continue

        batch_path = os.path.join(OUTPUT_DIR, batch_file)
        if not os.path.isfile(batch_path):
            continue

        with np.load(batch_path, allow_pickle=True) as data:
            saved_indices = data["indices"].astype(int)

        if idx in saved_indices:
            valid_success.add(idx)

    except Exception:
        continue

# Remove stale status records, preserving verified successes.
status_df = status_df[
    status_df["index"].astype(int).isin(valid_success)
].copy()
status_df.to_csv(STATUS_PATH, index=False)

remaining = [
    idx for idx in range(len(manifest))
    if idx not in valid_success
]

print("\nTotal videos:", len(manifest))
print("Verified completed:", len(valid_success))
print("Remaining:", len(remaining))

# ============================================================
# 3. MEDIAPIPE INITIALIZATION
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=N_HANDS
)

def save_status(rows):
    """Update status CSV after each batch."""
    global status_df

    if not rows:
        return

    new_rows = pd.DataFrame(rows)

    status_df = pd.concat(
        [status_df, new_rows],
        ignore_index=True
    ).drop_duplicates(subset=["index"], keep="last")

    status_df.to_csv(STATUS_PATH, index=False)

# ============================================================
# 4. FEATURE EXTRACTION
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    for batch_start in range(0, len(remaining), BATCH_SIZE):

        batch_indices = remaining[
            batch_start:batch_start + BATCH_SIZE
        ]

        batch_rows = []
        batch_features = []
        batch_flags = []
        batch_meta = []

        start_time = time.time()

        for idx in batch_indices:

            row = manifest.iloc[idx]
            video_path = os.path.join(
                VIDEO_ROOT, row["video_path"]
            )

            cap = None

            try:
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    raise RuntimeError("Could not open video")

                total_frames = int(
                    cap.get(cv2.CAP_PROP_FRAME_COUNT)
                )

                if total_frames < 1:
                    raise RuntimeError(
                        "Video contains no readable frames"
                    )

                frame_indices = np.linspace(
                    0, total_frames - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )

                flags = np.zeros(
                    (N_FRAMES, N_HANDS),
                    dtype=np.uint8
                )

                ambiguous_frames = 0

                for t, frame_idx in enumerate(frame_indices):

                    cap.set(
                        cv2.CAP_PROP_POS_FRAMES,
                        int(frame_idx)
                    )

                    ok, frame = cap.read()

                    if not ok:
                        raise RuntimeError(
                            f"Failed to read frame {frame_idx}"
                        )

                    rgb = cv2.cvtColor(
                        frame, cv2.COLOR_BGR2RGB
                    )

                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]

                    # Correct API: handedness is separate from
                    # landmark coordinates.
                    labels = []

                    for j in range(len(hands)):
                        categories = (
                            result.handedness[j]
                            if j < len(result.handedness)
                            else []
                        )

                        labels.append(
                            categories[0].category_name.lower()
                            if categories
                            else "unknown"
                        )

                    # Slot 0 = left, slot 1 = right when labels
                    # are distinct. Otherwise use detection order.
                    if len(hands) == 2:

                        distinct = (
                            len(labels) == 2
                            and set(labels) == {"left", "right"}
                        )

                        if distinct:
                            slots = [
                                (
                                    0 if labels[j] == "left" else 1,
                                    hands[j]
                                )
                                for j in range(2)
                            ]
                        else:
                            ambiguous_frames += 1
                            slots = [
                                (0, hands[0]),
                                (1, hands[1])
                            ]

                    elif len(hands) == 1:

                        label = labels[0] if labels else "unknown"

                        slot = (
                            0 if label == "left"
                            else 1 if label == "right"
                            else 0
                        )

                        slots = [(slot, hands[0])]

                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [
                                [lm.x, lm.y, lm.z]
                                for lm in hand
                            ],
                            dtype=np.float32
                        )

                        flags[t, slot] = 1

                cap.release()
                cap = None

                frames_with_any_hand = int(
                    flags.any(axis=1).sum()
                )
                frames_with_two_hands = int(
                    flags.all(axis=1).sum()
                )

                batch_features.append(sequence)
                batch_flags.append(flags)

                batch_meta.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total_frames,
                    "frames_with_any_hand": frames_with_any_hand,
                    "frames_with_two_hands": frames_with_two_hands,
                    "ambiguous_handedness_frames": ambiguous_frames
                })

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "success",
                    "batch_file": "",
                    "frames_with_any_hand": frames_with_any_hand,
                    "frames_with_two_hands": frames_with_two_hands,
                    "ambiguous_handedness_frames": ambiguous_frames,
                    "error": ""
                })

            except Exception as exc:

                batch_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "status": "failed",
                    "batch_file": "",
                    "frames_with_any_hand": 0,
                    "frames_with_two_hands": 0,
                    "ambiguous_handedness_frames": 0,
                    "error": str(exc)[:300]
                })

            finally:
                if cap is not None:
                    cap.release()

        # ----------------------------------------------------
        # 5. SAVE THIS BATCH BEFORE UPDATING STATUS
        # ----------------------------------------------------

        if batch_features:

            batch_number = 0

            while os.path.exists(
                os.path.join(
                    OUTPUT_DIR,
                    f"batch_{batch_number:04d}.npz"
                )
            ):
                batch_number += 1

            batch_file = f"batch_{batch_number:04d}.npz"
            batch_path = os.path.join(OUTPUT_DIR, batch_file)

            np.savez_compressed(
                batch_path,
                indices=np.array(
                    [m["index"] for m in batch_meta],
                    dtype=np.int32
                ),
                features=np.stack(batch_features),
                flags=np.stack(batch_flags),
                metadata=pd.DataFrame(
                    batch_meta
                ).to_json(orient="records")
            )

            for record in batch_rows:
                if record["status"] == "success":
                    record["batch_file"] = batch_file

        save_status(batch_rows)

        successful_this_batch = sum(
            record["status"] == "success"
            for record in batch_rows
        )

        failed_this_batch = sum(
            record["status"] == "failed"
            for record in batch_rows
        )

        print(
            f"Batch complete: "
            f"{min(batch_start + len(batch_indices), len(remaining))}"
            f"/{len(remaining)} remaining videos | "
            f"success: {successful_this_batch} | "
            f"failed: {failed_this_batch} | "
            f"time: {time.time() - start_time:.1f}s",
            flush=True
        )

# ============================================================
# 6. FINAL SUMMARY
# ============================================================

status_df = pd.read_csv(STATUS_PATH)

print("\nEXTRACTION RUN FINISHED")
print("Successful videos:", int(
    (status_df["status"] == "success").sum()
))
print("Failed videos:", int(
    (status_df["status"] == "failed").sum()
))
print("Status file:", STATUS_PATH)
print("Batch directory:", OUTPUT_DIR)


===========================================================================
CELL INDEX: 131
MATCHED KEYWORDS: HandLandmarker
===========================================================================
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

with HandLandmarker.create_from_options(options) as landmarker:
    print("HandLandmarker initialized successfully.")

===========================================================================
CELL INDEX: 132
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import os
import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

VIDEO = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "dataset/extracted/Adjectives/1. loud/MVI_5257.MOV"
)
MODEL = "/tmp/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.IMAGE,
    num_hands=2
)

cap = cv2.VideoCapture(VIDEO)
assert cap.isOpened(), f"Cannot open {VIDEO}"

frames = []
for frame_idx in [0, 10, 20, 30, 40]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if ok:
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

start = time.time()
with HandLandmarker.create_from_options(options) as landmarker:
    detections = []
    for frame in frames:
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        )
        detections.append(len(result.hand_landmarks))

elapsed = time.time() - start
print("Frames processed:", len(frames))
print("Hands detected per frame:", detections)
print("Elapsed seconds:", round(elapsed, 2))
print("Seconds per frame:", round(elapsed / len(frames), 3))

===========================================================================
CELL INDEX: 134
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================

import os
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")

N_FRAMES = 16
N_HANDS = 2
N_LANDMARKS = 21
N_COORDS = 3
BATCH_SIZE = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.isfile(MANIFEST_PATH), f"Manifest missing: {MANIFEST_PATH}"
assert os.path.isfile(MODEL_PATH), f"Model missing: {MODEL_PATH}"
assert os.path.isdir(VIDEO_ROOT), f"Video directory missing: {VIDEO_ROOT}"

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)

assert len(manifest) == 4257, f"Unexpected manifest size: {len(manifest)}"
assert manifest["video_path"].is_unique, "Duplicate video paths found"

status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

# ============================================================
# 2. LOAD AND VERIFY EXISTING CHECKPOINTS
# ============================================================

if os.path.isfile(STATUS_PATH):
    old_status = pd.read_csv(STATUS_PATH)
else:
    old_status = pd.DataFrame(columns=status_columns)

verified = {}
batch_cache = {}

for _, record in old_status.iterrows():
    if record.get("status") != "success":
        continue

    try:
        idx = int(record["index"])
        filename = record["batch_file"]

        if not isinstance(filename, str) or not filename:
            continue

        batch_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.isfile(batch_path):
            continue

        if filename not in batch_cache:
            with np.load(batch_path, allow_pickle=True) as z:
                batch_cache[filename] = z["indices"].astype(int).copy()

        if idx in batch_cache[filename]:
            verified[idx] = filename

    except Exception:
        continue

# Keep verified successes only; all others will be retried.
status_records = {}
for idx, filename in verified.items():
    row = manifest.iloc[idx]
    old = old_status[
        (old_status["index"].astype(int) == idx)
        & (old_status["status"] == "success")
    ]
    old_record = old.iloc[-1].to_dict() if not old.empty else {}

    status_records[idx] = {
        "index": idx,
        "video_path": row["video_path"],
        "status": "success",
        "batch_file": filename,
        "frames_with_any_hand": old_record.get("frames_with_any_hand", 0),
        "frames_with_two_hands": old_record.get("frames_with_two_hands", 0),
        "ambiguous_handedness_frames": old_record.get(
            "ambiguous_handedness_frames", 0
        ),
        "error": ""
    }

remaining = [i for i in range(len(manifest)) if i not in verified]

def write_status():
    df = pd.DataFrame(
        [status_records[i] for i in sorted(status_records)],
        columns=status_columns
    )
    temp_path = STATUS_PATH + ".tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, STATUS_PATH)

print("Total videos:", len(manifest))
print("Verified completed:", len(verified))
print("Remaining:", len(remaining))
print("Frames per video:", N_FRAMES)
print("Batch size:", BATCH_SIZE)

# ============================================================
# 3. INITIALIZE MEDIAPIPE
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=N_HANDS
)

# ============================================================
# 4. EXTRACT 16 FRAMES PER VIDEO
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    for start in range(0, len(remaining), BATCH_SIZE):
        indices = remaining[start:start + BATCH_SIZE]

        features_batch = []
        flags_batch = []
        metadata_batch = []
        successful_rows = []
        failed_rows = []
        batch_start_time = time.time()

        for idx in indices:
            row = manifest.iloc[idx]
            video_path = os.path.join(VIDEO_ROOT, row["video_path"])
            cap = None

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    raise RuntimeError("No readable frames reported")

                frame_indices = np.linspace(
                    0, total_frames - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )
                flags = np.zeros(
                    (N_FRAMES, N_HANDS),
                    dtype=np.uint8
                )
                ambiguous = 0

                for t, frame_index in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = cap.read()

                    if not ok:
                        raise RuntimeError(
                            f"Unable to read frame {frame_index}"
                        )

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]

                    # Handedness categories come from result.handedness,
                    # not from the landmark objects.
                    labels = []
                    for j in range(len(hands)):
                        categories = (
                            result.handedness[j]
                            if j < len(result.handedness) else []
                        )
                        labels.append(
                            categories[0].category_name.lower()
                            if categories else "unknown"
                        )

                    if len(hands) == 2:
                        if len(labels) == 2 and set(labels) == {"left", "right"}:
                            slots = [
                                (0 if labels[j] == "left" else 1, hands[j])
                                for j in range(2)
                            ]
                        else:
                            ambiguous += 1
                            slots = [(0, hands[0]), (1, hands[1])]

                    elif len(hands) == 1:
                        label = labels[0] if labels else "unknown"
                        slot = (
                            0 if label == "left"
                            else 1 if label == "right"
                            else 0
                        )
                        slots = [(slot, hands[0])]
                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in hand],
                            dtype=np.float32
                        )
                        flags[t, slot] = 1

                cap.release()
                cap = None

                any_hand = int(flags.any(axis=1).sum())
                two_hands = int(flags.all(axis=1).sum())

                features_batch.append(sequence)
                flags_batch.append(flags)

                metadata_batch.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total_frames,
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous
                })

                successful_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous
                })

            except Exception as exc:
                failed_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "error": str(exc)[:300]
                })

            finally:
                if cap is not None:
                    cap.release()

        # Save features first. Only then mark videos as successful.
        if features_batch:
            batch_number = 0
            while os.path.exists(
                os.path.join(OUTPUT_DIR, f"batch_{batch_number:04d}.npz")
            ):
                batch_number += 1

            batch_filename = f"batch_{batch_number:04d}.npz"
            batch_path = os.path.join(OUTPUT_DIR, batch_filename)
            temp_batch_path = batch_path + ".tmp.npz"

            np.savez_compressed(
                temp_batch_path,
                indices=np.array(
                    [m["index"] for m in metadata_batch],
                    dtype=np.int32
                ),
                features=np.stack(features_batch),
                flags=np.stack(flags_batch),
                metadata=pd.DataFrame(metadata_batch).to_json(
                    orient="records"
                )
            )
            os.replace(temp_batch_path, batch_path)

            for record in successful_rows:
                idx = record["index"]
                status_records[idx] = {
                    "index": idx,
                    "video_path": record["video_path"],
                    "status": "success",
                    "batch_file": batch_filename,
                    "frames_with_any_hand": record["frames_with_any_hand"],
                    "frames_with_two_hands": record["frames_with_two_hands"],
                    "ambiguous_handedness_frames": record[
                        "ambiguous_handedness_frames"
                    ],
                    "error": ""
                }

        for record in failed_rows:
            idx = record["index"]
            status_records[idx] = {
                "index": idx,
                "video_path": record["video_path"],
                "status": "failed",
                "batch_file": "",
                "frames_with_any_hand": 0,
                "frames_with_two_hands": 0,
                "ambiguous_handedness_frames": 0,
                "error": record["error"]
            }

        write_status()

        print(
            f"Run progress: {min(start + len(indices), len(remaining))}"
            f"/{len(remaining)} | "
            f"batch success: {len(successful_rows)} | "
            f"batch failed: {len(failed_rows)} | "
            f"elapsed: {time.time() - batch_start_time:.1f}s",
            flush=True
        )

# ============================================================
# 5. FINAL SUMMARY
# ============================================================

final_status = pd.read_csv(STATUS_PATH)

print("\nEXTRACTION RUN FINISHED")
print("Successful videos:", int((final_status["status"] == "success").sum()))
print("Failed videos:", int((final_status["status"] == "failed").sum()))
print("Status file:", STATUS_PATH)
print("Feature batches:", OUTPUT_DIR)


===========================================================================
CELL INDEX: 141
MATCHED KEYWORDS: HandLandmarker
===========================================================================
import inspect
import mediapipe as mp

print("MediaPipe version:", mp.__version__)

print("\nHandLandmarkerOptions signature:")
print(inspect.signature(
    mp.tasks.vision.HandLandmarkerOptions
))

===========================================================================
CELL INDEX: 142
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

videos = [
    "Animals/7. Horse/MVI_8575.MP4",
    "Days_and_Time/68. Tuesday/MVI_5013.MOV",
    "Days_and_Time/79. Hour/MVI_5051.MOV"
]

configurations = [
    ("Original", 0.50, 0.50),
    ("Test_1", 0.30, 0.30),
    ("Test_2", 0.20, 0.20)
]

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

results_summary = []

for config_name, detection_conf, presence_conf in configurations:

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=detection_conf,
        min_hand_presence_confidence=presence_conf
    )

    with HandLandmarker.create_from_options(options) as landmarker:

        for relative_path in videos:

            video_path = os.path.join(VIDEO_ROOT, relative_path)
            cap = cv2.VideoCapture(video_path)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_indices = np.linspace(
                0, total_frames - 1, 16
            ).round().astype(int)

            detected_frames = 0
            two_hand_frames = 0
            detected_slots = 0

            for frame_idx in frame_indices:

                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                success, frame = cap.read()

                if not success:
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                result = landmarker.detect(mp_image)

                hand_count = len(result.hand_landmarks)

                if hand_count >= 1:
                    detected_frames += 1

                if hand_count == 2:
                    two_hand_frames += 1

                detected_slots += hand_count

            cap.release()

            results_summary.append({
                "configuration": config_name,
                "video": relative_path,
                "detection_confidence": detection_conf,
                "presence_confidence": presence_conf,
                "frames_with_hand": detected_frames,
                "frames_with_two_hands": two_hand_frames,
                "total_detected_hands": detected_slots
            })

df_results = __import__("pandas").DataFrame(results_summary)

print("TARGETED MEDIAPIPE DETECTION COMPARISON")
display(df_results)

print("\nSUMMARY BY CONFIGURATION")
display(
    df_results.groupby("configuration")[
        ["frames_with_hand", "frames_with_two_hands",
         "total_detected_hands"]
    ].sum()
)

===========================================================================
CELL INDEX: 143
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

videos = [
    "Animals/7. Horse/MVI_8575.MP4",
    "Days_and_Time/68. Tuesday/MVI_5013.MOV",
    "Days_and_Time/79. Hour/MVI_5051.MOV"
]

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.20,
    min_hand_presence_confidence=0.20
)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for relative_path in videos:

        video_path = os.path.join(VIDEO_ROOT, relative_path)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(
            0, total_frames - 1, 16
        ).round().astype(int)

        detected_images = []

        for frame_idx in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = cap.read()

            if not success:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            if not result.hand_landmarks:
                continue

            height, width = rgb.shape[:2]
            annotated = rgb.copy()

            for hand_number, hand in enumerate(result.hand_landmarks):

                for landmark in hand:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)

                    cv2.circle(
                        annotated,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

                handedness = (
                    result.handedness[hand_number][0]
                    if result.handedness[hand_number]
                    else None
                )

                label = (
                    f"{handedness.category_name} "
                    f"({handedness.score:.2f})"
                    if handedness else "Unknown"
                )

                detected_images.append(
                    (int(frame_idx), annotated, label)
                )

        cap.release()

        print("\n" + "=" * 60)
        print("VIDEO:", relative_path)
        print("Detected sampled frames:", len(detected_images))

        if detected_images:
            fig, axes = plt.subplots(
                1, len(detected_images),
                figsize=(5 * len(detected_images), 5)
            )

            if len(detected_images) == 1:
                axes = [axes]

            for ax, (frame_idx, image, label) in zip(
                axes, detected_images
            ):
                ax.imshow(image)
                ax.set_title(f"Frame {frame_idx}\n{label}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()
        else:
            print("No landmarks detected at this threshold.")

===========================================================================
CELL INDEX: 144
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

videos = [
    "Animals/7. Horse/MVI_8575.MP4",
    "Days_and_Time/68. Tuesday/MVI_5013.MOV",
    "Days_and_Time/79. Hour/MVI_5051.MOV"
]

# OpenCV person detector
person_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_fullbody.xml"
)

def get_person_crop(frame):
    """
    Propose a person-region crop.
    If Haar detection fails, use a central crop.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    people = person_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 80)
    )

    if len(people) > 0:
        # Choose the largest detected region
        x, y, bw, bh = max(
            people, key=lambda box: box[2] * box[3]
        )

        # Expand crop around detected region
        pad_x = int(bw * 0.35)
        pad_y = int(bh * 0.15)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        return frame[y1:y2, x1:x2], "haar_person"

    # Fallback: central 70% of frame
    x1 = int(w * 0.15)
    x2 = int(w * 0.85)
    y1 = int(h * 0.05)
    y2 = int(h * 0.98)

    return frame[y1:y2, x1:x2], "central_fallback"


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

results = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for relative_path in videos:

        video_path = os.path.join(VIDEO_ROOT, relative_path)
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(
            0, total_frames - 1, 16
        ).round().astype(int)

        detected_frames = 0
        total_hands = 0
        crop_types = {}
        visual_samples = []

        for frame_idx in frame_indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = cap.read()

            if not success:
                continue

            crop, crop_type = get_person_crop(frame)

            crop_types[crop_type] = crop_types.get(crop_type, 0) + 1

            # Upscale crop for hand detection
            crop_h, crop_w = crop.shape[:2]
            scale = max(1.0, 640 / max(crop_h, crop_w))

            if scale > 1:
                crop = cv2.resize(
                    crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            hand_count = len(result.hand_landmarks)

            if hand_count > 0:
                detected_frames += 1
                total_hands += hand_count

                annotated = rgb.copy()

                for hand in result.hand_landmarks:
                    for landmark in hand:
                        x = int(landmark.x * annotated.shape[1])
                        y = int(landmark.y * annotated.shape[0])

                        cv2.circle(
                            annotated,
                            (x, y),
                            4,
                            (0, 255, 0),
                            -1
                        )

                visual_samples.append(
                    (int(frame_idx), annotated)
                )

        cap.release()

        results.append({
            "video": relative_path,
            "frames_with_hand": detected_frames,
            "frames_without_hand": 16 - detected_frames,
            "total_detected_hands": total_hands,
            "crop_methods": crop_types
        })

        print("\nVideo:", relative_path)
        print("Frames with hands:", detected_frames, "/16")
        print("Total detected hands:", total_hands)
        print("Crop methods:", crop_types)

        if visual_samples:
            fig, axes = plt.subplots(
                1, len(visual_samples),
                figsize=(5 * len(visual_samples), 5)
            )

            if len(visual_samples) == 1:
                axes = [axes]

            for ax, (frame_idx, image) in zip(
                axes, visual_samples
            ):
                ax.imshow(image)
                ax.set_title(f"Frame {frame_idx}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()

print("\nFINAL CROP EXPERIMENT SUMMARY")

import pandas as pd
display(pd.DataFrame(results))

===========================================================================
CELL INDEX: 148
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

videos = [
    "Animals/7. Horse/MVI_8575.MP4",
    "Days_and_Time/68. Tuesday/MVI_5013.MOV",
    "Days_and_Time/79. Hour/MVI_5051.MOV"
]

def central_crop(frame):
    h, w = frame.shape[:2]

    x1 = int(w * 0.175)
    x2 = int(w * 0.825)

    crop = frame[:, x1:x2]
    return crop

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

results = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for relative_path in videos:

        cap = cv2.VideoCapture(
            os.path.join(VIDEO_ROOT, relative_path)
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 16).round().astype(int)

        detected_frames = 0
        total_hands = 0
        visual_samples = []

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()

            if not success:
                continue

            crop = central_crop(frame)

            h, w = crop.shape[:2]
            scale = 640 / max(h, w)

            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            count = len(result.hand_landmarks)

            if count:
                detected_frames += 1
                total_hands += count

                annotated = rgb.copy()

                for hand in result.hand_landmarks:
                    for point in hand:
                        x = int(point.x * annotated.shape[1])
                        y = int(point.y * annotated.shape[0])
                        cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

                visual_samples.append((int(idx), annotated))

        cap.release()

        results.append({
            "video": relative_path,
            "frames_with_hand": detected_frames,
            "frames_without_hand": 16 - detected_frames,
            "total_detected_hands": total_hands
        })

        print("\nVideo:", relative_path)
        print("Detected frames:", detected_frames, "/16")
        print("Total detected hands:", total_hands)

        if visual_samples:
            fig, axes = plt.subplots(
                1, len(visual_samples),
                figsize=(5 * len(visual_samples), 5)
            )

            if len(visual_samples) == 1:
                axes = [axes]

            for ax, (idx, image) in zip(axes, visual_samples):
                ax.imshow(image)
                ax.set_title(f"Frame {idx}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()

print("\nCENTRAL CROP EXPERIMENT SUMMARY")
display(pd.DataFrame(results))

===========================================================================
CELL INDEX: 149
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)

VIDEO_ROOT = os.path.join(
    BASE_DIR, "dataset", "extracted"
)

manifest = pd.read_csv(MANIFEST_PATH)

# One reproducible sample from each category
sample = (
    manifest.groupby("category", group_keys=False)
    .sample(n=1, random_state=42)
    .reset_index(drop=True)
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

results = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for _, row in sample.iterrows():

        video_path = os.path.join(
            VIDEO_ROOT, row["video_path"]
        )

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or total_frames == 0:
            cap.release()
            print("Unreadable:", row["video_path"])
            continue

        indices = np.linspace(
            0, total_frames - 1, 16
        ).round().astype(int)

        full_count = 0
        crop_count = 0

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()

            if not success:
                continue

            # Full-frame detection
            full_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            full_result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=full_rgb
                )
            )

            if len(full_result.hand_landmarks) > 0:
                full_count += 1

            # Central crop detection
            h, w = frame.shape[:2]

            crop = frame[
                :,
                int(w * 0.175):int(w * 0.825)
            ]

            crop_h, crop_w = crop.shape[:2]

            scale = 640 / max(crop_h, crop_w)

            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            crop_rgb = cv2.cvtColor(
                crop, cv2.COLOR_BGR2RGB
            )

            crop_result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=crop_rgb
                )
            )

            if len(crop_result.hand_landmarks) > 0:
                crop_count += 1

        cap.release()

        results.append({
            "category": row["category"],
            "video": row["video_path"],
            "full_frame_detections": full_count,
            "central_crop_detections": crop_count
        })

df = pd.DataFrame(results)

print("REPRESENTATIVE VIDEO COMPARISON")
display(df)

print("\nOVERALL RESULTS")
print("Videos tested:", len(df))
print("Full-frame detected frames:", df["full_frame_detections"].sum())
print("Central-crop detected frames:", df["central_crop_detections"].sum())

===========================================================================
CELL INDEX: 150
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)

VIDEO_ROOT = os.path.join(
    BASE_DIR, "dataset", "extracted"
)

manifest = pd.read_csv(MANIFEST_PATH)

# Reproducible sample: 5 videos from each category
sample = (
    manifest.groupby("category", group_keys=False)
    .sample(n=5, random_state=42)
    .reset_index(drop=True)
)

print("Videos selected:", len(sample))
print("Categories:", sample["category"].nunique())

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

results = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for video_number, (_, row) in enumerate(sample.iterrows(), 1):

        video_path = os.path.join(
            VIDEO_ROOT, row["video_path"]
        )

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or total_frames == 0:
            cap.release()
            print("Unreadable:", row["video_path"])
            continue

        indices = np.linspace(
            0, total_frames - 1, 16
        ).round().astype(int)

        full_count = 0
        crop_count = 0

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()

            if not success:
                continue

            # Full-frame detection
            full_rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            full_result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=full_rgb
                )
            )

            if full_result.hand_landmarks:
                full_count += 1

            # Central crop detection
            h, w = frame.shape[:2]

            crop = frame[
                :,
                int(w * 0.175):int(w * 0.825)
            ]

            crop_h, crop_w = crop.shape[:2]
            scale = 640 / max(crop_h, crop_w)

            crop = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            crop_rgb = cv2.cvtColor(
                crop, cv2.COLOR_BGR2RGB
            )

            crop_result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=crop_rgb
                )
            )

            if crop_result.hand_landmarks:
                crop_count += 1

        cap.release()

        results.append({
            "category": row["category"],
            "video_path": row["video_path"],
            "full_frame": full_count,
            "central_crop": crop_count,
            "difference": crop_count - full_count
        })

        if video_number % 10 == 0:
            print(f"Processed {video_number}/75 videos")

df = pd.DataFrame(results)

print("\nVIDEO-LEVEL RESULTS")
display(df)

print("\nOVERALL COMPARISON")
print("Videos tested:", len(df))
print("Full-frame detections:", df["full_frame"].sum())
print("Central-crop detections:", df["central_crop"].sum())

print("\nVIDEO-WISE COMPARISON")
print("Crop improved:", (df["difference"] > 0).sum())
print("No change:", (df["difference"] == 0).sum())
print("Crop reduced:", (df["difference"] < 0).sum())

print("\nCATEGORY-WISE RESULTS")
display(
    df.groupby("category")[
        ["full_frame", "central_crop", "difference"]
    ].sum().reset_index()
)

# Save only the experiment results, not the original features
audit_path = os.path.join(
    BASE_DIR, "audit", "crop_comparison_75_videos.csv"
)

df.to_csv(audit_path, index=False)
print("\nExperiment report saved:", audit_path)

===========================================================================
CELL INDEX: 151
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)

VIDEO_ROOT = os.path.join(
    BASE_DIR, "dataset", "extracted"
)

manifest = pd.read_csv(MANIFEST_PATH)

sample = (
    manifest.groupby("category", group_keys=False)
    .sample(n=5, random_state=42)
    .reset_index(drop=True)
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

results = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for video_no, (_, row) in enumerate(sample.iterrows(), 1):

        video_path = os.path.join(
            VIDEO_ROOT, row["video_path"]
        )

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or total_frames == 0:
            cap.release()
            continue

        indices = np.linspace(
            0, total_frames - 1, 16
        ).round().astype(int)

        full_success = 0
        crop_attempts = 0
        crop_recovered = 0
        hybrid_success = 0

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()

            if not success:
                continue

            # First attempt: original full frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            full_result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            if full_result.hand_landmarks:
                full_success += 1
                hybrid_success += 1
                continue

            # Fallback: central crop
            crop_attempts += 1

            h, w = frame.shape[:2]
            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]

            crop_h, crop_w = crop.shape[:2]
            scale = 640 / max(crop_h, crop_w)

            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            crop_rgb = cv2.cvtColor(
                resized, cv2.COLOR_BGR2RGB
            )

            crop_result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=crop_rgb
                )
            )

            if crop_result.hand_landmarks:
                crop_recovered += 1
                hybrid_success += 1

        cap.release()

        results.append({
            "category": row["category"],
            "video_path": row["video_path"],
            "full_frame_success": full_success,
            "fallback_attempts": crop_attempts,
            "fallback_recovered": crop_recovered,
            "hybrid_success": hybrid_success
        })

        if video_no % 10 == 0:
            print(f"Processed {video_no}/75 videos")

df = pd.DataFrame(results)

print("\nHYBRID FALLBACK RESULTS")
display(df)

print("\nOVERALL SUMMARY")

print("Videos tested:", len(df))
print("Full-frame detections:", df["full_frame_success"].sum())
print("Frames requiring fallback:", df["fallback_attempts"].sum())
print("Frames recovered by fallback:", df["fallback_recovered"].sum())
print("Hybrid detections:", df["hybrid_success"].sum())

print(
    "Hybrid detection coverage:",
    round(df["hybrid_success"].sum() / (len(df) * 16) * 100, 2),
    "%"
)

print("\nCATEGORY-WISE SUMMARY")

display(
    df.groupby("category")[
        [
            "full_frame_success",
            "fallback_attempts",
            "fallback_recovered",
            "hybrid_success"
        ]
    ].sum().reset_index()
)

audit_path = os.path.join(
    BASE_DIR, "audit", "hybrid_fallback_75_videos.csv"
)

df.to_csv(audit_path, index=False)

print("\nSaved experiment:", audit_path)

===========================================================================
CELL INDEX: 152
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")

videos = [
    "Animals/7. Horse/MVI_8575.MP4",
    "Days_and_Time/68. Tuesday/MVI_5013.MOV",
    "Days_and_Time/79. Hour/MVI_5051.MOV"
]

X_START = 0.175
X_END = 0.825
CROP_WIDTH_RATIO = X_END - X_START

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path="/tmp/hand_landmarker.task"
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

def transform_crop_landmarks(hand, original_width):
    transformed = []

    for point in hand:
        x = X_START + CROP_WIDTH_RATIO * point.x
        y = point.y
        z = CROP_WIDTH_RATIO * point.z

        transformed.append((x, y, z))

    return transformed

def detect_hands(landmarker, frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return landmarker.detect(
        mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )
    )

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for relative_path in videos:

        path = os.path.join(VIDEO_ROOT, relative_path)
        cap = cv2.VideoCapture(path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 16).round().astype(int)

        visual_results = []
        full_count = 0
        fallback_count = 0

        for idx in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()

            if not success:
                continue

            h, w = frame.shape[:2]

            # First attempt: full frame
            result = detect_hands(landmarker, frame)
            source = "full_frame"
            mapped_hands = result.hand_landmarks

            if not mapped_hands:

                # Fallback: central crop
                x1 = int(w * X_START)
                x2 = int(w * X_END)

                crop = frame[:, x1:x2]
                crop_h, crop_w = crop.shape[:2]

                scale = 640 / max(crop_h, crop_w)

                resized = cv2.resize(
                    crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

                crop_result = detect_hands(landmarker, resized)

                mapped_hands = [
                    transform_crop_landmarks(hand, w)
                    for hand in crop_result.hand_landmarks
                ]

                source = "central_crop"

            else:
                full_count += 1
                mapped_hands = [
                    [(p.x, p.y, p.z) for p in hand]
                    for hand in mapped_hands
                ]

            if source == "central_crop" and mapped_hands:
                fallback_count += 1

            if mapped_hands:

                annotated = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                for hand in mapped_hands:
                    for x, y, z in hand:

                        px = int(x * w)
                        py = int(y * h)

                        cv2.circle(
                            annotated,
                            (px, py),
                            4,
                            (0, 255, 0),
                            -1
                        )

                visual_results.append(
                    (int(idx), source, annotated)
                )

        cap.release()

        print("\nVIDEO:", relative_path)
        print("Full-frame detections:", full_count)
        print("Crop fallback recoveries:", fallback_count)
        print("Total detected frames:", len(visual_results))

        if visual_results:

            fig, axes = plt.subplots(
                1,
                len(visual_results),
                figsize=(5 * len(visual_results), 5)
            )

            if len(visual_results) == 1:
                axes = [axes]

            for ax, (idx, source, image) in zip(
                axes, visual_results
            ):
                ax.imshow(image)
                ax.set_title(f"Frame {idx}\n{source}")
                ax.axis("off")

            plt.tight_layout()
            plt.show()

===========================================================================
CELL INDEX: 153
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence, crop_fallback
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")

MODEL_PATH = "/tmp/hand_landmarker.task"

VIDEOS = [
    "Animals/7. Horse/MVI_8575.MP4",
    "Days_and_Time/68. Tuesday/MVI_5013.MOV",
    "Days_and_Time/79. Hour/MVI_5051.MOV"
]

N_FRAMES = 16
N_HANDS = 2
N_LANDMARKS = 21
N_COORDS = 3

X_START = 0.175
X_END = 0.825
CROP_RATIO = X_END - X_START

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

def transform_landmarks(hand, original_width):
    transformed = []

    for point in hand:
        x = X_START + CROP_RATIO * point.x
        y = point.y
        z = CROP_RATIO * point.z

        transformed.append([x, y, z])

    return np.asarray(transformed, dtype=np.float32)


def assign_hand_slot(handedness, fallback_slot):
    if handedness:
        label = handedness[0].category_name.lower()

        if label == "left":
            return 0

        if label == "right":
            return 1

    return fallback_slot


all_results = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for relative_path in VIDEOS:

        video_path = os.path.join(VIDEO_ROOT, relative_path)

        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if not cap.isOpened() or total_frames == 0:
            print("ERROR: Cannot open", relative_path)
            cap.release()
            continue

        frame_indices = np.linspace(
            0, total_frames - 1, N_FRAMES
        ).round().astype(int)

        features = np.zeros(
            (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
            dtype=np.float32
        )

        flags = np.zeros(
            (N_FRAMES, N_HANDS),
            dtype=np.uint8
        )

        source_counts = {
            "full_frame": 0,
            "central_crop": 0,
            "no_detection": 0
        }

        handedness_counts = {
            "left": 0,
            "right": 0,
            "unknown": 0
        }

        for frame_number, frame_idx in enumerate(frame_indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            success, frame = cap.read()

            if not success:
                continue

            h, w = frame.shape[:2]

            # First attempt: full frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            detected_hands = result.hand_landmarks
            source = "full_frame"

            if not detected_hands:

                # Fallback: central crop
                x1 = int(w * X_START)
                x2 = int(w * X_END)

                crop = frame[:, x1:x2]

                crop_h, crop_w = crop.shape[:2]

                scale = 640 / max(crop_h, crop_w)

                resized = cv2.resize(
                    crop,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

                crop_rgb = cv2.cvtColor(
                    resized,
                    cv2.COLOR_BGR2RGB
                )

                crop_result = landmarker.detect(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=crop_rgb
                    )
                )

                detected_hands = crop_result.hand_landmarks
                handedness = crop_result.handedness
                source = "central_crop"

            else:
                handedness = result.handedness

            if not detected_hands:
                source_counts["no_detection"] += 1
                continue

            source_counts[source] += 1

            used_slots = set()

            for hand_number, hand in enumerate(detected_hands):

                hand_label = (
                    handedness[hand_number]
                    if hand_number < len(handedness)
                    else []
                )

                if hand_label:
                    label = hand_label[0].category_name.lower()
                else:
                    label = "unknown"

                if label in handedness_counts:
                    handedness_counts[label] += 1
                else:
                    handedness_counts["unknown"] += 1

                slot = assign_hand_slot(
                    hand_label,
                    hand_number
                )

                # Prevent overwriting an already assigned slot
                if slot in used_slots:
                    available = [
                        s for s in range(N_HANDS)
                        if s not in used_slots
                    ]

                    if not available:
                        continue

                    slot = available[0]

                used_slots.add(slot)

                if source == "central_crop":
                    coords = transform_landmarks(hand, w)
                else:
                    coords = np.asarray(
                        [[p.x, p.y, p.z] for p in hand],
                        dtype=np.float32
                    )

                features[frame_number, slot] = coords
                flags[frame_number, slot] = 1

        cap.release()

        finite_check = np.isfinite(features).all()

        active_coords = features[flags == 1]

        bounds_check = (
            np.all(active_coords[:, :, :2] >= 0)
            and np.all(active_coords[:, :, :2] <= 1)
            if len(active_coords) else False
        )

        shape_check = (
            features.shape == (
                N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS
            )
        )

        flags_check = np.isin(flags, [0, 1]).all()

        result_record = {
            "video": relative_path,
            "feature_shape": features.shape,
            "detected_frames": int(flags.any(axis=1).sum()),
            "detected_hand_slots": int(flags.sum()),
            "full_frame": source_counts["full_frame"],
            "crop_fallback": source_counts["central_crop"],
            "no_detection": source_counts["no_detection"],
            "finite_values": finite_check,
            "normalized_xy_bounds": bounds_check,
            "shape_valid": shape_check,
            "flags_valid": flags_check,
            "handedness": handedness_counts
        }

        all_results.append(result_record)

        print("\n" + "=" * 60)
        print("VIDEO:", relative_path)
        print("Feature shape:", features.shape)
        print("Detected frames:", result_record["detected_frames"])
        print("Detected hand slots:", result_record["detected_hand_slots"])
        print("Detection sources:", source_counts)
        print("Handedness:", handedness_counts)
        print("Finite values:", finite_check)
        print("Normalized XY bounds:", bounds_check)
        print("Shape valid:", shape_check)
        print("Flags valid:", flags_check)

print("\n" + "=" * 60)
print("FINAL VALIDATION SUMMARY")

for result in all_results:
    print(
        result["video"],
        "| Shape:", result["shape_valid"],
        "| Finite:", result["finite_values"],
        "| Bounds:", result["normalized_xy_bounds"],
        "| Flags:", result["flags_valid"]
    )

===========================================================================
CELL INDEX: 154
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MODEL_PATH = "/tmp/hand_landmarker.task"

VIDEOS = [
    "Animals/7. Horse/MVI_8575.MP4",
    "Days_and_Time/68. Tuesday/MVI_5013.MOV",
    "Days_and_Time/79. Hour/MVI_5051.MOV"
]

X_START = 0.175
X_END = 0.825
CROP_RATIO = X_END - X_START

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.30,
    min_hand_presence_confidence=0.30
)

records = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for relative_path in VIDEOS:

        cap = cv2.VideoCapture(
            os.path.join(VIDEO_ROOT, relative_path)
        )

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames - 1, 16).round().astype(int)

        for frame_number, idx in enumerate(indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            success, frame = cap.read()

            if not success:
                continue

            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            source = "full_frame"

            if not result.hand_landmarks:

                x1 = int(w * X_START)
                x2 = int(w * X_END)

                crop = frame[:, x1:x2]
                ch, cw = crop.shape[:2]

                scale = 640 / max(ch, cw)

                resized = cv2.resize(
                    crop, None,
                    fx=scale, fy=scale,
                    interpolation=cv2.INTER_CUBIC
                )

                crop_rgb = cv2.cvtColor(
                    resized, cv2.COLOR_BGR2RGB
                )

                result = landmarker.detect(
                    mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=crop_rgb
                    )
                )

                source = "central_crop"

            for hand_no, hand in enumerate(result.hand_landmarks):

                category = result.handedness[hand_no][0]

                label = category.category_name.lower()
                score = float(category.score)

                xyz = np.array([
                    [p.x, p.y, p.z] for p in hand
                ], dtype=np.float32)

                if source == "central_crop":
                    xyz[:, 0] = X_START + CROP_RATIO * xyz[:, 0]
                    xyz[:, 2] = CROP_RATIO * xyz[:, 2]

                slot = 0 if label == "left" else 1

                records.append({
                    "video": relative_path,
                    "frame_number": frame_number,
                    "source": source,
                    "hand_label": label,
                    "hand_slot": slot,
                    "handedness_score": score,
                    "x_min": xyz[:, 0].min(),
                    "x_max": xyz[:, 0].max(),
                    "y_min": xyz[:, 1].min(),
                    "y_max": xyz[:, 1].max(),
                    "z_min": xyz[:, 2].min(),
                    "z_max": xyz[:, 2].max(),
                    "landmark_count": len(hand)
                })

        cap.release()

df = pd.DataFrame(records)

print("HANDEDNESS AND DEPTH AUDIT")
print("=" * 60)

print("\nTotal detected hands:", len(df))

print("\nHandedness distribution:")
print(df["hand_label"].value_counts())

print("\nHandedness confidence:")
print(df["handedness_score"].describe())

print("\nCoordinate ranges:")
display(
    df[["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]]
    .describe()
)

print("\nLandmark count validation:")
print(df["landmark_count"].value_counts())

print("\nDuplicate handedness labels within two-hand frames:")

two_hand_frames = (
    df.groupby(["video", "frame_number"])
    .filter(lambda group: len(group) == 2)
)

if len(two_hand_frames):
    duplicate_labels = (
        two_hand_frames.groupby(["video", "frame_number"])
        ["hand_label"].nunique()
    )

    print("Two-hand frames:", len(duplicate_labels))
    print("Frames with duplicate labels:", (duplicate_labels == 1).sum())
else:
    print("No two-hand frames found.")

print("\nFINAL RECORDS:")
display(df)

===========================================================================
CELL INDEX: 158
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import mediapipe as mp
from google.colab.patches import cv2_imshow

VIDEO_ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition/dataset/extracted"
VIDEO_PATH = f"{VIDEO_ROOT}/Days_and_Time/68. Tuesday/MVI_5013.MOV"
MODEL_PATH = "/tmp/hand_landmarker.task"

assert os.path.exists(VIDEO_PATH), "Video not found"
assert os.path.exists(MODEL_PATH), "MediaPipe model not found"

# Configure MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

cap = cv2.VideoCapture(VIDEO_PATH)

found = False

with HandLandmarker.create_from_options(options) as landmarker:
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        h, w = frame.shape[:2]

        # Central crop
        x1 = int(w * 0.175)
        x2 = int(w * 0.825)
        crop = frame[:, x1:x2]

        # Resize crop
        scale = 640 / max(crop.shape[:2])
        resized = cv2.resize(
            crop, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        # Detect hands
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:
            output = frame.copy()

            x_offset = x1 / w
            x_ratio = (x2 - x1) / w
            z_ratio = resized.shape[1] / w

            for hand in result.hand_landmarks:
                for lm in hand:
                    # Transform crop coordinates to original frame
                    x_original = x_offset + x_ratio * lm.x
                    y_original = lm.y
                    z_original = z_ratio * lm.z

                    px = int(round(x_original * w))
                    py = int(round(y_original * h))

                    cv2.circle(
                        output,
                        (px, py),
                        5,
                        (0, 255, 0),
                        -1
                    )

            found = True

            print("Detected frame index:", frame_index)
            print("Hands detected:", len(result.hand_landmarks))
            print("Original frame:", (w, h))
            print("Resized crop:", (resized.shape[1], resized.shape[0]))

            print("\nOriginal frame with transformed landmarks:")
            cv2_imshow(output)

            break

        frame_index += 1

cap.release()

if not found:
    print("No hand detected in the video frames.")

===========================================================================
CELL INDEX: 159
MATCHED KEYWORDS: HandLandmarker, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import mediapipe as mp

from google.colab.patches import cv2_imshow

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

VIDEO_ROOT = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "dataset/extracted"
)

VIDEO_PATH = (
    f"{VIDEO_ROOT}/Days_and_Time/68. Tuesday/"
    "MVI_5013.MOV"
)

MODEL_PATH = "/tmp/hand_landmarker.task"

assert os.path.exists(VIDEO_PATH), "Video not found"
assert os.path.exists(MODEL_PATH), "MediaPipe model not found"

# --------------------------------------------------
# 2. Read first video frame
# --------------------------------------------------

cap = cv2.VideoCapture(VIDEO_PATH)

success, frame = cap.read()
cap.release()

assert success, "Unable to read video frame"

h, w = frame.shape[:2]

print("Original frame dimensions:", (w, h))

# --------------------------------------------------
# 3. Create central crop
# --------------------------------------------------

x1 = int(w * 0.175)
x2 = int(w * 0.825)

crop = frame[:, x1:x2]

scale = 640 / max(crop.shape[:2])

resized = cv2.resize(
    crop,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

print("Crop dimensions:", crop.shape[:2])
print("Resized dimensions:", resized.shape[:2])

# --------------------------------------------------
# 4. MediaPipe configuration
# --------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

# --------------------------------------------------
# 5. Detect hands
# --------------------------------------------------

rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

with HandLandmarker.create_from_options(options) as landmarker:

    result = landmarker.detect(mp_image)

# --------------------------------------------------
# 6. Draw transformed hand skeletons
# --------------------------------------------------

output = frame.copy()

if not result.hand_landmarks:

    print("No hands detected.")

else:

    # Standard MediaPipe hand connections
    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    x_offset = x1 / w
    x_ratio = (x2 - x1) / w
    z_ratio = resized.shape[1] / w

    print("Hands detected:", len(result.hand_landmarks))

    for hand_index, hand in enumerate(result.hand_landmarks):

        points = []

        for lm in hand:

            # Crop coordinates -> original frame coordinates
            x_original = x_offset + x_ratio * lm.x
            y_original = lm.y
            z_original = z_ratio * lm.z

            px = int(round(x_original * w))
            py = int(round(y_original * h))

            points.append((px, py))

        # Draw connections
        for connection in connections:

            start_idx = connection.start
            end_idx = connection.end

            cv2.line(
                output,
                points[start_idx],
                points[end_idx],
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Draw 21 landmarks
        for landmark_idx, point in enumerate(points):

            cv2.circle(
                output,
                point,
                5,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                output,
                str(landmark_idx),
                (point[0] + 5, point[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 0, 0),
                1
            )

        print(
            f"Hand {hand_index + 1}: "
            f"{len(points)} landmarks"
        )

        print(
            "Transformed X range:",
            min(p[0] for p in points),
            "to",
            max(p[0] for p in points)
        )

        print(
            "Transformed Y range:",
            min(p[1] for p in points),
            "to",
            max(p[1] for p in points)
        )

    # --------------------------------------------------
    # 7. Display result
    # --------------------------------------------------

    print("\nOriginal frame with transformed hand skeletons:")
    cv2_imshow(output)

===========================================================================
CELL INDEX: 160
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import cv2
import mediapipe as mp
import os

VIDEO_PATH = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "dataset/extracted/Days_and_Time/68. Tuesday/"
    "MVI_5013.MOV"
)

MODEL_PATH = "/tmp/hand_landmarker.task"

cap = cv2.VideoCapture(VIDEO_PATH)
success, frame = cap.read()
cap.release()

assert success, "Could not read video"

h, w = frame.shape[:2]

x1 = int(w * 0.175)
x2 = int(w * 0.825)

crop = frame[:, x1:x2]

scale = 640 / max(crop.shape[:2])

resized = cv2.resize(
    crop,
    None,
    fx=scale,
    fy=scale,
    interpolation=cv2.INTER_CUBIC
)

rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

mp_image = mp.Image(
    image_format=mp.ImageFormat.SRGB,
    data=rgb
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

with mp.tasks.vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    result = landmarker.detect(mp_image)

print("HANDEDNESS AUDIT")
print("-" * 40)

for i, hand_categories in enumerate(result.handedness):

    category = hand_categories[0]

    label = category.category_name.lower()
    confidence = category.score

    slot = 0 if label == "left" else 1 if label == "right" else None

    print(f"Detected hand: {i + 1}")
    print(f"MediaPipe label: {label}")
    print(f"Confidence: {confidence:.6f}")
    print(f"Assigned slot: {slot}")
    print("-" * 40)

print("Total detected hands:", len(result.handedness))

===========================================================================
CELL INDEX: 161
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from collections import Counter

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

MANIFEST = f"{ROOT}/audit/full_include_working_manifest.csv"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"
MODEL_PATH = "/tmp/hand_landmarker.task"

OUTPUT = f"{ROOT}/audit/handedness_audit_15.csv"

assert os.path.exists(MANIFEST), "Manifest not found"
assert os.path.exists(MODEL_PATH), "MediaPipe model not found"

# --------------------------------------------------
# 2. Select one video per category
# --------------------------------------------------

df = pd.read_csv(MANIFEST)

assert "category" in df.columns, (
    f"Missing category column. Available columns: {df.columns.tolist()}"
)

assert "video_path" in df.columns, (
    f"Missing video_path column. Available columns: {df.columns.tolist()}"
)

selected = (
    df.groupby("category", group_keys=False)
      .sample(n=1, random_state=42)
      .sort_values("category")
      .reset_index(drop=True)
)

print("Selected videos:", len(selected))
print("Categories:", selected["category"].nunique())

# --------------------------------------------------
# 3. MediaPipe configuration
# --------------------------------------------------

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

# --------------------------------------------------
# 4. Audit selected videos
# --------------------------------------------------

records = []

with mp.tasks.vision.HandLandmarker.create_from_options(
    options
) as landmarker:

    for _, row in selected.iterrows():

        category = row["category"]
        video_path = str(row["video_path"])

        full_path = os.path.join(VIDEO_ROOT, video_path)

        record = {
            "category": category,
            "video_path": video_path,
            "status": "failed",
            "hands_detected": 0,
            "left_count": 0,
            "right_count": 0,
            "unknown_count": 0,
            "duplicate_labels": False,
            "labels": "",
            "confidences": ""
        }

        cap = cv2.VideoCapture(full_path)

        success, frame = cap.read()
        cap.release()

        if not success:
            record["status"] = "unreadable"
            records.append(record)
            continue

        h, w = frame.shape[:2]

        # Central crop
        x1 = int(w * 0.175)
        x2 = int(w * 0.825)

        crop = frame[:, x1:x2]

        scale = 640 / max(crop.shape[:2])

        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        labels = []
        confidences = []

        for categories in result.handedness:

            category_result = categories[0]

            label = category_result.category_name.lower()
            confidence = float(category_result.score)

            labels.append(label)
            confidences.append(confidence)

        counts = Counter(labels)

        record.update({
            "status": "success",
            "hands_detected": len(labels),
            "left_count": counts.get("left", 0),
            "right_count": counts.get("right", 0),
            "unknown_count": sum(
                count for label, count in counts.items()
                if label not in ("left", "right")
            ),
            "duplicate_labels": (
                len(labels) == 2 and len(set(labels)) < 2
            ),
            "labels": ",".join(labels),
            "confidences": ",".join(
                f"{c:.6f}" for c in confidences
            )
        })

        records.append(record)

        print(
            f"{category}: "
            f"{len(labels)} hands | "
            f"{','.join(labels) if labels else 'none'}"
        )

# --------------------------------------------------
# 5. Create audit report
# --------------------------------------------------

audit = pd.DataFrame(records)

audit.to_csv(OUTPUT, index=False)

successful = audit[audit["status"] == "success"]

print("\n" + "=" * 55)
print("HANDEDNESS AUDIT SUMMARY")
print("=" * 55)

print("Videos selected:", len(audit))
print("Successfully processed:", len(successful))
print("Unreadable videos:", (audit["status"] == "unreadable").sum())

print(
    "Videos with at least one detected hand:",
    (successful["hands_detected"] > 0).sum()
)

print(
    "Videos with two detected hands:",
    (successful["hands_detected"] == 2).sum()
)

print(
    "Left-hand detections:",
    successful["left_count"].sum()
)

print(
    "Right-hand detections:",
    successful["right_count"].sum()
)

print(
    "Unknown handedness detections:",
    successful["unknown_count"].sum()
)

print(
    "Duplicate-label two-hand frames:",
    successful["duplicate_labels"].sum()
)

all_confidences = [
    float(value)
    for values in successful["confidences"]
    for value in values.split(",")
    if value
]

if all_confidences:
    print(
        "Mean handedness confidence:",
        round(np.mean(all_confidences), 6)
    )

print("\nCategory-wise results:")
print(
    audit[
        [
            "category",
            "hands_detected",
            "left_count",
            "right_count",
            "duplicate_labels"
        ]
    ].to_string(index=False)
)

print("\nAudit saved to:")
print(OUTPUT)

===========================================================================
CELL INDEX: 162
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from collections import Counter

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

MANIFEST = f"{ROOT}/audit/full_include_working_manifest.csv"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"
MODEL_PATH = "/tmp/hand_landmarker.task"

OUTPUT = f"{ROOT}/audit/temporal_handedness_audit_15.csv"

N_FRAMES = 16

df = pd.read_csv(MANIFEST)

selected = (
    df.groupby("category", group_keys=False)
      .sample(n=1, random_state=42)
      .sort_values("category")
      .reset_index(drop=True)
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

records = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for _, row in selected.iterrows():

        category = row["category"]
        video_path = str(row["video_path"])
        full_path = os.path.join(VIDEO_ROOT, video_path)

        cap = cv2.VideoCapture(full_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames <= 0:
            cap.release()
            print(category, "UNREADABLE")
            continue

        indices = np.linspace(
            0, total_frames - 1,
            min(N_FRAMES, total_frames),
            dtype=int
        )

        video_labels = []
        detected_frames = 0
        two_hand_frames = 0
        duplicate_frames = 0
        unknown_frames = 0
        confidences = []

        for frame_index in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))

            success, frame = cap.read()

            if not success:
                continue

            h, w = frame.shape[:2]

            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]

            scale = 640 / max(crop.shape[:2])

            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            labels = []

            for categories in result.handedness:

                item = categories[0]
                label = item.category_name.lower()

                labels.append(label)
                confidences.append(float(item.score))

            if labels:
                detected_frames += 1

            if len(labels) == 2:
                two_hand_frames += 1

                if len(set(labels)) != 2:
                    duplicate_frames += 1

            if any(label not in ("left", "right") for label in labels):
                unknown_frames += 1

            video_labels.extend(labels)

        cap.release()

        counts = Counter(video_labels)

        records.append({
            "category": category,
            "video_path": video_path,
            "sampled_frames": len(indices),
            "detected_frames": detected_frames,
            "two_hand_frames": two_hand_frames,
            "left_count": counts["left"],
            "right_count": counts["right"],
            "unknown_frames": unknown_frames,
            "duplicate_two_hand_frames": duplicate_frames,
            "mean_confidence": (
                np.mean(confidences) if confidences else np.nan
            )
        })

        print(
            f"{category}: "
            f"{detected_frames}/{len(indices)} frames detected | "
            f"left={counts['left']} | "
            f"right={counts['right']} | "
            f"duplicates={duplicate_frames}"
        )

# Save report
audit = pd.DataFrame(records)
audit.to_csv(OUTPUT, index=False)

print("\n" + "=" * 55)
print("TEMPORAL HANDEDNESS AUDIT")
print("=" * 55)

print("Videos audited:", len(audit))
print("Sampled frames:", audit["sampled_frames"].sum())
print("Frames with hands:", audit["detected_frames"].sum())
print("Two-hand frames:", audit["two_hand_frames"].sum())
print("Left detections:", audit["left_count"].sum())
print("Right detections:", audit["right_count"].sum())
print("Unknown frames:", audit["unknown_frames"].sum())
print(
    "Duplicate two-hand frames:",
    audit["duplicate_two_hand_frames"].sum()
)

print(
    "Overall mean confidence:",
    round(audit["mean_confidence"].mean(), 6)
)

print("\nCategory-wise report:")
print(audit.to_string(index=False))

print("\nSaved:", OUTPUT)

===========================================================================
CELL INDEX: 163
MATCHED KEYWORDS: handedness
===========================================================================
import cv2
import numpy as np
import pandas as pd
import os
from google.colab.patches import cv2_imshow

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

VIDEO_ROOT = f"{ROOT}/dataset/extracted"

AUDIT_PATH = (
    f"{ROOT}/audit/duplicate_handedness_frames.csv"
)

audit = pd.read_csv(AUDIT_PATH)

for _, row in audit.iterrows():

    video_path = os.path.join(
        VIDEO_ROOT,
        row["video_path"]
    )

    cap = cv2.VideoCapture(video_path)

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        int(row["frame_index"])
    )

    success, frame = cap.read()
    cap.release()

    if not success:
        print("Unable to read:", row["video_path"])
        continue

    h, w = frame.shape[:2]

    x1 = int(w * 0.175)
    x2 = int(w * 0.825)

    crop = frame[:, x1:x2]

    print("\nCategory:", row["category"])
    print("Frame:", row["frame_index"])
    print("Labels:", row["label_1"], row["label_2"])

    print("Original frame:")
    cv2_imshow(frame)

    print("Central crop:")
    cv2_imshow(crop)

===========================================================================
CELL INDEX: 164
MATCHED KEYWORDS: handedness
===========================================================================
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

VIDEO_ROOT = f"{ROOT}/dataset/extracted"
AUDIT_PATH = f"{ROOT}/audit/duplicate_handedness_frames.csv"

audit = pd.read_csv(AUDIT_PATH)

fig, axes = plt.subplots(5, 2, figsize=(16, 20))

for i, (_, row) in enumerate(audit.iterrows()):

    path = os.path.join(VIDEO_ROOT, row["video_path"])

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))

    success, frame = cap.read()
    cap.release()

    if not success:
        continue

    h, w = frame.shape[:2]

    x1 = int(w * 0.175)
    x2 = int(w * 0.825)

    crop = frame[:, x1:x2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    axes[i, 0].imshow(frame_rgb)
    axes[i, 0].set_title(
        f"{row['category']} | Frame {row['frame_index']} | Original"
    )

    axes[i, 1].imshow(crop_rgb)
    axes[i, 1].set_title(
        f"{row['category']} | Central Crop | "
        f"{row['label_1']}, {row['label_2']}"
    )

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

plt.tight_layout()

output = f"{ROOT}/audit/duplicate_frames_comparison.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print("Comparison image saved:", output)

===========================================================================
CELL INDEX: 165
MATCHED KEYWORDS: handedness
===========================================================================
import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

VIDEO_ROOT = f"{ROOT}/dataset/extracted"
AUDIT_PATH = f"{ROOT}/audit/duplicate_handedness_frames.csv"

audit = pd.read_csv(AUDIT_PATH)

fig, axes = plt.subplots(5, 2, figsize=(16, 20))

for i, (_, row) in enumerate(audit.iterrows()):

    path = os.path.join(VIDEO_ROOT, row["video_path"])

    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))

    success, frame = cap.read()
    cap.release()

    if not success:
        continue

    h, w = frame.shape[:2]

    x1 = int(w * 0.175)
    x2 = int(w * 0.825)

    crop = frame[:, x1:x2]

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    axes[i, 0].imshow(frame_rgb)
    axes[i, 0].set_title(
        f"{row['category']} | Frame {row['frame_index']} | Original"
    )

    axes[i, 1].imshow(crop_rgb)
    axes[i, 1].set_title(
        f"{row['category']} | Central Crop | "
        f"{row['label_1']}, {row['label_2']}"
    )

    axes[i, 0].axis("off")
    axes[i, 1].axis("off")

plt.tight_layout()

output = f"{ROOT}/audit/duplicate_frames_comparison.png"
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.show()

print("Comparison image saved:", output)

===========================================================================
CELL INDEX: 166
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"

AUDIT_PATH = f"{ROOT}/audit/duplicate_handedness_frames.csv"
MODEL_PATH = "/tmp/hand_landmarker.task"

OUTPUT = f"{ROOT}/audit/duplicate_handedness_skeletons.png"

audit = pd.read_csv(AUDIT_PATH)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

fig, axes = plt.subplots(5, 1, figsize=(14, 24))

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for i, (_, row) in enumerate(audit.iterrows()):

        path = os.path.join(VIDEO_ROOT, row["video_path"])

        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(row["frame_index"]))

        success, frame = cap.read()
        cap.release()

        if not success:
            axes[i].set_title("Frame unavailable")
            axes[i].axis("off")
            continue

        h, w = frame.shape[:2]

        x1 = int(w * 0.175)
        x2 = int(w * 0.825)

        crop = frame[:, x1:x2]

        scale = 640 / max(crop.shape[:2])

        resized = cv2.resize(
            crop, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        result = landmarker.detect(
            mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
        )

        output = frame.copy()

        x_offset = x1 / w
        x_ratio = (x2 - x1) / w

        for hand_idx, hand in enumerate(result.hand_landmarks):

            points = []

            for lm in hand:

                px = int(round((x_offset + x_ratio * lm.x) * w))
                py = int(round(lm.y * h))

                points.append((px, py))

            for connection in connections:

                cv2.line(
                    output,
                    points[connection.start],
                    points[connection.end],
                    (0, 255, 255),
                    3
                )

            for point in points:
                cv2.circle(output, point, 5, (0, 255, 0), -1)

            category = result.handedness[hand_idx][0]

            label = (
                f"Hand {hand_idx + 1}: "
                f"{category.category_name} "
                f"({category.score:.3f})"
            )

            cv2.putText(
                output,
                label,
                (30, 45 + hand_idx * 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

        axes[i].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
        axes[i].set_title(
            f"{row['category']} | Frame {row['frame_index']}"
        )
        axes[i].axis("off")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
plt.show()

print("Saved:", OUTPUT)

===========================================================================
CELL INDEX: 167
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import pandas as pd
import mediapipe as mp

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

VIDEO_ROOT = f"{ROOT}/dataset/extracted"

AUDIT_PATH = (
    f"{ROOT}/audit/duplicate_handedness_frames.csv"
)

MODEL_PATH = "/tmp/hand_landmarker.task"

audit = pd.read_csv(AUDIT_PATH)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for _, row in audit.iterrows():

        path = os.path.join(VIDEO_ROOT, row["video_path"])
        target = int(row["frame_index"])

        cap = cv2.VideoCapture(path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print("\n", row["category"])
        print("Target frame:", target)

        for frame_idx in range(
            max(0, target - 3),
            min(total, target + 4)
        ):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

            success, frame = cap.read()

            if not success:
                continue

            h, w = frame.shape[:2]

            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]

            scale = 640 / max(crop.shape[:2])

            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            result = landmarker.detect(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
            )

            labels = [
                (
                    item[0].category_name,
                    round(float(item[0].score), 3)
                )
                for item in result.handedness
            ]

            print(
                f"Frame {frame_idx}: "
                f"{len(labels)} hands | {labels}"
            )

        cap.release()

===========================================================================
CELL INDEX: 168
MATCHED KEYWORDS: HandLandmarker, detect_for_video, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import mediapipe as mp

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"

VIDEO_PATH = (
    ROOT + "/dataset/extracted/"
    "Electronics/56. Laptop/MVI_4547.MOV"
)

MODEL_PATH = "/tmp/hand_landmarker.task"

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

assert fps > 0, "Invalid video FPS"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

with HandLandmarker.create_from_options(options) as landmarker:

    for frame_index in range(46, 53):

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = cap.read()

        if not success:
            print("Could not read frame:", frame_index)
            continue

        h, w = frame.shape[:2]

        x1 = int(w * 0.175)
        x2 = int(w * 0.825)

        crop = frame[:, x1:x2]

        scale = 640 / max(crop.shape[:2])

        resized = cv2.resize(
            crop,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_CUBIC
        )

        rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int(round(frame_index * 1000 / fps))

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        labels = []

        for categories in result.handedness:

            item = categories[0]

            labels.append(
                (
                    item.category_name,
                    round(float(item.score), 4)
                )
            )

        print(
            f"Frame {frame_index}: "
            f"{len(labels)} hands | {labels}"
        )

cap.release()

===========================================================================
CELL INDEX: 169
MATCHED KEYWORDS: HandLandmarker, detect_for_video, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import mediapipe as mp

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"
MODEL_PATH = "/tmp/hand_landmarker.task"

targets = {
    "Adjectives": ("Adjectives/9. Nice/MVI_9590.MOV", 3),
    "Clothes": ("Clothes/45. Pocket/MVI_5178.MOV", 41),
    "Jobs": ("Jobs/93. Soldier/MVI_8894.MP4", 22),
    "Places": ("Places/25. Office/MVI_3460.MOV", 6)
}

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for category, (relative_path, target_frame) in targets.items():

        path = os.path.join(VIDEO_ROOT, relative_path)

        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        print("\n" + "=" * 50)
        print(category, "| Target frame:", target_frame)

        for frame_index in range(
            max(0, target_frame - 3),
            target_frame + 4
        ):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            success, frame = cap.read()

            if not success:
                print("Frame unavailable:", frame_index)
                continue

            h, w = frame.shape[:2]

            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]

            scale = 640 / max(crop.shape[:2])

            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(round(frame_index * 1000 / fps))

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            labels = [
                (
                    item[0].category_name,
                    round(float(item[0].score), 4)
                )
                for item in result.handedness
            ]

            print(
                f"Frame {frame_index}: "
                f"{len(labels)} hands | {labels}"
            )

        cap.release()

===========================================================================
CELL INDEX: 170
MATCHED KEYWORDS: HandLandmarker, detect_for_video, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import mediapipe as mp

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"
MODEL_PATH = "/tmp/hand_landmarker.task"

targets = {
    "Adjectives": ("Adjectives/9. Nice/MVI_9590.MOV", 3),
    "Clothes": ("Clothes/45. Pocket/MVI_5178.MOV", 41),
    "Jobs": ("Jobs/93. Soldier/MVI_8894.MP4", 22),
    "Places": ("Places/25. Office/MVI_3460.MOV", 6)
}

for category, (relative_path, target_frame) in targets.items():

    path = os.path.join(VIDEO_ROOT, relative_path)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("\n" + "=" * 55)
    print(category, "| Target frame:", target_frame)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for frame_index in range(target_frame - 3, target_frame + 4):

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            success, frame = cap.read()

            if not success:
                print("Frame unavailable:", frame_index)
                continue

            h, w = frame.shape[:2]

            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]

            scale = 640 / max(crop.shape[:2])

            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # Timestamp remains unique within this video
            timestamp_ms = int(round(frame_index * 1000 / fps))

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            labels = [
                (
                    item[0].category_name,
                    round(float(item[0].score), 4)
                )
                for item in result.handedness
            ]

            print(
                f"Frame {frame_index}: "
                f"{len(labels)} hands | {labels}"
            )

    cap.release()

===========================================================================
CELL INDEX: 171
MATCHED KEYWORDS: HandLandmarker, detect_for_video, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import mediapipe as mp

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"
MODEL_PATH = "/tmp/hand_landmarker.task"

targets = {
    "Electronics": ("Electronics/56. Laptop/MVI_4547.MOV", 49),
    "Adjectives": ("Adjectives/9. Nice/MVI_9590.MOV", 3),
    "Clothes": ("Clothes/45. Pocket/MVI_5178.MOV", 41),
    "Jobs": ("Jobs/93. Soldier/MVI_8894.MP4", 22),
    "Places": ("Places/25. Office/MVI_3460.MOV", 6)
}

for category, (relative_path, target) in targets.items():

    video_path = os.path.join(VIDEO_ROOT, relative_path)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    assert fps > 0, f"Invalid FPS: {category}"

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    print("\n" + "=" * 55)
    print(category, "| Target:", target)

    with mp.tasks.vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        frame_index = 0

        while frame_index <= target + 3:

            success, frame = cap.read()

            if not success:
                break

            h, w = frame.shape[:2]

            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]

            scale = 640 / max(crop.shape[:2])

            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(
                resized,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(round(frame_index * 1000 / fps))

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            if target - 3 <= frame_index <= target + 3:

                labels = [
                    (
                        item[0].category_name,
                        round(float(item[0].score), 4)
                    )
                    for item in result.handedness
                ]

                print(
                    f"Frame {frame_index}: "
                    f"{len(labels)} hands | {labels}"
                )

            frame_index += 1

    cap.release()

===========================================================================
CELL INDEX: 172
MATCHED KEYWORDS: handedness
===========================================================================
import pandas as pd

path = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "audit/duplicate_handedness_frames.csv"
)

df = pd.read_csv(path)

df["confidence_difference"] = (
    df["confidence_1"] - df["confidence_2"]
).abs()

print("DUPLICATE HANDEDNESS CONFIDENCE ANALYSIS")
print("=" * 55)

print(
    df[
        [
            "category",
            "confidence_1",
            "confidence_2",
            "confidence_difference"
        ].to_string(index=False)
    ]
)

print("\nMean confidence difference:",
      round(df["confidence_difference"].mean(), 4))

print("Mean first-hand confidence:",
      round(df["confidence_1"].mean(), 4))

print("Mean second-hand confidence:",
      round(df["confidence_2"].mean(), 4))

===========================================================================
CELL INDEX: 173
MATCHED KEYWORDS: handedness
===========================================================================
import pandas as pd

path = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "audit/duplicate_handedness_frames.csv"
)

df = pd.read_csv(path)

df["confidence_difference"] = (
    df["confidence_1"] - df["confidence_2"]
).abs()

print("DUPLICATE HANDEDNESS CONFIDENCE ANALYSIS")
print("=" * 55)

columns = [
    "category",
    "confidence_1",
    "confidence_2",
    "confidence_difference"
]

print(df[columns].to_string(index=False))

print("\nMean confidence difference:",
      round(df["confidence_difference"].mean(), 4))

print("Mean first-hand confidence:",
      round(df["confidence_1"].mean(), 4))

print("Mean second-hand confidence:",
      round(df["confidence_2"].mean(), 4))

===========================================================================
CELL INDEX: 174
MATCHED KEYWORDS: HandLandmarker, detect_for_video, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"
MODEL_PATH = "/tmp/hand_landmarker.task"

targets = {
    "Electronics": ("Electronics/56. Laptop/MVI_4547.MOV", 46, 52),
    "Clothes": ("Clothes/45. Pocket/MVI_5178.MOV", 38, 44)
}

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

records = []

for category, (relative_path, start_frame, end_frame) in targets.items():

    video_path = os.path.join(VIDEO_ROOT, relative_path)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    assert fps > 0

    previous_hands = None

    print("\n" + "=" * 60)
    print(category)

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

        frame_index = 0

        while frame_index <= end_frame:

            success, frame = cap.read()

            if not success:
                break

            if frame_index < start_frame:
                frame_index += 1
                continue

            h, w = frame.shape[:2]

            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]

            scale = 640 / max(crop.shape[:2])

            resized = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            result = landmarker.detect_for_video(
                mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                ),
                int(round(frame_index * 1000 / fps))
            )

            current_hands = []

            for i, hand in enumerate(result.hand_landmarks):

                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand],
                    dtype=np.float32
                )

                label = result.handedness[i][0].category_name
                confidence = result.handedness[i][0].score

                current_hands.append({
                    "landmarks": landmarks,
                    "label": label,
                    "confidence": confidence
                })

            print(f"\nFrame {frame_index}")

            for i, hand in enumerate(current_hands):

                print(
                    f"Detection {i + 1}: "
                    f"{hand['label']} "
                    f"confidence={hand['confidence']:.3f}"
                )

            # Compare current detections with previous detections
            if previous_hands and current_hands:

                print("Landmark matching distances:")

                for old_i, old_hand in enumerate(previous_hands):

                    for new_i, new_hand in enumerate(current_hands):

                        distance = np.mean(
                            np.linalg.norm(
                                old_hand["landmarks"] -
                                new_hand["landmarks"],
                                axis=1
                            )
                        )

                        print(
                            f"Previous hand {old_i + 1} -> "
                            f"Current hand {new_i + 1}: "
                            f"{distance:.5f}"
                        )

                        records.append({
                            "category": category,
                            "frame": frame_index,
                            "previous_hand": old_i + 1,
                            "current_hand": new_i + 1,
                            "previous_label": old_hand["label"],
                            "current_label": new_hand["label"],
                            "landmark_distance": float(distance)
                        })

            previous_hands = current_hands
            frame_index += 1

    cap.release()

# Save diagnostic results separately
output = f"{ROOT}/audit/temporal_landmark_matching.csv"

pd.DataFrame(records).to_csv(output, index=False)

print("\nMatching analysis saved:", output)
print("Total pairwise comparisons:", len(records))

===========================================================================
CELL INDEX: 176
MATCHED KEYWORDS: HandLandmarker, detect_for_video, handedness, min_hand_detection_confidence
===========================================================================
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import urllib.request
from pathlib import Path

# ---------------- CONFIGURATION ----------------
ROOT = Path("/content/gdrive/MyDrive/ISL_Sign_Recognition")
MANIFEST = ROOT / "audit/full_include_working_manifest.csv"
REPORT = ROOT / "audit/step_4AB_temporal_validation.csv"

MODEL = "/tmp/hand_landmarker.task"

TARGETS = [
    ("Adjectives", "Adjectives/9. Nice/MVI_9590.MOV"),
    ("Clothes", "Clothes/45. Pocket/MVI_5178.MOV"),
    ("Electronics", "Electronics/56. Laptop/MVI_4547.MOV"),
    ("Jobs", "Jobs/93. Soldier/MVI_8894.MP4"),
    ("Places", "Places/25. Office/MVI_3460.MOV")
]

# Restore MediaPipe model if runtime was reset
if not os.path.exists(MODEL):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task",
        MODEL
    )

df = pd.read_csv(MANIFEST)
mp_tasks = mp.tasks
BaseOptions = mp_tasks.BaseOptions
HandLandmarker = mp_tasks.vision.HandLandmarker
Options = mp_tasks.vision.HandLandmarkerOptions
RunningMode = mp_tasks.vision.RunningMode

results = []

def landmark_distance(a, b):
    a = np.array([[p.x, p.y, p.z] for p in a])
    b = np.array([[p.x, p.y, p.z] for p in b])
    return float(np.mean(np.linalg.norm(a - b, axis=1)))

for category, relative_path in TARGETS:
    print(f"\nProcessing: {category}")

    matches = df[df["video_path"].astype(str).str.endswith(relative_path)]

    if matches.empty:
        print("Video not found in manifest:", relative_path)
        continue

    row = matches.iloc[0]

    # Adjust this column if your manifest uses another path column
    video_path = str(row["video_path"])

    # If manifest contains relative rather than absolute paths
    if not os.path.exists(video_path):
        candidates = list((ROOT / "dataset/extracted").rglob(
            Path(relative_path).name
        ))
        if not candidates:
            print("Video file not found:", relative_path)
            continue
        video_path = str(candidates[0])

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Could not open:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    previous = None
    duplicate_frames = 0
    two_hand_frames = 0
    one_hand_frames = 0
    zero_hand_frames = 0
    comparison_count = 0

    options = Options(
        base_options=BaseOptions(model_asset_path=MODEL),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    with HandLandmarker.create_from_options(options) as landmarker:
        for frame_no in range(total):
            ok, frame = cap.read()
            if not ok:
                break

            timestamp = int(frame_no * 1000 / max(fps, 1))

            # Full-frame first
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            detection = landmarker.detect_for_video(
                mp_image, timestamp
            )

            hands = list(detection.hand_landmarks)
            labels = [
                h[0].category_name
                for h in detection.handedness
            ]

            # Crop fallback only when no hand detected
            if len(hands) == 0:
                h, w = frame.shape[:2]
                x1, x2 = int(0.175*w), int(0.825*w)
                crop = frame[:, x1:x2]
                scale = min(640 / crop.shape[1],
                            640 / crop.shape[0])
                resized = cv2.resize(
                    crop,
                    (int(crop.shape[1]*scale),
                     int(crop.shape[0]*scale))
                )
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )
                detection = landmarker.detect_for_video(
                    mp_image, timestamp
                )
                hands = list(detection.hand_landmarks)
                labels = [
                    h[0].category_name
                    for h in detection.handedness
                ]

            count = len(hands)

            if count == 0:
                zero_hand_frames += 1
            elif count == 1:
                one_hand_frames += 1
            elif count >= 2:
                two_hand_frames += 1

            duplicate = (
                count == 2 and len(labels) >= 2
                and labels[0] == labels[1]
            )

            if duplicate:
                duplicate_frames += 1

            if count == 2 and previous is not None:
                d11 = landmark_distance(
                    previous[0], hands[0]
                )
                d12 = landmark_distance(
                    previous[0], hands[1]
                )
                d21 = landmark_distance(
                    previous[1], hands[0]
                )
                d22 = landmark_distance(
                    previous[1], hands[1]
                )

                direct = d11 + d22
                crossed = d12 + d21

                results.append({
                    "category": category,
                    "frame": frame_no,
                    "direct_cost": direct,
                    "crossed_cost": crossed,
                    "best_cost": min(direct, crossed),
                    "matching_margin": abs(direct-crossed),
                    "preferred_match": (
                        "direct" if direct < crossed else "crossed"
                    ),
                    "duplicate_labels": duplicate,
                    "previous_hands": 2,
                    "current_hands": 2
                })

                comparison_count += 1

            previous = hands if count == 2 else None

    cap.release()

    print(
        f"Frames: {total} | Two-hand: {two_hand_frames} | "
        f"One-hand: {one_hand_frames} | Zero-hand: {zero_hand_frames} | "
        f"Duplicate labels: {duplicate_frames} | "
        f"Comparisons: {comparison_count}"
    )

# ---------------- SAVE REPORT ----------------
report = pd.DataFrame(results)
report.to_csv(REPORT, index=False)

print("\n" + "="*55)
print("STEP 4AB — CONSOLIDATED VALIDATION")
print("="*55)

if not report.empty:
    print(report.to_string(index=False))
    print("\nTotal comparisons:", len(report))
    print("Mean best cost:", round(report["best_cost"].mean(), 5))
    print("Mean matching margin:",
          round(report["matching_margin"].mean(), 5))
    print("\nCategory-wise mean margin:")
    print(report.groupby("category")["matching_margin"].mean())
else:
    print("No valid two-hand comparisons were generated.")

print("\nReport saved:", REPORT)

===========================================================================
CELL INDEX: 177
MATCHED KEYWORDS: HandLandmarker, detect_for_video, handedness, min_hand_detection_confidence
===========================================================================

import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================

ROOT = Path("/content/gdrive/MyDrive/ISL_Sign_Recognition")

MANIFEST = ROOT / "audit/full_include_working_manifest.csv"
REPORT = ROOT / "audit/step_4AB_temporal_validation.csv"

MODEL = "/tmp/hand_landmarker.task"

TARGETS = [
    ("Adjectives", "Adjectives/9. Nice/MVI_9590.MOV"),
    ("Clothes", "Clothes/45. Pocket/MVI_5178.MOV"),
    ("Electronics", "Electronics/56. Laptop/MVI_4547.MOV"),
    ("Jobs", "Jobs/93. Soldier/MVI_8894.MP4"),
    ("Places", "Places/25. Office/MVI_3460.MOV")
]

# ==========================================
# RESTORE MEDIAPIPE MODEL IF REQUIRED
# ==========================================

if not os.path.exists(MODEL):
    print("Downloading MediaPipe model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task",
        MODEL
    )

# ==========================================
# LOAD MANIFEST
# ==========================================

df = pd.read_csv(MANIFEST)

# Automatically identify video path column
possible_columns = [
    "video_path", "path", "relative_path", "filepath"
]

PATH_COL = next(
    (c for c in possible_columns if c in df.columns),
    None
)

if PATH_COL is None:
    raise ValueError(
        f"No video path column found. Available columns: {list(df.columns)}"
    )

# ==========================================
# MEDIAPIPE CONFIGURATION
# ==========================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# ==========================================
# LANDMARK DISTANCE FUNCTION
# ==========================================

def landmark_distance(hand_a, hand_b):

    a = np.array([
        [p.x, p.y, p.z] for p in hand_a
    ])

    b = np.array([
        [p.x, p.y, p.z] for p in hand_b
    ])

    return float(
        np.mean(np.linalg.norm(a - b, axis=1))
    )

# ==========================================
# VIDEO PATH RESOLUTION
# ==========================================

def resolve_video_path(relative_path):

    candidates = df[
        df[PATH_COL].astype(str).str.endswith(relative_path)
    ]

    if candidates.empty:
        return None

    stored_path = str(candidates.iloc[0][PATH_COL])

    if os.path.isfile(stored_path):
        return stored_path

    extracted_root = ROOT / "dataset/extracted"

    candidate = extracted_root / stored_path

    if candidate.is_file():
        return str(candidate)

    filename_matches = list(
        extracted_root.rglob(Path(relative_path).name)
    )

    if filename_matches:
        return str(filename_matches[0])

    return None

# ==========================================
# CONSOLIDATED VALIDATION
# ==========================================

all_results = []
video_summary = []

for category, relative_path in TARGETS:

    print(f"\nProcessing: {category}")

    video_path = resolve_video_path(relative_path)

    if video_path is None:
        print("WARNING: Video not found:", relative_path)
        continue

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("WARNING: Could not open:", video_path)
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    previous_hands = None

    two_hand_frames = 0
    one_hand_frames = 0
    zero_hand_frames = 0
    duplicate_frames = 0
    comparisons = 0

    options = Options(
        base_options=BaseOptions(
            model_asset_path=MODEL
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )

    try:

        with HandLandmarker.create_from_options(options) as landmarker:

            for frame_no in range(total_frames):

                success, frame = cap.read()

                if not success:
                    break

                timestamp = int(
                    frame_no * 1000 / fps
                )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # Exactly ONE MediaPipe call per frame.
                detection = landmarker.detect_for_video(
                    mp_image,
                    timestamp
                )

                hands = list(
                    detection.hand_landmarks
                )

                labels = [
                    h[0].category_name
                    for h in detection.handedness
                ]

                count = len(hands)

                if count == 0:
                    zero_hand_frames += 1

                elif count == 1:
                    one_hand_frames += 1

                elif count >= 2:
                    two_hand_frames += 1

                duplicate = (
                    count == 2
                    and len(labels) >= 2
                    and labels[0] == labels[1]
                )

                if duplicate:
                    duplicate_frames += 1

                # Compare only consecutive frames
                # where BOTH frames have two hands.

                if (
                    count == 2
                    and previous_hands is not None
                ):

                    d11 = landmark_distance(
                        previous_hands[0], hands[0]
                    )

                    d12 = landmark_distance(
                        previous_hands[0], hands[1]
                    )

                    d21 = landmark_distance(
                        previous_hands[1], hands[0]
                    )

                    d22 = landmark_distance(
                        previous_hands[1], hands[1]
                    )

                    direct = d11 + d22
                    crossed = d12 + d21

                    all_results.append({

                        "category": category,

                        "frame": frame_no,

                        "direct_cost": direct,

                        "crossed_cost": crossed,

                        "best_cost": min(
                            direct, crossed
                        ),

                        "alternative_cost": max(
                            direct, crossed
                        ),

                        "matching_margin": abs(
                            direct - crossed
                        ),

                        "preferred_match": (
                            "direct"
                            if direct < crossed
                            else "crossed"
                        ),

                        "duplicate_labels": duplicate,

                        "hand1_label": (
                            labels[0]
                            if len(labels) > 0
                            else "Unknown"
                        ),

                        "hand2_label": (
                            labels[1]
                            if len(labels) > 1
                            else "Unknown"
                        )
                    })

                    comparisons += 1

                # Reset matching reference when hands disappear.
                # Never assume an absent hand retained its identity.

                previous_hands = (
                    hands if count == 2 else None
                )

    finally:
        cap.release()

    video_summary.append({

        "category": category,

        "total_frames": total_frames,

        "two_hand_frames": two_hand_frames,

        "one_hand_frames": one_hand_frames,

        "zero_hand_frames": zero_hand_frames,

        "duplicate_frames": duplicate_frames,

        "comparisons": comparisons
    })

    print(
        f"Frames: {total_frames} | "
        f"Two hands: {two_hand_frames} | "
        f"One hand: {one_hand_frames} | "
        f"No hands: {zero_hand_frames} | "
        f"Duplicate labels: {duplicate_frames} | "
        f"Comparisons: {comparisons}"
    )

# ==========================================
# SAVE REPORTS
# ==========================================

report = pd.DataFrame(all_results)
summary = pd.DataFrame(video_summary)

report.to_csv(REPORT, index=False)

summary.to_csv(
    ROOT / "audit/step_4AB_video_summary.csv",
    index=False
)

# ==========================================
# FINAL RESULTS
# ==========================================

print("\n" + "=" * 60)
print("STEP 4AB — TEMPORAL MATCHING VALIDATION")
print("=" * 60)

print("\nVIDEO SUMMARY")

if not summary.empty:
    print(summary.to_string(index=False))

print("\nMATCHING RESULTS")

if not report.empty:

    print(
        report[
            [
                "category",
                "frame",
                "direct_cost",
                "crossed_cost",
                "matching_margin",
                "preferred_match",
                "duplicate_labels"
            ]
        ].to_string(index=False)
    )

    print("\nTOTAL COMPARISONS:", len(report))

    print(
        "MEAN BEST COST:",
        round(report["best_cost"].mean(), 5)
    )

    print(
        "MEAN MATCHING MARGIN:",
        round(report["matching_margin"].mean(), 5)
    )

    print("\nCATEGORY-WISE MEAN MARGIN")

    print(
        report.groupby("category")[
            "matching_margin"
        ].mean().round(5)
    )

else:
    print("No valid consecutive two-hand comparisons.")

print("\nREPORTS SAVED:")
print(REPORT)
print(ROOT / "audit/step_4AB_video_summary.csv")

print("\nSTEP 4AB COMPLETED.")


===========================================================================
CELL INDEX: 178
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/content/gdrive/MyDrive/ISL_Sign_Recognition")

# Original data — read only
OLD_FEATURES = ROOT / "features/full_extraction"
MANIFEST = ROOT / "audit/full_include_working_manifest.csv"

# New output — separate directory
NEW_FEATURES = ROOT / "features/improved_extraction"

# Create destination without touching original batches
NEW_FEATURES.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("STEP 5A — EXTRACTION PREFLIGHT")
print("=" * 55)

# 1. Check manifest
df = pd.read_csv(MANIFEST)

print("\n1. MANIFEST")
print("Rows:", len(df))
print("Columns:", list(df.columns))

# 2. Check original batch files
batches = sorted(OLD_FEATURES.glob("*.npz"))

print("\n2. ORIGINAL FEATURE BATCHES")
print("NPZ files:", len(batches))

if not batches:
    raise FileNotFoundError("No original NPZ batches found!")

# 3. Inspect one batch without modifying it
with np.load(batches[0], allow_pickle=False) as data:

    print("\n3. ORIGINAL NPZ STRUCTURE")

    for key in data.files:
        arr = data[key]
        print(
            key,
            "| shape:", arr.shape,
            "| dtype:", arr.dtype
        )

# 4. Inspect status file
status_path = OLD_FEATURES / "status.csv"

print("\n4. ORIGINAL STATUS FILE")

if status_path.exists():
    status = pd.read_csv(status_path)
    print("Rows:", len(status))
    print("Columns:", list(status.columns))
    print(status.head(3).to_string(index=False))
else:
    print("status.csv not found")

# 5. Check output directory
print("\n5. NEW OUTPUT DIRECTORY")
print(NEW_FEATURES)
print("Existing files:", len(list(NEW_FEATURES.iterdir())))

print("\nPREFLIGHT COMPLETED.")
print("Original data was not modified.")


===========================================================================
CELL INDEX: 180
MATCHED KEYWORDS: HandLandmarker, crop_fallback_frames, ambiguous_handedness_frames, detect_for_video, handedness, min_hand_detection_confidence, improved_extraction, crop_fallback
===========================================================================

import os
import cv2
import json
import urllib.request
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path

# =====================================================
# STEP 5C — 25 VIDEO PILOT EXTRACTION
# =====================================================

ROOT = Path("/content/gdrive/MyDrive/ISL_Sign_Recognition")

MANIFEST = ROOT / "audit/full_include_working_manifest.csv"
MODEL = "/tmp/hand_landmarker.task"

OUT = ROOT / "features/improved_extraction/pilot_25"

OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 10
NUM_FRAMES = 16
CONFIDENCE = 0.3

# =====================================================
# RESTORE MODEL
# =====================================================

if not os.path.exists(MODEL):
    print("Downloading MediaPipe model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task",
        MODEL
    )

# =====================================================
# LOAD MANIFEST AND SELECT 25 VIDEOS
# =====================================================

df = pd.read_csv(MANIFEST)

# Deterministic representative sample across categories
sample = (
    df.groupby("category", group_keys=False)
      .apply(
          lambda x: x.sample(
              n=min(2, len(x)),
              random_state=42
          ),
          include_groups=True
      )
      .sort_values("video_path")
      .head(25)
      .reset_index(drop=True)
)

# Add stable original manifest indices
df = df.reset_index(drop=True)
sample = sample.copy()

# Resolve each sample back to its original index
sample["index"] = sample["video_path"].map(
    dict(zip(df["video_path"], df.index))
)

print("Pilot videos:", len(sample))
print(sample["category"].value_counts().to_string())

# =====================================================
# MEDIAPIPE SETUP
# =====================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# =====================================================
# CHECKPOINT RECOVERY
# =====================================================

STATUS_FILE = OUT / "status.csv"

if STATUS_FILE.exists():
    old_status = pd.read_csv(STATUS_FILE)
    completed = set(
        old_status.loc[
            old_status["status"] == "success",
            "index"
        ].astype(int)
    )
else:
    old_status = pd.DataFrame()
    completed = set()

print("\nAlready completed:", len(completed))

# =====================================================
# VIDEO PATH
# =====================================================

EXTRACTED = ROOT / "dataset/extracted"

def resolve_video(relative_path):
    path = EXTRACTED / relative_path
    return str(path) if path.is_file() else None

# =====================================================
# FRAME SAMPLING
# =====================================================

def sample_frame_indices(total_frames, count=16):
    return np.linspace(
        0,
        total_frames - 1,
        count
    ).round().astype(int)

# =====================================================
# LANDMARK CONVERSION
# =====================================================

def convert_landmarks(hand, z_scale=1.0):
    return np.array(
        [
            [p.x, p.y, p.z * z_scale]
            for p in hand
        ],
        dtype=np.float32
    )

# =====================================================
# DETECTION WITH FULL-FRAME / CROP FALLBACK
# =====================================================

def detect_frame(landmarker, frame, timestamp):

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    detection = landmarker.detect_for_video(
        image,
        timestamp
    )

    hands = list(detection.hand_landmarks)
    handedness = list(detection.handedness)

    # Crop fallback only if full-frame detects no hands.
    # Use a different timestamp for the second VIDEO call.
    if len(hands) == 0:

        x1 = int(0.175 * w)
        x2 = int(0.825 * w)

        crop = frame[:, x1:x2]

        scale = min(
            640 / crop.shape[1],
            640 / crop.shape[0]
        )

        resized_w = int(crop.shape[1] * scale)
        resized_h = int(crop.shape[0] * scale)

        resized = cv2.resize(
            crop,
            (resized_w, resized_h)
        )

        crop_rgb = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        crop_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=crop_rgb
        )

        # Millisecond timestamps must be strictly increasing.
        detection = landmarker.detect_for_video(
            crop_image,
            timestamp + 1
        )

        hands = list(detection.hand_landmarks)
        handedness = list(detection.handedness)

        converted = []

        for hand in hands:
            points = convert_landmarks(
                hand,
                z_scale=scale
            )

            # Convert crop-normalized X to original frame X.
            points[:, 0] = (
                x1 / w
                + (x2 - x1) / w * points[:, 0]
            )

            # Crop covers full image height.
            converted.append(points)

        return converted, handedness, True

    converted = [
        convert_landmarks(hand)
        for hand in hands
    ]

    return converted, handedness, False

# =====================================================
# PROCESS ONE VIDEO
# =====================================================

def process_video(row):

    index = int(row["index"])
    relative_path = str(row["video_path"])

    video_path = resolve_video(relative_path)

    if video_path is None:
        raise FileNotFoundError(relative_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    selected = sample_frame_indices(
        total_frames,
        NUM_FRAMES
    )

    features = np.zeros(
        (NUM_FRAMES, 2, 21, 3),
        dtype=np.float32
    )

    flags = np.zeros(
        (NUM_FRAMES, 2),
        dtype=np.uint8
    )

    frames_any = 0
    frames_two = 0
    ambiguous_frames = 0
    crop_frames = 0

    options = Options(
        base_options=BaseOptions(
            model_asset_path=MODEL
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=CONFIDENCE,
        min_hand_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE
    )

    try:

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            for t, frame_no in enumerate(selected):

                cap.set(
                    cv2.CAP_PROP_POS_FRAMES,
                    int(frame_no)
                )

                ok, frame = cap.read()

                if not ok:
                    continue

                timestamp = int(
                    t * 1000 / fps
                ) * 2

                hands, handedness, used_crop = detect_frame(
                    landmarker,
                    frame,
                    timestamp
                )

                if used_crop:
                    crop_frames += 1

                count = len(hands)

                if count:
                    frames_any += 1

                if count >= 2:
                    frames_two += 1

                if count == 0:
                    continue

                labels = []

                for item in handedness:
                    if item:
                        labels.append(
                            item[0].category_name
                        )
                    else:
                        labels.append("Unknown")

                # Only accept distinct valid handedness.
                # Do not guess duplicate or unknown identities.
                if count == 1:

                    label = labels[0] if labels else "Unknown"

                    if label == "Left":
                        features[t, 0] = hands[0]
                        flags[t, 0] = 1

                    elif label == "Right":
                        features[t, 1] = hands[0]
                        flags[t, 1] = 1

                    else:
                        ambiguous_frames += 1

                elif count >= 2:

                    first = labels[0] if len(labels) > 0 else "Unknown"
                    second = labels[1] if len(labels) > 1 else "Unknown"

                    if (
                        first in ("Left", "Right")
                        and second in ("Left", "Right")
                        and first != second
                    ):

                        slot_a = 0 if first == "Left" else 1
                        slot_b = 0 if second == "Left" else 1

                        features[t, slot_a] = hands[0]
                        features[t, slot_b] = hands[1]

                        flags[t, slot_a] = 1
                        flags[t, slot_b] = 1

                    else:
                        ambiguous_frames += 1

    finally:
        cap.release()

    metadata = {
        "index": index,
        "video_path": relative_path,
        "category": str(row["category"]),
        "label": str(row["label"]),
        "split": str(row["split"]),
        "total_video_frames": total_frames,
        "frames_with_any_hand": frames_any,
        "frames_with_two_hands": frames_two,
        "ambiguous_handedness_frames": ambiguous_frames,
        "crop_fallback_frames": crop_frames
    }

    return features, flags, metadata

# =====================================================
# BATCH PROCESSING WITH ATOMIC SAVING
# =====================================================

status_rows = (
    old_status.to_dict("records")
    if not old_status.empty
    else []
)

for start in range(0, len(sample), BATCH_SIZE):

    batch_rows = sample.iloc[
        start:start + BATCH_SIZE
    ]

    pending = batch_rows[
        ~batch_rows["index"].isin(completed)
    ]

    if pending.empty:
        continue

    batch_features = []
    batch_flags = []
    batch_indices = []
    batch_metadata = []

    print(
        f"\nProcessing pilot batch "
        f"{start // BATCH_SIZE + 1}"
    )

    for _, row in pending.iterrows():

        idx = int(row["index"])

        try:

            features, flags, metadata = process_video(
                row
            )

            batch_features.append(features)
            batch_flags.append(flags)
            batch_indices.append(idx)
            batch_metadata.append(metadata)

            status_rows.append({
                "index": idx,
                "video_path": row["video_path"],
                "status": "success",
                "batch_file": f"pilot_batch_{start // BATCH_SIZE:04d}.npz",
                "frames_with_any_hand": metadata["frames_with_any_hand"],
                "frames_with_two_hands": metadata["frames_with_two_hands"],
                "ambiguous_handedness_frames": metadata["ambiguous_handedness_frames"],
                "error": ""
            })

            print("SUCCESS:", row["video_path"])

        except Exception as e:

            status_rows.append({
                "index": idx,
                "video_path": row["video_path"],
                "status": "failed",
                "batch_file": "",
                "frames_with_any_hand": 0,
                "frames_with_two_hands": 0,
                "ambiguous_handedness_frames": 0,
                "error": repr(e)
            })

            print("FAILED:", row["video_path"], repr(e))

        # Persist checkpoint after every video.
        pd.DataFrame(status_rows).drop_duplicates(
            subset=["index"],
            keep="last"
        ).to_csv(STATUS_FILE, index=False)

    if batch_indices:

        batch_number = start // BATCH_SIZE

        destination = OUT / f"pilot_batch_{batch_number:04d}.npz"
        temporary = OUT / f"pilot_batch_{batch_number:04d}.tmp.npz"

        np.savez_compressed(
            temporary,
            indices=np.array(
                batch_indices,
                dtype=np.int32
            ),
            features=np.stack(
                batch_features
            ).astype(np.float32),
            flags=np.stack(
                batch_flags
            ).astype(np.uint8),
            metadata=np.array(
                json.dumps(
                    batch_metadata
                )
            )
        )

        os.replace(temporary, destination)

        print("SAVED:", destination)

# =====================================================
# FINAL PILOT AUDIT
# =====================================================

final_status = pd.read_csv(STATUS_FILE)

success = final_status[
    final_status["status"] == "success"
]

failed = final_status[
    final_status["status"] == "failed"
]

print("\n" + "=" * 55)
print("STEP 5C — PILOT RESULTS")
print("=" * 55)

print("Successful videos:", len(success))
print("Failed videos:", len(failed))

print(
    "Total crop fallback frames:",
    success["frames_with_any_hand"].sum()
    if len(success) else 0
)

print("\nSaved pilot batches:")

for file in sorted(OUT.glob("pilot_batch_*.npz")):
    print(file.name)

print("\nPilot status:", STATUS_FILE)
print("Pilot output:", OUT)
print("\nPILOT EXTRACTION FINISHED.")


===========================================================================
CELL INDEX: 181
MATCHED KEYWORDS: improved_extraction
===========================================================================

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/pilot_25"
)

batches = sorted(OUT.glob("pilot_batch_*.npz"))

total = 0
errors = []

for batch in batches:

    with np.load(batch, allow_pickle=False) as data:

        indices = data["indices"]
        features = data["features"]
        flags = data["flags"]
        metadata = data["metadata"]

        print("\n", batch.name)
        print("Indices:", indices.shape)
        print("Features:", features.shape)
        print("Flags:", flags.shape)

        if features.shape[1:] != (16, 2, 21, 3):
            errors.append(f"{batch.name}: Incorrect feature shape")

        if flags.shape[1:] != (16, 2):
            errors.append(f"{batch.name}: Incorrect flag shape")

        if not np.isfinite(features).all():
            errors.append(f"{batch.name}: Non-finite values")

        if not np.isin(flags, [0, 1]).all():
            errors.append(f"{batch.name}: Invalid flags")

        if len(indices) != len(features):
            errors.append(f"{batch.name}: Index mismatch")

        total += len(indices)

status = pd.read_csv(OUT / "status.csv")

print("\n" + "=" * 45)
print("PILOT INTEGRITY AUDIT")
print("=" * 45)

print("Batch files:", len(batches))
print("Total records:", total)
print("Successful status rows:", (status.status == "success").sum())
print("Failed status rows:", (status.status == "failed").sum())
print("Unique indices:", status.loc[
    status.status == "success", "index"
].nunique())

print("\nErrors:", errors)

print("\nAUDIT:", "PASSED" if not errors and total == 25 else "FAILED")


===========================================================================
CELL INDEX: 182
MATCHED KEYWORDS: HandLandmarker, crop_fallback_frames, ambiguous_handedness_frames, detect_for_video, handedness, min_hand_detection_confidence, improved_extraction, crop_fallback
===========================================================================

import os
import cv2
import json
import urllib.request
import numpy as np
import pandas as pd
import mediapipe as mp

from pathlib import Path

# =========================================================
# STEP 6 — RESUMABLE FULL DATASET EXTRACTION
# =========================================================

ROOT = Path("/content/gdrive/MyDrive/ISL_Sign_Recognition")

MANIFEST = ROOT / "audit/full_include_working_manifest.csv"
EXTRACTED = ROOT / "dataset/extracted"
MODEL = "/tmp/hand_landmarker.task"

OUT = ROOT / "features/improved_extraction/full_run"
CHECKPOINTS = OUT / "checkpoints"
BATCH_DIR = OUT / "batches"

for folder in [OUT, CHECKPOINTS, BATCH_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

NUM_FRAMES = 16
BATCH_SIZE = 10
CONFIDENCE = 0.3

# =========================================================
# 1. ENVIRONMENT AND MODEL
# =========================================================

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest missing: {MANIFEST}")

if not EXTRACTED.exists():
    raise FileNotFoundError(f"Extracted dataset missing: {EXTRACTED}")

if not os.path.exists(MODEL):
    print("Downloading MediaPipe model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/"
        "hand_landmarker.task",
        MODEL
    )

df = pd.read_csv(MANIFEST).reset_index(drop=True)

required = {"video_path", "category", "label", "split"}

if not required.issubset(df.columns):
    raise ValueError(
        f"Manifest missing columns: {required - set(df.columns)}"
    )

if df["video_path"].duplicated().any():
    raise ValueError("Duplicate video paths in manifest. Stopping.")

if len(df) != 4257:
    raise ValueError(
        f"Expected 4257 manifest rows, found {len(df)}. Stopping."
    )

print("=" * 60)
print("RESUMABLE FULL EXTRACTION")
print("=" * 60)
print("Videos in manifest:", len(df))
print("Checkpoint directory:", CHECKPOINTS)
print("Batch directory:", BATCH_DIR)

# =========================================================
# 2. MEDIAPIPE SETUP
# =========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
Options = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

# =========================================================
# 3. CHECKPOINT VALIDATION
# =========================================================

def checkpoint_path(index):
    return CHECKPOINTS / f"video_{index:06d}.npz"


def checkpoint_is_valid(index):

    path = checkpoint_path(index)

    if not path.exists():
        return False

    try:
        with np.load(path, allow_pickle=False) as data:

            features = data["features"]
            flags = data["flags"]
            metadata = json.loads(
                str(data["metadata"].item())
            )

            return (
                features.shape == (16, 2, 21, 3)
                and features.dtype == np.float32
                and flags.shape == (16, 2)
                and flags.dtype == np.uint8
                and np.isfinite(features).all()
                and np.isin(flags, [0, 1]).all()
                and int(metadata["index"]) == index
                and bool(metadata["status"] == "success")
            )

    except Exception:
        return False


def save_checkpoint(index, features, flags, metadata):

    destination = checkpoint_path(index)

    temporary = CHECKPOINTS / (
        f"video_{index:06d}.tmp.npz"
    )

    np.savez_compressed(
        temporary,
        indices=np.array([index], dtype=np.int32),
        features=features.astype(np.float32),
        flags=flags.astype(np.uint8),
        metadata=np.array(json.dumps(metadata))
    )

    # Publish only after the complete file has been written.
    os.replace(temporary, destination)


# =========================================================
# 4. VIDEO PATH
# =========================================================

def resolve_video(relative_path):

    path = EXTRACTED / str(relative_path)

    if path.is_file():
        return path

    return None


# =========================================================
# 5. LANDMARK CONVERSION
# =========================================================

def convert_landmarks(hand, z_scale=1.0):

    return np.array(
        [
            [p.x, p.y, p.z * z_scale]
            for p in hand
        ],
        dtype=np.float32
    )


# =========================================================
# 6. FULL-FRAME + CROP FALLBACK
# =========================================================

def detect_frame(landmarker, frame, timestamp):

    h, w = frame.shape[:2]

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # First attempt: original full frame.
    detection = landmarker.detect_for_video(
        image,
        timestamp
    )

    hands = list(detection.hand_landmarks)
    handedness = list(detection.handedness)

    if hands:

        converted = [
            convert_landmarks(hand)
            for hand in hands
        ]

        return converted, handedness, False

    # Second attempt: central crop.
    x1 = int(0.175 * w)
    x2 = int(0.825 * w)

    crop = frame[:, x1:x2]

    scale = min(
        640 / crop.shape[1],
        640 / crop.shape[0]
    )

    resized_w = max(1, int(crop.shape[1] * scale))
    resized_h = max(1, int(crop.shape[0] * scale))

    resized = cv2.resize(
        crop,
        (resized_w, resized_h)
    )

    crop_rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    crop_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=crop_rgb
    )

    # One millisecond after the full-frame call.
    detection = landmarker.detect_for_video(
        crop_image,
        timestamp + 1
    )

    hands = list(detection.hand_landmarks)
    handedness = list(detection.handedness)

    converted = []

    for hand in hands:

        points = convert_landmarks(
            hand,
            z_scale=scale
        )

        # Restore original normalized X coordinates.
        points[:, 0] = (
            x1 / w
            + ((x2 - x1) / w) * points[:, 0]
        )

        # Crop covers the complete original height.
        converted.append(points)

    return converted, handedness, True


# =========================================================
# 7. PROCESS ONE VIDEO CONTINUOUSLY
# =========================================================

def process_video(index, row):

    relative_path = str(row["video_path"])

    video_path = resolve_video(relative_path)

    if video_path is None:
        raise FileNotFoundError(relative_path)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError("Could not open video")

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 30.0

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("Video has no readable frames")

    selected = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).round().astype(int)

    selected_set = set(selected.tolist())

    features = np.zeros(
        (NUM_FRAMES, 2, 21, 3),
        dtype=np.float32
    )

    flags = np.zeros(
        (NUM_FRAMES, 2),
        dtype=np.uint8
    )

    frames_any = 0
    frames_two = 0
    ambiguous_frames = 0
    crop_frames = 0

    selected_position = {
        frame_number: position
        for position, frame_number in enumerate(selected)
    }

    options = Options(
        base_options=BaseOptions(
            model_asset_path=MODEL
        ),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=CONFIDENCE,
        min_hand_presence_confidence=CONFIDENCE,
        min_tracking_confidence=CONFIDENCE
    )

    try:

        with HandLandmarker.create_from_options(
            options
        ) as landmarker:

            frame_no = 0

            # Decode and process EVERY frame to preserve
            # continuous VIDEO-mode tracking.
            while True:

                ok, frame = cap.read()

                if not ok:
                    break

                timestamp = int(
                    frame_no * 1000 / fps
                )

                hands, handedness, used_crop = detect_frame(
                    landmarker,
                    frame,
                    timestamp * 2
                )

                if used_crop:
                    crop_frames += 1

                # Only save the 16 selected frames.
                if frame_no in selected_set:

                    t = selected_position[frame_no]

                    count = len(hands)

                    if count:
                        frames_any += 1

                    if count >= 2:
                        frames_two += 1

                    labels = [
                        item[0].category_name
                        if item and len(item) > 0
                        else "Unknown"
                        for item in handedness
                    ]

                    if count == 1:

                        label = (
                            labels[0]
                            if labels else "Unknown"
                        )

                        if label == "Left":
                            features[t, 0] = hands[0]
                            flags[t, 0] = 1

                        elif label == "Right":
                            features[t, 1] = hands[0]
                            flags[t, 1] = 1

                        else:
                            ambiguous_frames += 1

                    elif count >= 2:

                        first = (
                            labels[0]
                            if len(labels) > 0
                            else "Unknown"
                        )

                        second = (
                            labels[1]
                            if len(labels) > 1
                            else "Unknown"
                        )

                        # Accept only distinct valid labels.
                        if (
                            first in ("Left", "Right")
                            and second in ("Left", "Right")
                            and first != second
                        ):

                            slot_a = (
                                0 if first == "Left" else 1
                            )

                            slot_b = (
                                0 if second == "Left" else 1
                            )

                            features[t, slot_a] = hands[0]
                            features[t, slot_b] = hands[1]

                            flags[t, slot_a] = 1
                            flags[t, slot_b] = 1

                        else:
                            # Do not invent left/right identity.
                            ambiguous_frames += 1

                frame_no += 1

    finally:
        cap.release()

    if frame_no == 0:
        raise RuntimeError("No frames were decoded")

    metadata = {
        "index": int(index),
        "video_path": relative_path,
        "category": str(row["category"]),
        "label": str(row["label"]),
        "split": str(row["split"]),
        "total_video_frames": int(frame_no),
        "frames_with_any_hand": int(frames_any),
        "frames_with_two_hands": int(frames_two),
        "ambiguous_handedness_frames": int(ambiguous_frames),
        "crop_fallback_frames": int(crop_frames),
        "status": "success"
    }

    return features, flags, metadata


# =========================================================
# 8. RESUME FROM VERIFIED CHECKPOINTS
# =========================================================

valid_completed = 0
invalid_existing = 0

for index in range(len(df)):

    path = checkpoint_path(index)

    if path.exists():

        if checkpoint_is_valid(index):
            valid_completed += 1
        else:
            invalid_existing += 1

print("\nRESUME CHECK")
print("Verified completed videos:", valid_completed)
print("Invalid/incomplete checkpoints:", invalid_existing)
print("Videos remaining:", len(df) - valid_completed)

# =========================================================
# 9. EXTRACTION LOOP
# =========================================================

errors = []
newly_completed = 0

for index, row in df.iterrows():

    # Never redo a verified completed video.
    if checkpoint_is_valid(index):
        continue

    try:

        features, flags, metadata = process_video(
            index,
            row
        )

        save_checkpoint(
            index,
            features,
            flags,
            metadata
        )

        # Verify after saving.
        if not checkpoint_is_valid(index):
            raise RuntimeError(
                "Checkpoint verification failed after save"
            )

        newly_completed += 1

        print(
            f"SUCCESS {index + 1}/{len(df)} | "
            f"{row['video_path']}"
        )

    except Exception as e:

        errors.append({
            "index": int(index),
            "video_path": str(row["video_path"]),
            "error": repr(e)
        })

        print(
            f"FAILED {index + 1}/{len(df)} | "
            f"{row['video_path']} | {repr(e)}"
        )

    # Periodic progress summary.
    if (index + 1) % 50 == 0:

        completed_now = sum(
            checkpoint_is_valid(i)
            for i in range(len(df))
        )

        print(
            f"\nPROGRESS: {completed_now}/{len(df)} "
            f"videos safely checkpointed\n"
        )

# Save current error report.
error_file = OUT / "errors_latest.csv"

pd.DataFrame(
    errors,
    columns=["index", "video_path", "error"]
).to_csv(error_file, index=False)

# =========================================================
# 10. FINAL CHECKPOINT INVENTORY
# =========================================================

completed_indices = [
    i for i in range(len(df))
    if checkpoint_is_valid(i)
]

missing_indices = sorted(
    set(range(len(df))) - set(completed_indices)
)

print("\n" + "=" * 60)
print("EXTRACTION SESSION SUMMARY")
print("=" * 60)

print("Newly completed this session:", newly_completed)
print("Total verified completed:", len(completed_indices))
print("Remaining:", len(missing_indices))
print("Errors this session:", len(errors))

# =========================================================
# 11. CONSOLIDATE INTO BATCHES ONLY WHEN COMPLETE
# =========================================================

if len(completed_indices) == len(df):

    print("\nAll videos completed. Building final batches...")

    for batch_number, start in enumerate(
        range(0, len(df), BATCH_SIZE)
    ):

        indices = list(
            range(start, min(start + BATCH_SIZE, len(df)))
        )

        destination = (
            BATCH_DIR / f"batch_{batch_number:04d}.npz"
        )

        # Skip an already correct consolidated batch.
        if destination.exists():

            try:
                with np.load(
                    destination,
                    allow_pickle=False
                ) as existing:

                    if np.array_equal(
                        existing["indices"],
                        np.array(indices, dtype=np.int32)
                    ):
                        continue

            except Exception:
                pass

        feature_list = []
        flag_list = []
        metadata_list = []

        for index in indices:

            with np.load(
                checkpoint_path(index),
                allow_pickle=False
            ) as data:

                feature_list.append(data["features"])
                flag_list.append(data["flags"])

                metadata_list.append(
                    json.loads(
                        str(data["metadata"].item())
                    )
                )

        temporary = BATCH_DIR / (
            f"batch_{batch_number:04d}.tmp.npz"
        )

        np.savez_compressed(
            temporary,
            indices=np.array(indices, dtype=np.int32),
            features=np.stack(feature_list).astype(np.float32),
            flags=np.stack(flag_list).astype(np.uint8),
            metadata=np.array(
                json.dumps(metadata_list)
            )
        )

        os.replace(temporary, destination)

        print("BATCH SAVED:", destination.name)

    print("\nFULL DATASET EXTRACTION COMPLETED.")

else:

    print(
        "\nExtraction is incomplete. "
        "Rerun this same cell after reconnecting."
    )

    print(
        "Verified checkpoints will be skipped automatically."
    )

print("\nOutput directory:", OUT)
print("Checkpoint directory:", CHECKPOINTS)
print("Final batches directory:", BATCH_DIR)
print("Error report:", error_file)


===========================================================================
CELL INDEX: 183
MATCHED KEYWORDS: ambiguous_handedness_frames, handedness, improved_extraction
===========================================================================

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path("/content/gdrive/MyDrive/ISL_Sign_Recognition")
OUT = ROOT / "features/improved_extraction/full_run"
BATCH_DIR = OUT / "batches"
CHECKPOINTS = OUT / "checkpoints"
MANIFEST = ROOT / "audit/full_include_working_manifest.csv"

df = pd.read_csv(MANIFEST)
files = sorted(BATCH_DIR.glob("batch_*.npz"))

errors = []
all_indices = []
total_records = 0
total_nonzero_flags = 0
total_ambiguous = 0

for file in files:
    try:
        with np.load(file, allow_pickle=False) as data:
            indices = data["indices"]
            features = data["features"]
            flags = data["flags"]
            metadata = json.loads(str(data["metadata"].item()))

            if features.shape != (len(indices), 16, 2, 21, 3):
                errors.append(f"{file.name}: feature shape error")

            if flags.shape != (len(indices), 16, 2):
                errors.append(f"{file.name}: flag shape error")

            if not np.isfinite(features).all():
                errors.append(f"{file.name}: non-finite features")

            if not np.isin(flags, [0, 1]).all():
                errors.append(f"{file.name}: invalid flags")

            if len(metadata) != len(indices):
                errors.append(f"{file.name}: metadata count mismatch")

            if len(indices) > 10:
                errors.append(f"{file.name}: batch exceeds 10 records")

            if len(indices) != len(set(indices.tolist())):
                errors.append(f"{file.name}: duplicate indices")

            for idx, item in zip(indices, metadata):
                if int(item["index"]) != int(idx):
                    errors.append(f"{file.name}: metadata index mismatch")
                if item["video_path"] != df.iloc[int(idx)]["video_path"]:
                    errors.append(f"{file.name}: manifest path mismatch")

            all_indices.extend(indices.tolist())
            total_records += len(indices)
            total_nonzero_flags += int(flags.sum())
            total_ambiguous += sum(
                int(item.get("ambiguous_handedness_frames", 0))
                for item in metadata
            )

    except Exception as e:
        errors.append(f"{file.name}: {repr(e)}")

expected = set(range(len(df)))
actual = set(all_indices)

missing = sorted(expected - actual)
unexpected = sorted(actual - expected)

print("=" * 55)
print("FINAL FULL EXTRACTION INTEGRITY AUDIT")
print("=" * 55)

print("Manifest records:", len(df))
print("Batch files:", len(files))
print("Total records:", total_records)
print("Unique indices:", len(actual))
print("Missing indices:", len(missing))
print("Unexpected indices:", len(unexpected))
print("Nonzero hand flags:", total_nonzero_flags)
print("Ambiguous sampled frames:", total_ambiguous)

print("\nErrors:", len(errors))
for error in errors[:20]:
    print(error)

print("\nAUDIT RESULT:",
      "PASSED" if (
          len(files) == 426
          and total_records == 4257
          and actual == expected
          and not errors
      ) else "FAILED")

print("\nOriginal extraction directory was not modified.")


===========================================================================
CELL INDEX: 184
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import numpy as np

# Improved extraction directory — read-only inspection
FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

print("=" * 55)
print("IMPROVED EXTRACTION — FEATURE STRUCTURE INSPECTION")
print("=" * 55)

print("Batch files found:", len(batch_files))

if not batch_files:
    raise FileNotFoundError("No batch files found. Check the directory path.")

# Inspect only the first batch
with np.load(batch_files[0], allow_pickle=True) as data:

    print("\nFirst batch:", os.path.basename(batch_files[0]))
    print("\nStored arrays:")

    for key in data.files:
        arr = data[key]
        print(f"\n{key}")
        print("  Shape:", arr.shape)
        print("  Data type:", arr.dtype)

        # Show one example for metadata or small arrays
        if key.lower() in ["metadata", "indices"] or arr.size <= 10:
            print("  Sample:", arr.flat[:3])

print("\nInspection completed. No files were modified.")


===========================================================================
CELL INDEX: 185
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

total_videos = 0
nan_count = 0
inf_count = 0
invalid_flags = 0
missing_slot_nonzero = 0

flag_counts = {0: 0, 1: 0}
frames_with_hands = 0
frames_without_hands = 0
frames_with_two_hands = 0

coordinate_min = np.inf
coordinate_max = -np.inf

video_no_hands = 0
metadata_records = []

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        indices = data["indices"]
        features = data["features"]
        flags = data["flags"]
        metadata = data["metadata"]

        total_videos += len(indices)

        nan_count += int(np.isnan(features).sum())
        inf_count += int(np.isinf(features).sum())

        invalid_flags += int((~np.isin(flags, [0, 1])).sum())

        coordinate_min = min(coordinate_min, float(np.nanmin(features)))
        coordinate_max = max(coordinate_max, float(np.nanmax(features)))

        # Count detected and missing hand slots
        flag_counts[0] += int((flags == 0).sum())
        flag_counts[1] += int((flags == 1).sum())

        # Check whether missing-hand slots contain nonzero coordinates
        missing_mask = (flags == 0)
        missing_slot_nonzero += int(
            np.any(features[missing_mask] != 0, axis=(1, 2, 3)).sum()
        ) if missing_mask.any() else 0

        # Frame-level detection statistics
        hands_per_frame = flags.sum(axis=2)

        frames_with_hands += int((hands_per_frame >= 1).sum())
        frames_without_hands += int((hands_per_frame == 0).sum())
        frames_with_two_hands += int((hands_per_frame == 2).sum())

        # Video-level no-hand count
        video_no_hands += int((flags.sum(axis=(1, 2)) == 0).sum())

print("=" * 58)
print("IMPROVED EXTRACTION — NUMERICAL QUALITY AUDIT")
print("=" * 58)

print("Batch files:", len(batch_files))
print("Videos inspected:", total_videos)

print("\nNUMERICAL VALIDITY")
print("NaN values:", nan_count)
print("Infinite values:", inf_count)
print("Invalid flag values:", invalid_flags)

print("\nCOORDINATE RANGE")
print("Minimum:", coordinate_min)
print("Maximum:", coordinate_max)

print("\nHAND SLOT STATISTICS")
print("Missing slots (flag 0):", flag_counts[0])
print("Detected slots (flag 1):", flag_counts[1])
print("Missing slots with nonzero coordinates:", missing_slot_nonzero)

print("\nFRAME STATISTICS")
print("Frames with at least one hand:", frames_with_hands)
print("Frames without hands:", frames_without_hands)
print("Frames with two hands:", frames_with_two_hands)

print("\nVIDEO STATISTICS")
print("Videos with no detected hands:", video_no_hands)

print("\nAUDIT COMPLETED")


===========================================================================
CELL INDEX: 186
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import numpy as np

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

total_videos = 0
nan_count = 0
inf_count = 0
invalid_flags = 0
missing_slot_nonzero = 0

flag_counts = {0: 0, 1: 0}
frames_with_hands = 0
frames_without_hands = 0
frames_with_two_hands = 0
video_no_hands = 0

coordinate_min = np.inf
coordinate_max = -np.inf

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        indices = data["indices"]
        features = data["features"]
        flags = data["flags"]

        total_videos += len(indices)

        nan_count += int(np.isnan(features).sum())
        inf_count += int(np.isinf(features).sum())
        invalid_flags += int((~np.isin(flags, [0, 1])).sum())

        coordinate_min = min(coordinate_min, float(np.min(features)))
        coordinate_max = max(coordinate_max, float(np.max(features)))

        flag_counts[0] += int((flags == 0).sum())
        flag_counts[1] += int((flags == 1).sum())

        # Corrected missing-slot coordinate check
        missing_mask = (flags == 0)
        if missing_mask.any():
            missing_coordinates = features[missing_mask]
            missing_slot_nonzero += int(
                np.any(missing_coordinates != 0, axis=(1, 2)).sum()
            )

        hands_per_frame = flags.sum(axis=2)

        frames_with_hands += int((hands_per_frame >= 1).sum())
        frames_without_hands += int((hands_per_frame == 0).sum())
        frames_with_two_hands += int((hands_per_frame == 2).sum())

        video_no_hands += int(
            (flags.sum(axis=(1, 2)) == 0).sum()
        )

print("=" * 55)
print("IMPROVED EXTRACTION — NUMERICAL QUALITY AUDIT")
print("=" * 55)

print("Batch files:", len(batch_files))
print("Videos inspected:", total_videos)

print("\nNUMERICAL VALIDITY")
print("NaN values:", nan_count)
print("Infinite values:", inf_count)
print("Invalid flag values:", invalid_flags)

print("\nCOORDINATE RANGE")
print("Minimum:", coordinate_min)
print("Maximum:", coordinate_max)

print("\nHAND SLOT STATISTICS")
print("Missing slots (flag 0):", flag_counts[0])
print("Detected slots (flag 1):", flag_counts[1])
print("Missing slots with nonzero coordinates:", missing_slot_nonzero)

print("\nFRAME STATISTICS")
print("Frames with at least one hand:", frames_with_hands)
print("Frames without hands:", frames_without_hands)
print("Frames with two hands:", frames_with_two_hands)

print("\nVIDEO STATISTICS")
print("Videos with no detected hands:", video_no_hands)

print("\nAUDIT COMPLETED")


===========================================================================
CELL INDEX: 187
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import numpy as np

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

total_detected_landmarks = 0
outside_x = 0
outside_y = 0
outside_z = 0

x_min, x_max = np.inf, -np.inf
y_min, y_max = np.inf, -np.inf
z_min, z_max = np.inf, -np.inf

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        features = data["features"]
        flags = data["flags"]

        # Only inspect detected hands
        detected = features[flags == 1]

        total_detected_landmarks += detected.shape[0] * 21

        x = detected[:, :, 0]
        y = detected[:, :, 1]
        z = detected[:, :, 2]

        outside_x += int(((x < 0) | (x > 1)).sum())
        outside_y += int(((y < 0) | (y > 1)).sum())
        outside_z += int(((z < 0) | (z > 1)).sum())

        x_min = min(x_min, float(x.min()))
        x_max = max(x_max, float(x.max()))

        y_min = min(y_min, float(y.min()))
        y_max = max(y_max, float(y.max()))

        z_min = min(z_min, float(z.min()))
        z_max = max(z_max, float(z.max()))

print("=" * 55)
print("LANDMARK COORDINATE BOUNDARY AUDIT")
print("=" * 55)

print("Detected hand slots:", int(total_detected_landmarks / 21))
print("Total detected landmarks:", total_detected_landmarks)

print("\nX COORDINATES")
print("Range:", x_min, "to", x_max)
print("Outside [0,1]:", outside_x)

print("\nY COORDINATES")
print("Range:", y_min, "to", y_max)
print("Outside [0,1]:", outside_y)

print("\nZ COORDINATES")
print("Range:", z_min, "to", z_max)
print("Outside [0,1]:", outside_z)

print("\nBoundary audit completed.")


===========================================================================
CELL INDEX: 188
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import json
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

below_zero = 0
above_one = 0
total_y = 0

affected_videos = []

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        indices = data["indices"]
        features = data["features"]
        flags = data["flags"]

        metadata = json.loads(str(data["metadata"].item()))

        for i in range(len(indices)):
            # Only inspect detected hands
            detected = features[i][flags[i] == 1]

            if detected.shape[0] == 0:
                continue

            y = detected[:, :, 1].flatten()

            low = int((y < 0).sum())
            high = int((y > 1).sum())

            below_zero += low
            above_one += high
            total_y += len(y)

            if low + high > 0:
                affected_videos.append({
                    "index": int(indices[i]),
                    "video_path": metadata[i]["video_path"],
                    "below_zero": low,
                    "above_one": high,
                    "total_outliers": low + high
                })

df = pd.DataFrame(affected_videos)

print("=" * 60)
print("Y-COORDINATE OUTLIER INVESTIGATION")
print("=" * 60)

print("Total detected Y coordinates:", total_y)
print("Below zero:", below_zero)
print("Above one:", above_one)
print("Total outliers:", below_zero + above_one)

print(
    "Outlier percentage:",
    round((below_zero + above_one) / total_y * 100, 4),
    "%"
)

print("\nVideos containing outliers:", len(df))

print("\nTOP 20 VIDEOS WITH MOST OUTLIERS")

if not df.empty:
    print(
        df.sort_values("total_outliers", ascending=False)
          .head(20)
          .to_string(index=False)
    )
else:
    print("No affected videos.")

print("\nInvestigation completed. No files modified.")


===========================================================================
CELL INDEX: 189
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import json
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

category_stats = {}
video_stats = {}
frame_stats = np.zeros(16, dtype=np.int64)

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        indices = data["indices"]
        features = data["features"]
        flags = data["flags"]
        metadata = json.loads(str(data["metadata"].item()))

        for i in range(len(indices)):
            category = metadata[i]["category"]
            video_path = metadata[i]["video_path"]

            # Initialize category counters
            if category not in category_stats:
                category_stats[category] = {
                    "videos": 0,
                    "affected_videos": 0,
                    "total_y": 0,
                    "below_zero": 0,
                    "above_one": 0
                }

            category_stats[category]["videos"] += 1

            # Only inspect detected hand slots
            detected_mask = flags[i] == 1
            y = features[i, :, :, :, 1]

            valid_y = y[detected_mask]
            total_y = valid_y.size

            below = int((valid_y < 0).sum())
            above = int((valid_y > 1).sum())
            outliers = below + above

            category_stats[category]["total_y"] += total_y
            category_stats[category]["below_zero"] += below
            category_stats[category]["above_one"] += above

            if outliers > 0:
                category_stats[category]["affected_videos"] += 1

            video_stats[video_path] = {
                "index": int(indices[i]),
                "category": category,
                "total_y": total_y,
                "below_zero": below,
                "above_one": above,
                "outliers": outliers,
                "outlier_percent": (
                    outliers / total_y * 100 if total_y else 0
                )
            }

            # Count outliers by sampled frame position
            for frame in range(16):
                frame_y = y[frame][detected_mask[frame]]
                frame_stats[frame] += int(
                    ((frame_y < 0) | (frame_y > 1)).sum()
                )

# Category report
category_rows = []

for category, stats in category_stats.items():
    total_outliers = stats["below_zero"] + stats["above_one"]

    category_rows.append({
        "category": category,
        "videos": stats["videos"],
        "affected_videos": stats["affected_videos"],
        "below_zero": stats["below_zero"],
        "above_one": stats["above_one"],
        "total_outliers": total_outliers,
        "outlier_percent": (
            total_outliers / stats["total_y"] * 100
            if stats["total_y"] else 0
        )
    })

category_df = pd.DataFrame(category_rows).sort_values(
    "outlier_percent", ascending=False
)

video_df = pd.DataFrame(video_stats.values()).sort_values(
    "outlier_percent", ascending=False
)

frame_df = pd.DataFrame({
    "sampled_frame_position": np.arange(1, 17),
    "outlier_coordinates": frame_stats
})

print("=" * 65)
print("CATEGORY-WISE Y OUTLIER ANALYSIS")
print("=" * 65)
print(category_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "=" * 65)
print("TOP 20 VIDEOS BY OUTLIER PERCENTAGE")
print("=" * 65)
print(
    video_df.head(20).to_string(
        index=False, float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 65)
print("OUTLIERS BY SAMPLED FRAME POSITION")
print("=" * 65)
print(frame_df.to_string(index=False))

print("\nAnalysis completed. No feature files modified.")


===========================================================================
CELL INDEX: 190
MATCHED KEYWORDS: crop_fallback_frames, improved_extraction, crop_fallback
===========================================================================

import os
import glob
import json
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

records = []

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        features = data["features"]
        flags = data["flags"]
        indices = data["indices"]
        metadata = json.loads(str(data["metadata"].item()))

        for i in range(len(indices)):
            detected_mask = flags[i] == 1
            y = features[i, :, :, :, 1]

            detected_y = y[detected_mask]

            outliers = int(
                ((detected_y < 0) | (detected_y > 1)).sum()
            )

            records.append({
                "index": int(indices[i]),
                "category": metadata[i]["category"],
                "video_path": metadata[i]["video_path"],
                "crop_fallback_frames": metadata[i]["crop_fallback_frames"],
                "outliers": outliers,
                "outlier_percent": (
                    outliers / detected_y.size * 100
                    if detected_y.size else 0
                )
            })

df = pd.DataFrame(records)

print("=" * 60)
print("CROP FALLBACK VS Y-COORDINATE OUTLIERS")
print("=" * 60)

df["crop_group"] = np.where(
    df["crop_fallback_frames"] == 0,
    "No crop fallback",
    "Crop fallback used"
)

print("\nGROUP SUMMARY")

print(
    df.groupby("crop_group").agg(
        videos=("index", "count"),
        total_outliers=("outliers", "sum"),
        mean_outliers_per_video=("outliers", "mean"),
        mean_outlier_percent=("outlier_percent", "mean")
    ).round(4).to_string()
)

print("\nANIMALS CATEGORY")

animals = df[df["category"] == "Animals"]

print(
    animals.groupby("crop_group").agg(
        videos=("index", "count"),
        total_outliers=("outliers", "sum"),
        mean_outlier_percent=("outlier_percent", "mean")
    ).round(4).to_string()
)

print("\nAnalysis completed. No files modified.")


===========================================================================
CELL INDEX: 191
MATCHED KEYWORDS: crop_fallback_frames, crop_fallback
===========================================================================

# Uses the dataframe `df` created in Step 6.

category_crop = (
    df.groupby(["category", "crop_group"])
      .agg(
          videos=("index", "count"),
          total_outliers=("outliers", "sum"),
          mean_outlier_percent=("outlier_percent", "mean")
      )
      .reset_index()
)

print("=" * 75)
print("CATEGORY-WISE CROP FALLBACK COMPARISON")
print("=" * 75)

print(
    category_crop.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

print("\n" + "=" * 75)
print("ANIMALS — TOP AFFECTED VIDEOS")
print("=" * 75)

animals = df[df["category"] == "Animals"]

print(
    animals.sort_values("outlier_percent", ascending=False)
           .head(15)[[
               "video_path",
               "crop_fallback_frames",
               "outliers",
               "outlier_percent"
           ]]
           .to_string(
               index=False,
               float_format=lambda x: f"{x:.4f}"
           )
)

print("\nComparison completed. No files modified.")


===========================================================================
CELL INDEX: 192
MATCHED KEYWORDS: crop_fallback_frames, ambiguous_handedness_frames, handedness, improved_extraction, crop_fallback
===========================================================================

import os
import glob
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

PROJECT_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition"
)

FEATURE_DIR = os.path.join(
    PROJECT_DIR, "features/improved_extraction/full_run/batches"
)

EXTRACTED_DIR = os.path.join(
    PROJECT_DIR, "dataset/extracted"
)

TARGETS = [
    (169, "Animals/6. Mouse/MVI_3051.MOV"),
    (1409, "Animals/2. Cat/MVI_3033.MOV")
]

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

def find_video_features(target_index):
    for batch_path in batch_files:
        with np.load(batch_path, allow_pickle=True) as data:
            matches = np.where(data["indices"] == target_index)[0]

            if len(matches):
                i = int(matches[0])
                metadata = json.loads(str(data["metadata"].item()))[i]

                return (
                    data["features"][i].copy(),
                    data["flags"][i].copy(),
                    metadata
                )

    raise ValueError(f"Index {target_index} not found")

for target_index, relative_path in TARGETS:

    features, flags, metadata = find_video_features(target_index)

    video_path = os.path.join(EXTRACTED_DIR, relative_path)

    print("\n" + "=" * 65)
    print("VIDEO:", relative_path)
    print("Index:", target_index)
    print("Crop fallback frames:", metadata["crop_fallback_frames"])
    print("Ambiguous frames:", metadata["ambiguous_handedness_frames"])
    print("=" * 65)

    if not os.path.exists(video_path):
        print("VIDEO FILE NOT FOUND:", video_path)
        continue

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        print("Could not read video.")
        cap.release()
        continue

    sampled_indices = np.linspace(
        0, total_frames - 1, 16, dtype=int
    )

    # Find sampled positions containing out-of-bound Y landmarks
    candidate_frames = []

    for frame in range(16):
        frame_outliers = 0

        for hand in range(2):
            if flags[frame, hand] == 1:
                y = features[frame, hand, :, 1]
                frame_outliers += int(
                    ((y < 0) | (y > 1)).sum()
                )

        if frame_outliers:
            candidate_frames.append((frame, frame_outliers))

    candidate_frames.sort(key=lambda x: x[1], reverse=True)

    if not candidate_frames:
        print("No Y outliers found in this video.")
        cap.release()
        continue

    # Display up to four most affected sampled frames
    selected = candidate_frames[:4]

    fig, axes = plt.subplots(
        1, len(selected),
        figsize=(5 * len(selected), 6)
    )

    if len(selected) == 1:
        axes = [axes]

    for ax, (frame, outlier_count) in zip(axes, selected):

        original_frame_index = int(sampled_indices[frame])

        cap.set(cv2.CAP_PROP_POS_FRAMES, original_frame_index)
        success, image = cap.read()

        if not success:
            ax.set_title("Frame could not be read")
            ax.axis("off")
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]

        ax.imshow(image)

        for hand in range(2):
            if flags[frame, hand] != 1:
                continue

            landmarks = features[frame, hand]

            for landmark in landmarks:
                x, y, z = landmark

                px = int(x * width)
                py = int(y * height)

                is_outlier = y < 0 or y > 1

                if is_outlier:
                    # Mark outliers at the nearest visible image edge
                    visible_y = np.clip(py, 0, height - 1)

                    ax.scatter(
                        px, visible_y,
                        color="red", s=35
                    )
                else:
                    ax.scatter(
                        px, py,
                        color="lime", s=12
                    )

        ax.set_title(
            f"Sample {frame + 1}\n"
            f"Source frame {original_frame_index}\n"
            f"Y outliers: {outlier_count}"
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()

    cap.release()

print("\nVisual inspection completed. No files modified.")


===========================================================================
CELL INDEX: 193
MATCHED KEYWORDS: crop_fallback_frames, ambiguous_handedness_frames, handedness, improved_extraction, crop_fallback
===========================================================================

import os
import glob
import json
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz")))

records = []

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        flags = data["flags"]
        indices = data["indices"]
        metadata = json.loads(str(data["metadata"].item()))

        for i in range(len(indices)):
            frames_with_hands = int(
                (flags[i].sum(axis=1) >= 1).sum()
            )
            frames_with_two_hands = int(
                (flags[i].sum(axis=1) == 2).sum()
            )

            records.append({
                "index": int(indices[i]),
                "video_path": metadata[i]["video_path"],
                "category": metadata[i]["category"],
                "split": metadata[i]["split"],
                "frames_with_hands": frames_with_hands,
                "frames_with_two_hands": frames_with_two_hands,
                "ambiguous_frames": metadata[i]["ambiguous_handedness_frames"],
                "crop_fallback_frames": metadata[i]["crop_fallback_frames"]
            })

df_quality = pd.DataFrame(records)

df_quality["hand_detection_percent"] = (
    df_quality["frames_with_hands"] / 16 * 100
)

print("=" * 65)
print("VIDEO-LEVEL HAND DETECTION QUALITY")
print("=" * 65)

print("\nDETECTION COVERAGE")

print("Videos:", len(df_quality))
print("Mean detected frames:", round(df_quality["frames_with_hands"].mean(), 2))
print("Mean detection coverage (%):",
      round(df_quality["hand_detection_percent"].mean(), 2))

print("\nVIDEOS BY DETECTION COVERAGE")

print("16/16 frames:",
      int((df_quality["frames_with_hands"] == 16).sum()))

print("12–15 frames:",
      int(df_quality["frames_with_hands"].between(12, 15).sum()))

print("8–11 frames:",
      int(df_quality["frames_with_hands"].between(8, 11).sum()))

print("1–7 frames:",
      int(df_quality["frames_with_hands"].between(1, 7).sum()))

print("\nVIDEOS WITH TWO HANDS IN ALL 16 FRAMES:",
      int((df_quality["frames_with_two_hands"] == 16).sum()))

print("\nTOTAL AMBIGUOUS FRAMES:",
      int(df_quality["ambiguous_frames"].sum()))

print("\nTOTAL CROP FALLBACK FRAMES:",
      int(df_quality["crop_fallback_frames"].sum()))

print("\n20 VIDEOS WITH LOWEST HAND DETECTION")

print(
    df_quality.sort_values("frames_with_hands")
    [["video_path", "frames_with_hands",
      "frames_with_two_hands", "ambiguous_frames"]]
    .head(20).to_string(index=False)
)

print("\nAnalysis completed. No files modified.")


===========================================================================
CELL INDEX: 197
MATCHED KEYWORDS: improved_extraction
===========================================================================
import mediapipe as mp
from pathlib import Path

print("MediaPipe version:", mp.__version__)

OUT = Path(
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/checkpoints"
)

print("Checkpoint files:", len(list(OUT.glob("video_*.npz"))))

===========================================================================
CELL INDEX: 199
MATCHED KEYWORDS: handedness
===========================================================================
import pandas as pd

path = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "audit/temporal_handedness_audit_15.csv"
)

df = pd.read_csv(path)

duplicates = df[df["duplicate_two_hand_frames"] > 0]

print("Videos requiring investigation:")
print(
    duplicates[
        ["category", "video_path", "duplicate_two_hand_frames"]
    ].to_string(index=False)
)

print("\nTotal affected videos:", len(duplicates))
print(
    "Total duplicate-label frames:",
    duplicates["duplicate_two_hand_frames"].sum()
)

===========================================================================
CELL INDEX: 200
MATCHED KEYWORDS: HandLandmarker, handedness, min_hand_detection_confidence
===========================================================================
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

ROOT = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
MANIFEST = f"{ROOT}/audit/full_include_working_manifest.csv"
VIDEO_ROOT = f"{ROOT}/dataset/extracted"
MODEL_PATH = "/tmp/hand_landmarker.task"

OUTPUT = f"{ROOT}/audit/duplicate_handedness_frames.csv"

df = pd.read_csv(MANIFEST)

targets = {
    "Adjectives/9. Nice/MVI_9590.MOV",
    "Clothes/45. Pocket/MVI_5178.MOV",
    "Electronics/56. Laptop/MVI_4547.MOV",
    "Jobs/93. Soldier/MVI_8894.MP4",
    "Places/25. Office/MVI_3460.MOV"
}

selected = df[df["video_path"].isin(targets)].copy()

assert len(selected) == 5, "Expected exactly five videos"

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3
)

records = []

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

    for _, row in selected.iterrows():

        video_path = str(row["video_path"])
        full_path = os.path.join(VIDEO_ROOT, video_path)

        cap = cv2.VideoCapture(full_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        indices = np.linspace(
            0, total - 1, 16, dtype=int
        )

        for sample_number, frame_index in enumerate(indices):

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            success, frame = cap.read()

            if not success:
                continue

            h, w = frame.shape[:2]

            x1 = int(w * 0.175)
            x2 = int(w * 0.825)

            crop = frame[:, x1:x2]
            scale = 640 / max(crop.shape[:2])

            resized = cv2.resize(
                crop, None,
                fx=scale, fy=scale,
                interpolation=cv2.INTER_CUBIC
            )

            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect(mp_image)

            if len(result.handedness) != 2:
                continue

            labels = [
                item[0].category_name.lower()
                for item in result.handedness
            ]

            if labels[0] == labels[1]:

                confidences = [
                    float(item[0].score)
                    for item in result.handedness
                ]

                records.append({
                    "category": row["category"],
                    "video_path": video_path,
                    "sample_number": sample_number,
                    "frame_index": int(frame_index),
                    "label_1": labels[0],
                    "label_2": labels[1],
                    "confidence_1": confidences[0],
                    "confidence_2": confidences[1]
                })

                print(
                    row["category"],
                    "| frame:", frame_index,
                    "| labels:", labels,
                    "| confidence:", confidences
                )

        cap.release()

audit = pd.DataFrame(records)
audit.to_csv(OUTPUT, index=False)

print("\nDuplicate frames found:", len(audit))
print("Saved:", OUTPUT)

print("\nFull audit:")
print(audit.to_string(index=False))

===========================================================================
CELL INDEX: 203
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import json
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(
    glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz"))
)

if not batch_files:
    raise FileNotFoundError("No extraction batch files found.")

records = []

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        flags = data["flags"]
        indices = data["indices"]
        metadata = json.loads(str(data["metadata"].item()))

        for i in range(len(indices)):
            detected_frames = int(
                (flags[i].sum(axis=1) >= 1).sum()
            )

            records.append({
                "index": int(indices[i]),
                "category": metadata[i]["category"],
                "split": metadata[i]["split"],
                "frames_with_hands": detected_frames
            })

df_quality = pd.DataFrame(records)

# Calculate split-level statistics
split_report = (
    df_quality.groupby("split")
    .agg(
        total_videos=("index", "count"),
        low_coverage_videos=(
            "frames_with_hands",
            lambda x: (x < 12).sum()
        ),
        critical_videos=(
            "frames_with_hands",
            lambda x: (x < 8).sum()
        ),
        perfect_detection_videos=(
            "frames_with_hands",
            lambda x: (x == 16).sum()
        )
    )
    .reset_index()
)

split_report["low_coverage_percent"] = (
    split_report["low_coverage_videos"]
    / split_report["total_videos"] * 100
)

split_report["critical_percent"] = (
    split_report["critical_videos"]
    / split_report["total_videos"] * 100
)

print("=" * 65)
print("DATASET SPLIT QUALITY COMPARISON")
print("=" * 65)

print(
    split_report.to_string(
        index=False,
        float_format=lambda x: f"{x:.2f}"
    )
)

print("\nOVERALL DATASET")
print("Total videos:", len(df_quality))
print("Low coverage:", int((df_quality["frames_with_hands"] < 12).sum()))
print("Critical coverage:", int((df_quality["frames_with_hands"] < 8).sum()))

print("\nAnalysis completed. No files modified.")


===========================================================================
CELL INDEX: 204
MATCHED KEYWORDS: ambiguous_handedness_frames, handedness, improved_extraction
===========================================================================

import os
import glob
import json
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(
    glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz"))
)

records = []

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        indices = data["indices"]
        metadata = json.loads(str(data["metadata"].item()))

        for i in range(len(indices)):
            records.append({
                "index": int(indices[i]),
                "video_path": metadata[i]["video_path"],
                "category": metadata[i]["category"],
                "split": metadata[i]["split"],
                "ambiguous_frames": int(
                    metadata[i]["ambiguous_handedness_frames"]
                )
            })

df_ambiguity = pd.DataFrame(records)

print("=" * 65)
print("HANDEDNESS AMBIGUITY AUDIT")
print("=" * 65)

print("\nOVERALL STATISTICS")
print("Total videos:", len(df_ambiguity))
print("Total ambiguous frames:",
      df_ambiguity["ambiguous_frames"].sum())
print("Videos with ambiguity:",
      (df_ambiguity["ambiguous_frames"] > 0).sum())
print("Videos without ambiguity:",
      (df_ambiguity["ambiguous_frames"] == 0).sum())

print("\nCATEGORY-WISE AMBIGUITY")

category_report = (
    df_ambiguity.groupby("category")
    .agg(
        videos=("index", "count"),
        videos_with_ambiguity=(
            "ambiguous_frames",
            lambda x: (x > 0).sum()
        ),
        total_ambiguous_frames=("ambiguous_frames", "sum")
    )
    .reset_index()
    .sort_values("total_ambiguous_frames", ascending=False)
)

print(category_report.to_string(index=False))

print("\nSPLIT-WISE AMBIGUITY")

split_report = (
    df_ambiguity.groupby("split")
    .agg(
        videos=("index", "count"),
        videos_with_ambiguity=(
            "ambiguous_frames",
            lambda x: (x > 0).sum()
        ),
        total_ambiguous_frames=("ambiguous_frames", "sum")
    )
    .reset_index()
)

print(split_report.to_string(index=False))

print("\nTOP 20 VIDEOS WITH MOST AMBIGUOUS FRAMES")

print(
    df_ambiguity.sort_values(
        "ambiguous_frames", ascending=False
    )[
        ["video_path", "split", "ambiguous_frames"]
    ].head(20).to_string(index=False)
)

print("\nAudit completed. No files modified.")


===========================================================================
CELL INDEX: 205
MATCHED KEYWORDS: improved_extraction
===========================================================================

import os
import glob
import numpy as np
import pandas as pd

FEATURE_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition/"
    "features/improved_extraction/full_run/batches"
)

batch_files = sorted(
    glob.glob(os.path.join(FEATURE_DIR, "batch_*.npz"))
)

total_frames = 0
slot0_detected = 0
slot1_detected = 0
both_detected = 0
neither_detected = 0

slot_patterns = {
    "Left only": 0,
    "Right only": 0,
    "Both hands": 0,
    "No hands": 0
}

for batch_path in batch_files:
    with np.load(batch_path, allow_pickle=True) as data:
        flags = data["flags"]

        total_frames += flags.shape[0] * flags.shape[1]

        slot0_detected += int((flags[:, :, 0] == 1).sum())
        slot1_detected += int((flags[:, :, 1] == 1).sum())

        both = (flags[:, :, 0] == 1) & (flags[:, :, 1] == 1)
        left_only = (flags[:, :, 0] == 1) & (flags[:, :, 1] == 0)
        right_only = (flags[:, :, 0] == 0) & (flags[:, :, 1] == 1)
        neither = (flags[:, :, 0] == 0) & (flags[:, :, 1] == 0)

        both_detected += int(both.sum())
        neither_detected += int(neither.sum())

        slot_patterns["Both hands"] += int(both.sum())
        slot_patterns["Left only"] += int(left_only.sum())
        slot_patterns["Right only"] += int(right_only.sum())
        slot_patterns["No hands"] += int(neither.sum())

print("=" * 65)
print("HAND SLOT CONSISTENCY — DETECTION AUDIT")
print("=" * 65)

print("Total sampled frames:", total_frames)

print("\nSLOT DETECTION")
print("Slot 0 detections:", slot0_detected)
print("Slot 1 detections:", slot1_detected)

print("\nFRAME PATTERNS")

for pattern, count in slot_patterns.items():
    print(
        f"{pattern}: {count} "
        f"({count / total_frames * 100:.2f}%)"
    )

print("\nTOTAL FRAMES WITH BOTH HANDS:", both_detected)
print("TOTAL FRAMES WITH NO HANDS:", neither_detected)

print("\nAudit completed. No files modified.")


===========================================================================
CELL INDEX: 206
MATCHED KEYWORDS: HandLandmarker, crop_fallback_frames, ambiguous_handedness_frames, handedness, min_hand_detection_confidence, improved_extraction, crop_fallback
===========================================================================

import os
import re

PROJECT_DIR = (
    "/content/gdrive/MyDrive/ISL_Sign_Recognition"
)

# Distinctive identifiers from our improved extraction
KEYWORDS = [
    "HandLandmarker",
    "crop_fallback_frames",
    "ambiguous_handedness_frames",
    "improved_extraction",
    "min_hand_detection_confidence",
    "crop_fallback"
]

# Avoid scanning large dataset and feature directories
SKIP_DIRS = {
    "raw",
    "extracted",
    "batches",
    "checkpoints",
    ".git",
    "__pycache__"
}

EXTENSIONS = {".py", ".ipynb"}

matches = []

print("=" * 65)
print("SEARCHING GOOGLE DRIVE FOR IMPROVED EXTRACTION CODE")
print("=" * 65)

for root, dirs, files in os.walk(PROJECT_DIR):

    dirs[:] = [
        d for d in dirs
        if d not in SKIP_DIRS
    ]

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()

        if ext not in EXTENSIONS:
            continue

        filepath = os.path.join(root, filename)

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            found = [
                keyword for keyword in KEYWORDS
                if keyword.lower() in content.lower()
            ]

            if found:
                matches.append({
                    "path": filepath,
                    "keywords": found,
                    "size_kb": round(os.path.getsize(filepath) / 1024, 2)
                })

        except Exception as e:
            print("Could not inspect:", filepath, str(e))

print("\nFILES MATCHING EXTRACTION CODE:")

if matches:
    for i, item in enumerate(matches, 1):
        print(f"\n{i}. {item['path']}")
        print("   Size:", item["size_kb"], "KB")
        print("   Matched:", ", ".join(item["keywords"]))
else:
    print("No matching files found.")

print("\nTotal matching files:", len(matches))


===========================================================================
CELL INDEX: 207
MATCHED KEYWORDS: HandLandmarker, crop_fallback_frames, ambiguous_handedness_frames, detect_for_video, handedness, min_hand_detection_confidence, improved_extraction, crop_fallback
===========================================================================

import os
import re

DRIVE_ROOT = "/content/gdrive/MyDrive"

KEYWORDS = [
    "crop_fallback_frames",
    "ambiguous_handedness_frames",
    "HandLandmarker",
    "min_hand_detection_confidence",
    "improved_extraction",
    "crop_fallback",
    "landmarker.detect_for_video"
]

EXTENSIONS = {".py", ".ipynb"}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".cache"
}

results = []

print("=" * 65)
print("FULL GOOGLE DRIVE EXTRACTION CODE SEARCH")
print("=" * 65)

for root, dirs, files in os.walk(DRIVE_ROOT):

    dirs[:] = [
        d for d in dirs
        if d not in SKIP_DIRS
    ]

    for filename in files:
        if os.path.splitext(filename)[1].lower() not in EXTENSIONS:
            continue

        filepath = os.path.join(root, filename)

        try:
            if os.path.getsize(filepath) > 50 * 1024 * 1024:
                continue

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            matched = [
                k for k in KEYWORDS
                if k.lower() in content.lower()
            ]

            if matched:
                results.append((filepath, matched))

        except Exception:
            continue

print("\nMATCHING FILES:", len(results))

for i, (path, keywords) in enumerate(results, 1):
    print(f"\n{i}. {path}")
    print("   Matches:", ", ".join(keywords))

print("\nSEARCH FINISHED.")


===========================================================================
CELL INDEX: 209
MATCHED KEYWORDS: HandLandmarker, ambiguous_handedness_frames, handedness
===========================================================================

import os
import time
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = "/content/gdrive/MyDrive/ISL_Sign_Recognition"
VIDEO_ROOT = os.path.join(BASE_DIR, "dataset", "extracted")
MANIFEST_PATH = os.path.join(
    BASE_DIR, "audit", "full_include_working_manifest.csv"
)
MODEL_PATH = "/tmp/hand_landmarker.task"
OUTPUT_DIR = os.path.join(BASE_DIR, "features", "full_extraction")
STATUS_PATH = os.path.join(OUTPUT_DIR, "status.csv")

N_FRAMES = 16
N_HANDS = 2
N_LANDMARKS = 21
N_COORDS = 3
BATCH_SIZE = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.isfile(MANIFEST_PATH), f"Manifest missing: {MANIFEST_PATH}"
assert os.path.isfile(MODEL_PATH), f"Model missing: {MODEL_PATH}"
assert os.path.isdir(VIDEO_ROOT), f"Video directory missing: {VIDEO_ROOT}"

manifest = pd.read_csv(MANIFEST_PATH).reset_index(drop=True)

assert len(manifest) == 4257, f"Unexpected manifest size: {len(manifest)}"
assert manifest["video_path"].is_unique, "Duplicate video paths found"

status_columns = [
    "index", "video_path", "status", "batch_file",
    "frames_with_any_hand", "frames_with_two_hands",
    "ambiguous_handedness_frames", "error"
]

# ============================================================
# 2. LOAD AND VERIFY EXISTING CHECKPOINTS
# ============================================================

if os.path.isfile(STATUS_PATH):
    old_status = pd.read_csv(STATUS_PATH)
else:
    old_status = pd.DataFrame(columns=status_columns)

verified = {}
batch_cache = {}

for _, record in old_status.iterrows():
    if record.get("status") != "success":
        continue

    try:
        idx = int(record["index"])
        filename = record["batch_file"]

        if not isinstance(filename, str) or not filename:
            continue

        batch_path = os.path.join(OUTPUT_DIR, filename)
        if not os.path.isfile(batch_path):
            continue

        if filename not in batch_cache:
            with np.load(batch_path, allow_pickle=True) as z:
                batch_cache[filename] = z["indices"].astype(int).copy()

        if idx in batch_cache[filename]:
            verified[idx] = filename

    except Exception:
        continue

# Keep verified successes only; all others will be retried.
status_records = {}
for idx, filename in verified.items():
    row = manifest.iloc[idx]
    old = old_status[
        (old_status["index"].astype(int) == idx)
        & (old_status["status"] == "success")
    ]
    old_record = old.iloc[-1].to_dict() if not old.empty else {}

    status_records[idx] = {
        "index": idx,
        "video_path": row["video_path"],
        "status": "success",
        "batch_file": filename,
        "frames_with_any_hand": old_record.get("frames_with_any_hand", 0),
        "frames_with_two_hands": old_record.get("frames_with_two_hands", 0),
        "ambiguous_handedness_frames": old_record.get(
            "ambiguous_handedness_frames", 0
        ),
        "error": ""
    }

remaining = [i for i in range(len(manifest)) if i not in verified]

def write_status():
    df = pd.DataFrame(
        [status_records[i] for i in sorted(status_records)],
        columns=status_columns
    )
    temp_path = STATUS_PATH + ".tmp"
    df.to_csv(temp_path, index=False)
    os.replace(temp_path, STATUS_PATH)

print("Total videos:", len(manifest))
print("Verified completed:", len(verified))
print("Remaining:", len(remaining))
print("Frames per video:", N_FRAMES)
print("Batch size:", BATCH_SIZE)

# ============================================================
# 3. INITIALIZE MEDIAPIPE
# ============================================================

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=N_HANDS
)

# ============================================================
# 4. EXTRACT 16 FRAMES PER VIDEO
# ============================================================

with HandLandmarker.create_from_options(options) as landmarker:

    for start in range(0, len(remaining), BATCH_SIZE):
        indices = remaining[start:start + BATCH_SIZE]

        features_batch = []
        flags_batch = []
        metadata_batch = []
        successful_rows = []
        failed_rows = []
        batch_start_time = time.time()

        for idx in indices:
            row = manifest.iloc[idx]
            video_path = os.path.join(VIDEO_ROOT, row["video_path"])
            cap = None

            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError("Could not open video")

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames < 1:
                    raise RuntimeError("No readable frames reported")

                frame_indices = np.linspace(
                    0, total_frames - 1, N_FRAMES
                ).round().astype(int)

                sequence = np.zeros(
                    (N_FRAMES, N_HANDS, N_LANDMARKS, N_COORDS),
                    dtype=np.float32
                )
                flags = np.zeros(
                    (N_FRAMES, N_HANDS),
                    dtype=np.uint8
                )
                ambiguous = 0

                for t, frame_index in enumerate(frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    ok, frame = cap.read()

                    if not ok:
                        raise RuntimeError(
                            f"Unable to read frame {frame_index}"
                        )

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        )
                    )

                    hands = result.hand_landmarks[:N_HANDS]

                    # Handedness categories come from result.handedness,
                    # not from the landmark objects.
                    labels = []
                    for j in range(len(hands)):
                        categories = (
                            result.handedness[j]
                            if j < len(result.handedness) else []
                        )
                        labels.append(
                            categories[0].category_name.lower()
                            if categories else "unknown"
                        )

                    if len(hands) == 2:
                        if len(labels) == 2 and set(labels) == {"left", "right"}:
                            slots = [
                                (0 if labels[j] == "left" else 1, hands[j])
                                for j in range(2)
                            ]
                        else:
                            ambiguous += 1
                            slots = [(0, hands[0]), (1, hands[1])]

                    elif len(hands) == 1:
                        label = labels[0] if labels else "unknown"
                        slot = (
                            0 if label == "left"
                            else 1 if label == "right"
                            else 0
                        )
                        slots = [(slot, hands[0])]
                    else:
                        slots = []

                    for slot, hand in slots:
                        sequence[t, slot] = np.asarray(
                            [[lm.x, lm.y, lm.z] for lm in hand],
                            dtype=np.float32
                        )
                        flags[t, slot] = 1

                cap.release()
                cap = None

                any_hand = int(flags.any(axis=1).sum())
                two_hands = int(flags.all(axis=1).sum())

                features_batch.append(sequence)
                flags_batch.append(flags)

                metadata_batch.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "category": row["category"],
                    "label": row["label"],
                    "split": row["split"],
                    "total_video_frames": total_frames,
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous
                })

                successful_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "frames_with_any_hand": any_hand,
                    "frames_with_two_hands": two_hands,
                    "ambiguous_handedness_frames": ambiguous
                })

            except Exception as exc:
                failed_rows.append({
                    "index": idx,
                    "video_path": row["video_path"],
                    "error": str(exc)[:300]
                })

            finally:
                if cap is not None:
                    cap.release()

        # Save features first. Only then mark videos as successful.
        if features_batch:
            batch_number = 0
            while os.path.exists(
                os.path.join(OUTPUT_DIR, f"batch_{batch_number:04d}.npz")
            ):
                batch_number += 1

            batch_filename = f"batch_{batch_number:04d}.npz"
            batch_path = os.path.join(OUTPUT_DIR, batch_filename)
            temp_batch_path = batch_path + ".tmp.npz"

            np.savez_compressed(
                temp_batch_path,
                indices=np.array(
                    [m["index"] for m in metadata_batch],
                    dtype=np.int32
                ),
                features=np.stack(features_batch),
                flags=np.stack(flags_batch),
                metadata=pd.DataFrame(metadata_batch).to_json(
                    orient="records"
                )
            )
            os.replace(temp_batch_path, batch_path)

            for record in successful_rows:
                idx = record["index"]
                status_records[idx] = {
                    "index": idx,
                    "video_path": record["video_path"],
                    "status": "success",
                    "batch_file": batch_filename,
                    "frames_with_any_hand": record["frames_with_any_hand"],
                    "frames_with_two_hands": record["frames_with_two_hands"],
                    "ambiguous_handedness_frames": record[
                        "ambiguous_handedness_frames"
                    ],
                    "error": ""
                }

        for record in failed_rows:
            idx = record["index"]
            status_records[idx] = {
                "index": idx,
                "video_path": record["video_path"],
                "status": "failed",
                "batch_file": "",
                "frames_with_any_hand": 0,
                "frames_with_two_hands": 0,
                "ambiguous_handedness_frames": 0,
                "error": record["error"]
            }

        write_status()

        print(
            f"Run progress: {min(start + len(indices), len(remaining))}"
            f"/{len(remaining)} | "
            f"batch success: {len(successful_rows)} | "
            f"batch failed: {len(failed_rows)} | "
            f"elapsed: {time.time() - batch_start_time:.1f}s",
            flush=True
        )

# ============================================================
# 5. FINAL SUMMARY
# ============================================================

final_status = pd.read_csv(STATUS_PATH)

print("\nEXTRACTION RUN FINISHED")
print("Successful videos:", int((final_status["status"] == "success").sum()))
print("Failed videos:", int((final_status["status"] == "failed").sum()))
print("Status file:", STATUS_PATH)
print("Feature batches:", OUTPUT_DIR)

RECOVERY COMPLETE
Notebook was only read. No changes made.

```