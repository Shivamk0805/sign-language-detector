import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ==============================
# CONFIG
# ==============================
SIGNS = ["finish","go","right","want","car","cook","how","what","when","name"]
DATA_DIR = "two_hand_v2"
LOOPS = 20           # number of clips per sign
CLIP_LEN = 4        # seconds per clip
BREAK = 5           # seconds pause between clips
FPS = 30

os.makedirs(DATA_DIR, exist_ok=True)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.6, min_tracking_confidence=0.6)
face = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1,
                        min_detection_confidence=0.6, min_tracking_confidence=0.6)

# Custom face indices
FACE_IDXS = {
    "nose": 1,
    "left_ear": 234,
    "right_ear": 454,
    "chin": 152,
    "forehead": 10
}

# ==============================
# Feature extractor
# ==============================
def extract_features(results_hands, results_face):
    features = []

    # Hand landmarks
    if results_hands.multi_hand_landmarks:
        hands_data = []
        for hand in results_hands.multi_hand_landmarks:
            coords = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            hands_data.append(coords)

        # If two hands detected
        if len(hands_data) == 2:
            left_hand = hands_data[0]
            right_hand = hands_data[1]

            # Relative to left wrist (landmark 0)
            origin = left_hand[0]
            left_rel = left_hand - origin
            right_rel = right_hand - origin

            features.extend(left_rel.flatten())
            features.extend(right_rel.flatten())

        # If only one hand detected → pad with zeros
        else:
            origin = hands_data[0][0]
            rel = hands_data[0] - origin
            features.extend(rel.flatten())
            features.extend(np.zeros(21 * 3))  # pad for missing hand
    else:
        features.extend(np.zeros(21 * 3 * 2))

    # Face landmarks (nose, ears, chin, forehead)
    if results_face.multi_face_landmarks:
        face_points = results_face.multi_face_landmarks[0].landmark
        for key in FACE_IDXS:
            lm = face_points[FACE_IDXS[key]]
            features.extend([lm.x, lm.y, lm.z])
    else:
        features.extend([0.0, 0.0, 0.0] * len(FACE_IDXS))

    return np.array(features, dtype=np.float32)


# ==============================
# Visual debug: draw only selected points
# ==============================
def draw_debug(frame, results_hands, results_face):
    h, w, _ = frame.shape

    if results_hands.multi_hand_landmarks:
        for hand in results_hands.multi_hand_landmarks:
            for lm in hand.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

    if results_face.multi_face_landmarks:
        face_points = results_face.multi_face_landmarks[0].landmark
        for key in FACE_IDXS:
            lm = face_points[FACE_IDXS[key]]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)


# ==============================
# Recording function
# ==============================
def record_clip(sign, loop_id):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not available")
        return

    frames = []
    start_time = time.time()
    frame_count = 0

    while (time.time() - start_time) < CLIP_LEN:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results_hands = hands.process(rgb)
        results_face = face.process(rgb)

        # Extract features
        feat = extract_features(results_hands, results_face)
        frames.append(feat)

        # Draw debug points
        draw_debug(frame, results_hands, results_face)

        # Display info
        cv2.putText(frame, f"Recording {sign} | Clip {loop_id+1}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        cv2.imshow("Recorder", frame)
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    from datetime import datetime
    
    frames = np.array(frames, dtype=np.float32)
    save_dir = os.path.join(DATA_DIR, sign)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"clip_{timestamp}.npy"
    np.save(os.path.join(save_dir, filename), frames)
    print(f"✅ Saved {sign}/{filename} ({frames.shape})")

# ==============================
# MAIN LOOP
# ==============================
def main():
    for sign in SIGNS:
        for i in range(LOOPS):
            print(f"\n▶ Perform: {sign.upper()} (clip {i+1}/{LOOPS})")
            time.sleep(3)
            record_clip(sign, i)

            if i != LOOPS - 1:
                print(f"⏸️ Rest {BREAK}s...")
                time.sleep(BREAK)

    hands.close()
    face.close()
    print("\n🎉 Done recording all signs!")


if __name__ == "__main__":
    main()

