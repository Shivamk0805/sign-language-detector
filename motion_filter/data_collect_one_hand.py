import cv2
import mediapipe as mp
import csv
import os
import time
import numpy as np

# ==============================
# CONFIG
# ==============================
SIGNS = ["fine","like","need","I_or_Me","My_OR_Mine","you","your","where","why_OR_because"]
DATA_DIR = "data/one_hand"
RECORD_TIME = 20   # seconds
PAUSE_TIME = 10    # seconds
MAX_HANDS = 1

os.makedirs(DATA_DIR, exist_ok=True)

# ==============================
# Normalization function
# ==============================
def normalize_hand_landmarks(hand_landmarks):
    """Make hand coords relative to wrist and scale-invariant."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = coords[0]  # landmark 0 = wrist
    rel_coords = coords - wrist

    # Scale by distance wrist->middle finger MCP (landmark 9)
    ref_dist = np.linalg.norm(coords[0][:2] - coords[9][:2])
    if ref_dist > 1e-6:
        rel_coords /= ref_dist

    return rel_coords.flatten().tolist()

def extract_body_anchors(pose_landmarks):
    """Extract face + torso anchors (absolute, normalized by torso size)."""
    if not pose_landmarks:
        return [0.0] * (8 * 3)  # 8 anchors × 3 coords

    lm = pose_landmarks.landmark

    # Chosen indices (mediapipe holistic/pose)
    anchor_idx = {
        "ear_L": 234, "ear_R": 454,    # approx from face mesh (use if available)
        "chin": 152,
        "lips": 13,
        "forehead": 10,
        "shoulder_L": 11,
        "shoulder_R": 12,
        "chest": 33  # approximate: nose base / mid chest
    }

    anchors = []
    for key, idx in anchor_idx.items():
        if idx < len(lm):
            anchors.extend([lm[idx].x, lm[idx].y, lm[idx].z])
        else:
            anchors.extend([0.0, 0.0, 0.0])

    # Normalize by shoulder distance (scale-invariant)
    if anchor_idx["shoulder_L"] < len(lm) and anchor_idx["shoulder_R"] < len(lm):
        shoulder_L = np.array([lm[anchor_idx["shoulder_L"]].x, lm[anchor_idx["shoulder_L"]].y])
        shoulder_R = np.array([lm[anchor_idx["shoulder_R"]].x, lm[anchor_idx["shoulder_R"]].y])
        shoulder_dist = np.linalg.norm(shoulder_L - shoulder_R)
        if shoulder_dist > 1e-6:
            anchors = [a / shoulder_dist for a in anchors]

    return anchors

# ==============================
# Helper: save one sign
# ==============================
def record_sign(sign_name):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )
    pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)

    csv_path = os.path.join(DATA_DIR, f"{sign_name}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)

        # header: frame + hand landmarks + body anchors
        hand_header = [f"hand_{i}_{axis}" for i in range(21) for axis in ("x", "y", "z")]
        anchor_header = [f"{name}_{axis}" for name in [
            "ear_L","ear_R","chin","lips","forehead","shoulder_L","shoulder_R","chest"
        ] for axis in ("x", "y", "z")]
        header = ["frame", "hand"] + hand_header + anchor_header
        writer.writerow(header)

        start_time = time.time()
        frame_count = 0

        while (time.time() - start_time) < RECORD_TIME:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results_hands = hands.process(rgb)
            results_pose = pose.process(rgb)

            if results_hands.multi_hand_landmarks:
                for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
                    row = [frame_count, hand_idx]

                    # Hand landmarks (relative + scaled)
                    row.extend(normalize_hand_landmarks(hand_landmarks.landmark))

                    # Body anchors
                    row.extend(extract_body_anchors(results_pose.pose_landmarks))

                    writer.writerow(row)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(frame, f"Recording: {sign_name}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv2.imshow("Dataset Recorder", frame)

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()
    print(f"✅ Saved {sign_name} to {csv_path}")

# ==============================
# MAIN LOOP
# ==============================
def main():
    for sign in SIGNS:
        print(f"\n▶ Get ready to perform: {sign.upper()}")
        time.sleep(3)
        record_sign(sign)

        if sign != SIGNS[-1]:
            print(f"⏸️ Pausing {PAUSE_TIME}s before next sign...")
            time.sleep(PAUSE_TIME)

    print("\n🎉 Recording complete! All CSVs saved in 'data/one_hand/'")

if __name__ == "__main__":
    main()

