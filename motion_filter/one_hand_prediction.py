import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import json

# ======================
# CONFIG
# ======================
MODEL_PATH = "one_hand_motion_model.pth"
META_PATH = "one_hand_motion_meta.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_WINDOW = "ASL Motion Prediction"
MAX_HANDS = 1

# ======================
# MODEL
# ======================
class LandmarkMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ======================
# LOAD MODEL + META
# ======================
with open(META_PATH, "r") as f:
    meta = json.load(f)

CLASSES = meta["classes"]
INPUT_SIZE = meta["input_size"]

model = LandmarkMLP(input_size=INPUT_SIZE, num_classes=len(CLASSES)).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"✅ Loaded model with {len(CLASSES)} classes: {CLASSES}")

# ======================
# FEATURE FUNCTIONS
# ======================
def normalize_hand_landmarks(hand_landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = coords[0]
    rel_coords = coords - wrist
    ref_dist = np.linalg.norm(coords[0][:2] - coords[9][:2])
    if ref_dist > 1e-6:
        rel_coords /= ref_dist
    return rel_coords.flatten().tolist()

def extract_body_anchors(pose_landmarks):
    if not pose_landmarks:
        return [0.0] * (8 * 3)

    lm = pose_landmarks.landmark
    anchor_idx = {
        "ear_L": 234, "ear_R": 454,
        "chin": 152, "lips": 13,
        "forehead": 10,
        "shoulder_L": 11, "shoulder_R": 12,
        "chest": 33
    }

    anchors = []
    for key, idx in anchor_idx.items():
        if idx < len(lm):
            anchors.extend([lm[idx].x, lm[idx].y, lm[idx].z])
        else:
            anchors.extend([0.0, 0.0, 0.0])

    if anchor_idx["shoulder_L"] < len(lm) and anchor_idx["shoulder_R"] < len(lm):
        shoulder_L = np.array([lm[anchor_idx["shoulder_L"]].x, lm[anchor_idx["shoulder_L"]].y])
        shoulder_R = np.array([lm[anchor_idx["shoulder_R"]].x, lm[anchor_idx["shoulder_R"]].y])
        shoulder_dist = np.linalg.norm(shoulder_L - shoulder_R)
        if shoulder_dist > 1e-6:
            anchors = [a / shoulder_dist for a in anchors]

    return anchors

# ======================
# LIVE PREDICTION
# ======================
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

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open camera")
    exit()

print("▶ Live prediction started. Press [q] to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results_hands = hands.process(rgb)
    results_pose = pose.process(rgb)

    pred_label = None
    if results_hands.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            coords = normalize_hand_landmarks(hand_landmarks.landmark)
            anchors = extract_body_anchors(results_pose.pose_landmarks)
            features = torch.tensor(coords + anchors, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = model(features)
                _, pred = torch.max(outputs, 1)
                pred_label = CLASSES[pred.item()]

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if pred_label:
        cv2.putText(frame, f"Prediction: {pred_label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow(IMG_WINDOW, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
pose.close()

