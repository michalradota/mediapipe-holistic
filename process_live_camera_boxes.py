import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

def draw_bounding_box(image, landmarks, label, color=(0,255,0)):
    h, w, _ = image.shape
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def draw_pose_part_box(image, landmarks, indices, label, color=(0,255,0)):
    h, w, _ = image.shape
    points = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        points.append((int(lm.x * w), int(lm.y * h)))
    x_min = min(p[0] for p in points)
    x_max = max(p[0] for p in points)
    y_min = min(p[1] for p in points)
    y_max = max(p[1] for p in points)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def webcam_feed():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam")
        return

    face_spec = [
        mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
    ]
    right_hand_spec = [
        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
    ]
    left_hand_spec = [
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
    ]
    pose_spec = [
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    ]

    torso_indices = [
        mp_pose.PoseLandmark.LEFT_SHOULDER.value,
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_HIP.value
    ]

    left_leg_indices = [
        mp_pose.PoseLandmark.LEFT_HIP.value,
        mp_pose.PoseLandmark.LEFT_KNEE.value,
        mp_pose.PoseLandmark.LEFT_ANKLE.value
    ]

    right_leg_indices = [
        mp_pose.PoseLandmark.RIGHT_HIP.value,
        mp_pose.PoseLandmark.RIGHT_KNEE.value,
        mp_pose.PoseLandmark.RIGHT_ANKLE.value
    ]

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                    face_spec[0], face_spec[1]
                )
                draw_bounding_box(image, results.face_landmarks, 'Head', color=(0,255,0))

            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.right_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    right_hand_spec[0], right_hand_spec[1]
                )
                draw_bounding_box(image, results.right_hand_landmarks, 'Right Hand', color=(0,0,255))

            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.left_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    left_hand_spec[0], left_hand_spec[1]
                )
                draw_bounding_box(image, results.left_hand_landmarks, 'Left Hand', color=(255,0,0))

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    pose_spec[0], pose_spec[1]
                )
                draw_pose_part_box(image, results.pose_landmarks, torso_indices, 'Torso', color=(255,165,0))
                draw_pose_part_box(image, results.pose_landmarks, left_leg_indices, 'Left Leg', color=(0,128,255))
                draw_pose_part_box(image, results.pose_landmarks, right_leg_indices, 'Right Leg', color=(0,255,128))

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_feed()
