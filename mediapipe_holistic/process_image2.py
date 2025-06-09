import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
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
    cv2.putText(image, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

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
    cv2.putText(image, label, (x_min, y_min - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def process_image_holistic(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_holistic.Holistic(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as holistic:

        results = holistic.process(image_rgb)

    annotated_image = image.copy()

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

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            face_spec[0], face_spec[1])
        draw_bounding_box(annotated_image, results.face_landmarks, 'Head', color=(0,255,0))

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.right_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            right_hand_spec[0], right_hand_spec[1])
        draw_bounding_box(annotated_image, results.right_hand_landmarks, 'Right Hand', color=(0,0,255))

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.left_hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            left_hand_spec[0], left_hand_spec[1])
        draw_bounding_box(annotated_image, results.left_hand_landmarks, 'Left Hand', color=(255,0,0))

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            pose_spec[0], pose_spec[1])
        draw_pose_part_box(annotated_image, results.pose_landmarks, torso_indices, 'Torso', color=(255,165,0))
        draw_pose_part_box(annotated_image, results.pose_landmarks, left_leg_indices, 'Left Leg', color=(0,128,255))
        draw_pose_part_box(annotated_image, results.pose_landmarks, right_leg_indices, 'Right Leg', color=(0,255,128))

    # Opcjonalnie: maska segmentacji, jeśli jest dostępna
    if results.segmentation_mask is not None:
        seg_mask = results.segmentation_mask
        condition = np.stack((seg_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        bg_image[:] = (192, 192, 192)
        annotated_image = np.where(condition, annotated_image, bg_image)

    return annotated_image

if __name__ == "__main__":
    input_image_path = "test_images/test3.png"

    output_image = process_image_holistic(input_image_path)

    if output_image is not None:
        cv2.imshow("MediaPipe Holistic Output", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Możesz też zapisać wynik:
        # cv2.imwrite("holistic_output_with_boxes.png", output_image)
    else:
        print("Image processing failed.")
