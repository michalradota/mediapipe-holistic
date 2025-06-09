import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_image_holistic(image_path):
    """
    Processes an image using MediaPipe Holistic and draws landmarks.

    Args:
        image_path (str): The path to the input image.

    Returns:
        np.ndarray: The image with landmarks drawn, or None if processing fails.
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find holistic landmarks
        with mp_holistic.Holistic(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as holistic:
            
            results = holistic.process(image_rgb)

        # Draw landmarks
        annotated_image = image.copy()

        # 1. Draw face landmarks
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.face_landmarks,
            connections=mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        # 2. Draw pose landmarks
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.pose_landmarks,
            connections=mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 3. Draw left hand landmarks
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

        # 4. Draw right hand landmarks
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
        # 5. Draw segmentation mask (optional)
        if results.segmentation_mask is not None:
            segmentation_mask = results.segmentation_mask
            condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (192, 192, 192) # Gray background
            annotated_image = np.where(condition, annotated_image, bg_image)


        return annotated_image

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    # Replace "path/to/your/image.jpg" with the actual path to your image
    input_image_path = "test_images/test2.png" 
    
    output_image = process_image_holistic(input_image_path)

    if output_image is not None:
        cv2.imshow("MediaPipe Holistic Output", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # Optionally, save the image
        # cv2.imwrite("holistic_output.png", output_image)
        # print("Processed image saved as holistic_output.png")
    else:
        print("Image processing failed.")
