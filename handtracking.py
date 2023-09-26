import math
import cv2
import mediapipe as mp

# Initialize mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize mediapipe drawing module
mp_drawing = mp.solutions.drawing_utils

# Capture video from the default camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Flip the frame for selfie view
    frame = cv2.flip(frame, 1)

    # Convert the BGR frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the hand landmarks
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:

            # Access thumb landmarks
            thumb = [landmarks.landmark[i] for i in range(1, 5)]
            index = [landmarks.landmark[i] for i in range(5, 9)]
            middle = [landmarks.landmark[i] for i in range(9, 13)]
            ring = [landmarks.landmark[i] for i in range(13, 17)]
            pinky = [landmarks.landmark[i] for i in range(17, 21)]

            middle_tip = [landmarks.landmark[i] for i in range(11, 12)]
            middle_bottom = [landmarks.landmark[i] for i in range(10)]

            for point in index + thumb + middle + ring + pinky : 
                x, y = int(point.x * frame.shape[1]), int(point.y * frame.shape[0])
                cv2.putText(frame, f"{x}, {y}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if thumb[0].y < (index[0].y + middle[0].y + ring[0].y + pinky[0].y) / 4:
                    cv2.putText(frame, "Thumb Up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                if thumb[0].y > (index[0].y + middle[0].y + ring[0].y + pinky[0].y) / 4:
                    cv2.putText(frame, "Thumb Down", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw a line between the tip of the thumb and the tip of the index finger
                cv2.line(frame, 
                        (int(thumb[3].x * frame.shape[1]), int(thumb[3].y * frame.shape[0])), 
                        (int(index[3].x * frame.shape[1]), int(index[3].y * frame.shape[0])), 
                        (0, 255, 0), 
                        2)

            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



