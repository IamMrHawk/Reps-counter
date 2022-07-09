
# requirements - pip install mediapipe
#              - pip install opencv-python
#              - pip install numpy

# imports
import cv2
import mediapipe as mp
import numpy as np
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# initializing cam
cap = cv2.VideoCapture(0)

# Counter variables
counter = 0
stage = None
timer = 0
num = 0
msg = ' '

# setting mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():

        ret, frame = cap.read()

        # convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # extract detections
        results = pose.process(image)

        # convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # creating box and text for timer
        cv2.rectangle(image, (0, 0), (400, 73), (245, 117, 16), -1)
        cv2.putText(image, 'Timer', (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(num),
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, msg,
                    (65, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        try:
            # Extracting landmarks
            landmarks = results.pose_landmarks.landmark

            # Extract coordinates
            hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            # Calculating leg fold angle
            def calculate_angle(a, b, c):
                a = np.array(a)  # First
                b = np.array(b)  # Mid
                c = np.array(c)  # End

                radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
                angle = np.abs(radians * 180.0 / np.pi)

                if angle > 180.0:
                    angle = 360 - angle

                return angle

            angle = calculate_angle(hip, knee, ankle)

            # display angle degree
            cv2.putText(image, str(angle),
                        tuple(np.multiply(knee, [640, 480]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                        )

            # counter logic for timer
            if angle > 140:
                stage = "down"
                num = 0
                msg = ' '
                timer = 0
            if angle < 50:
                if stage == 'down':
                    counter += 1
                    stage = "up"
                    print('Rep count : ',counter)
                timer += 1
                # print(timer)
                msg = 'Keep your knee bent'

                # if num > 0:
                if timer % 10 == 0:
                    print('Keep your knee bent for {} seconds'.format(num))
                    num -= 1

                if num == 0:
                    num = 8
                    timer = 0

        except:
            pass

        # display detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )

        cv2.imshow('Mediapipe Feed', image)
        time.sleep(0.01)    # sleep timer

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()
