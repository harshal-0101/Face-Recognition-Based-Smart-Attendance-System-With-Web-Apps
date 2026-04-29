import cv2
import face_recognition
import numpy as np
from database import get_all_users, mark_attendance
from utils import speak_async


class FaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_users()

    def load_users(self):
        """Load users from DB"""
        users = get_all_users()

        self.known_face_encodings.clear()
        self.known_face_names.clear()
        self.known_face_ids.clear()

        for user in users:
            if 'encoding' in user:
                self.known_face_encodings.append(np.array(user['encoding'], dtype=np.float64))
                self.known_face_names.append(user['name'])
                self.known_face_ids.append(user['user_id'])

        print(f"✅ Loaded {len(self.known_face_names)} users")

    # -------------------------------
    # IMAGE → ENCODING
    # -------------------------------
    def get_encoding_from_image(self, image):
        """Convert image to face encoding"""

        if image is None or image.size == 0:
            print("❌ Invalid image")
            return None

        # Ensure uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # Convert to RGB safely
        try:
            if len(image.shape) == 2:
                rgb_image = image

            elif image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            elif image.shape[2] == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

            else:
                print("❌ Unsupported image format:", image.shape)
                return None

        except Exception as e:
            print("❌ Color conversion error:", e)
            return None

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_image)

        if not face_locations:
            print("⚠️ No face found")
            return None

        encodings = face_recognition.face_encodings(rgb_image, face_locations)

        if not encodings:
            print("⚠️ Encoding failed")
            return None

        return encodings[0]

    # -------------------------------
    # VIDEO FRAME PROCESSING
    # -------------------------------
    def process_frame(self, frame):
        """Process webcam frame"""

        # Validate frame
        if frame is None or frame.size == 0:
            print("❌ Invalid frame")
            return frame

        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Resize for performance
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        except Exception as e:
            print("❌ Resize error:", e)
            return frame

        # Convert BGR → RGB
        try:
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print("❌ RGB conversion error:", e)
            return frame

        # Detect faces
        face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            name = "Unknown"

            if len(self.known_face_encodings) > 0:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

                if len(distances) > 0:
                    best_match_index = np.argmin(distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        user_id = self.known_face_ids[best_match_index]

                        # Mark attendance
                        try:
                            success, msg = mark_attendance(user_id, name)
                            if success:
                                print(msg)
                                speak_async(f"{name} marked present")
                        except Exception as e:
                            print("❌ Attendance error:", e)

            face_names.append(name)

        # Draw results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Label
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        return frame