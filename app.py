import os
import base64
import numpy as np
import cv2
from flask import Flask, render_template, request, jsonify, Response, send_file
from database import (
    insert_user,
    get_attendance_records,
    get_all_users,
    get_user_by_id,
    update_user,
    delete_user,
)
from face_module import FaceRecognitionSystem
from utils import export_attendance_csv

app = Flask(__name__)


fr_system = FaceRecognitionSystem()

camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def generate_frames():
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break
        else:
            frame = fr_system.process_frame(frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET'])
def register_page():
    # Make sure camera is released so browser can use it
    release_camera()
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def register_api():
    data = request.json
    user_id = data.get('user_id')
    name = data.get('name')
    department = data.get('department')
    image_data = data.get('image') # Base64 encoded JPEG
    
    if not all([user_id, name, department, image_data]):
        return jsonify({"error": "All fields are required"}), 400
        
    try:
        # Decode base64 image — handle both "data:...;base64,<data>" and raw base64
        encoded_data = image_data.split(',')[-1]  # works whether prefix exists or not
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"error": "Could not decode image. Please capture a clear photo and try again."}), 400

        # Convert BGR to RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get encoding
        encoding = fr_system.get_encoding_from_image(rgb_img)
        if encoding is None:
            return jsonify({"error": "No face found in image. Please try again."}), 400
            
        success, msg = insert_user(user_id, name, department, encoding)
        
        if success:
            # Reload known users in memory
            fr_system.load_users()
            return jsonify({"message": msg}), 200
        else:
            return jsonify({"error": msg}), 400
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during registration: {e}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/attendance')
def attendance_page():
    return render_template('attendance.html')

@app.route('/video_feed')
def video_feed():
    # Stream for attendance page
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    release_camera()
    return jsonify({"message": "Camera stopped"}), 200

@app.route('/records')
def records_page():
    records = get_attendance_records()
    users = get_all_users()
    return render_template('records.html', records=records, users=users)

@app.route('/users')
def users_page():
    users = get_all_users()
    # Remove encoding before passing to template
    for user in users:
        user.pop('encoding', None)
    return render_template('user_management.html', users=users)

@app.route('/api/users', methods=['GET'])
def users_api():
    users = get_all_users()
    # Remove encoding before returning JSON
    cleaned_users = []
    for user in users:
        user.pop('encoding', None)
        cleaned_users.append(user)
    return jsonify(cleaned_users), 200

@app.route('/api/user/<user_id>', methods=['PUT'])
def update_user_api(user_id):
    data = request.json or {}
    name = data.get('name')
    department = data.get('department')
    image_data = data.get('image')

    if name is None and department is None and image_data is None:
        return jsonify({"error": "Nothing to update."}), 400

    update_fields = {}
    if name is not None:
        update_fields['name'] = name
    if department is not None:
        update_fields['department'] = department

    if image_data:
        try:
            encoded_data = image_data.split(',')[-1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({"error": "Could not decode image."}), 400
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encoding = fr_system.get_encoding_from_image(rgb_img)
            if encoding is None:
                return jsonify({"error": "No face found in image. Please try again."}), 400
            update_fields['encoding'] = encoding
        except Exception as e:
            return jsonify({"error": f"Failed to process image: {str(e)}"}), 500

    success, msg = update_user(user_id, **update_fields)
    if success:
        fr_system.load_users()
        return jsonify({"message": msg}), 200
    return jsonify({"error": msg}), 400

@app.route('/api/user/<user_id>', methods=['DELETE'])
def delete_user_api(user_id):
    success, msg = delete_user(user_id)
    if success:
        fr_system.load_users()
        return jsonify({"message": msg}), 200
    return jsonify({"error": msg}), 404

@app.route('/export')
def export_csv():
    # Download the CSV file
    filepath = export_attendance_csv()
    if filepath and os.path.exists(filepath):
        return send_file(filepath, as_attachment=True)
    return "No records to export.", 404

if __name__ == '__main__':
    app.run(debug=True, port=5000, threaded=True)
