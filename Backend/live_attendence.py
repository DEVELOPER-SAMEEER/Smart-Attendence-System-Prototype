import cv2
import face_recognition
import pickle
import requests
from datetime import datetime
import numpy as np

SERVER_URL = "http://localhost:3000/attendance"
ENCODING_FILE = "student_encodings.pkl"

#Encodings load karein
try:
    with open(ENCODING_FILE, "rb") as f:
        student_data = pickle.load(f)
        known_face_names = list(student_data.keys())
        known_face_encodings = list(student_data.values())
        print(f"Total {len(known_face_names)} students ka data load hua: {known_face_names}")
except FileNotFoundError:
    print("Data not found! Please Register Firts.")
    exit()

# JSON Server implement
def mark_attendance_json(name):
    today_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Check karein aaj entry hui hai ya nahi
        records = requests.get(SERVER_URL).json()
        already_marked = False
        
        for rec in records:
            if rec.get('name') == name and rec.get('date') == today_date:
                already_marked = True
                break
        
        # Agar aaj entry nahi hui hai toh POST karein
        if not already_marked:
            time_string = datetime.now().strftime('%H:%M:%S')
            new_record = {
                "name": name, 
                "date": today_date, 
                "time": time_string, 
                "status": "Present"
            }
            requests.post(SERVER_URL, json=new_record)
            print(f"SUCCESS: Attendance Marked for {name} at {time_string}")
            
    except requests.exceptions.ConnectionError:
        print("Server connecting errrrr")

#Grand entry of open cv
print("Camera Opening...")
cap = cv2.VideoCapture(0) 

while True:
    ret, frame = cap.read()
    if not ret:
        print("frame not received")
        break
        
    # Frame croping
    small_frame = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Face detecting
    faces_in_frame = face_recognition.face_locations(rgb_small_frame)
    encodings_in_frame = face_recognition.face_encodings(rgb_small_frame, faces_in_frame)

    for encode_face, face_loc in zip(encodings_in_frame, faces_in_frame):
        #comparing
        matches = face_recognition.compare_faces(known_face_encodings, encode_face, tolerance=0.4)
        face_distance = face_recognition.face_distance(known_face_encodings, encode_face)
        
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            name = known_face_names[match_index]
            
            # JSON Server par mark karein
            mark_attendance_json(name)
            
            #Name and Border Show
            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Live Attendance (Press Q to quit)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
