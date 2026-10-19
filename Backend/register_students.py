import cv2
import face_recognition
import pickle
import os

ENCODING_FILE= "student_encodings.pkl"

#load existing data
if os.path.exists(ENCODING_FILE):
    with open(ENCODING_FILE, "rb") as f:
        student_data = pickle.load(f)
else:
    student_data = {}

name = input("Enter student name: ")


cap = cv2.VideoCapture(0) 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame not meet")
        break
    cv2.imshow("Press 's' to Save, 'q' to Quit", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'): 
        print("Face detectecting...")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb_frame)
        
        if len(faces) > 0:
            encoding = face_recognition.face_encodings(rgb_frame, faces)[0]
            student_data[name] = encoding
            
            #saving encoded face into pickle
            with open(ENCODING_FILE, "wb") as f:
                pickle.dump(student_data, f)
            print(f"{name}'Registration Completed")
            break
        else:
            print("Try Again:")
            
    elif key == ord('q'):
        break
    

cap.release()
cv2.destroyAllWindows()
