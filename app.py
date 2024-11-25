import cv2
import os
from flask import Flask, request, render_template
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

app = Flask(__name__)

nimgs = 10
imgBackground = cv2.imread("background1.jpg")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Create directories if they do not exist
os.makedirs('Attendance', exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('static/faces', exist_ok=True)

# Create attendance CSV if it doesn't exist
attendance_file = f'Attendance/Attendance-{datetoday}.csv'
if attendance_file not in os.listdir('Attendance'):
    with open(attendance_file, 'w') as f:
        f.write('Name,Roll,Time\n')

# Returns the total number of registered users
def totalreg():
    return len(os.listdir('static/faces'))

# Function to extract faces from the image using face detector
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except Exception as e:
        print(f"Error in extracting faces: {e}")
        return []

# Function to identify face using the trained model
def identify_face(facearray):
    try:
        model = joblib.load('static/face_recognition_model.pkl')
        return model.predict(facearray)
    except Exception as e:
        print(f"Error in identifying face: {e}")
        return None

# Function to train the KNN model
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

# Extract attendance data from the CSV
def extract_attendance():
    df = pd.read_csv(attendance_file)
    names = df['Name'].tolist()
    rolls = df['Roll'].tolist()
    times = df['Time'].tolist()
    l = len(df)
    return names, rolls, times, l

# Function to add a new attendance entry
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")
    df = pd.read_csv(attendance_file)

    if int(userid) not in df['Roll'].astype(int).tolist():
        with open(attendance_file, 'a') as f:
            f.write(f'{username},{userid},{current_time}\n')

# Get all registered users
def get_all_users():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, len(userlist)

@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    # Ensure the model exists before starting recognition
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2, 
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open webcam"

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))
            if identified_person is not None:
                add_attendance(identified_person[0])
                cv2.putText(frame, f'{identified_person[0]}', (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Center the frame on imgBackground
            start_y = (imgBackground.shape[0] - frame.shape[0]) // 2
            start_x = (imgBackground.shape[1] - frame.shape[1]) // 2
            imgBackground[start_y:start_y + frame.shape[0], start_x:start_x + frame.shape[1]] = frame

            cv2.imshow('Attendance', imgBackground)

        if cv2.waitKey(1) == 27:  # Press 'Esc' to quit
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Error: Could not open webcam"
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0 and i < nimgs:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if i >= nimgs or cv2.waitKey(1) == 27:
            break

        cv2.imshow('Adding new User', frame)

    cap.release()
    cv2.destroyAllWindows()

    print('Training Model')
    train_model()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)

@app.route('/view-users')
def view_users():
    userlist, names, rolls, l = get_all_users()
    zipped_list = list(zip(names, rolls))
    return render_template('view_users.html', zipped_list=zipped_list, l=l)

if __name__ == '__main__':
    app.run(debug=True)