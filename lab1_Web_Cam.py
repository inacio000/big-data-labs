import cv2

# Create a new camera project
cap = cv2.VideoCapture(0)

# Initialize the face search (Haar cascade by default)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
while True:
    # Reading the image from the camera
    ret, image = cap.read()

    # Conversion to grayscale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the photo
    faces = face_cascade.detectMultiScale(image_gray, 1.3,5)

    # # Draw a blue square for each detected face
    for x, y, width, height in faces:
        cv2.rectangle(image, (x, y), (x + width, y + height), color = (255, 0, 0), thickness = 2)

    cv2.imshow('Face Detection', image)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()