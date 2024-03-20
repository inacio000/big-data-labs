import cv2

# uploading image
image = cv2.imread("kids-image3.png")

# convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Initialize the face recognizer (Haar's default cascade)
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# Detect all faces in the image
faces = face_cascade.detectMultiScale(image_gray)

# Print the number of faces found
print(f'{len(faces)} faces detected in the image.')

# For all the faces detected, draw a blue square
for x, y, width, height in faces:
    cv2.rectangle(image, (x, y), (x + width, y + height), color = (255, 0, 0), thickness = 2)

# Save the image with detected faces
    cv2.imwrite("kids-image3-detected.png", image)