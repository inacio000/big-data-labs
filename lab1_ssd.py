import cv2
import numpy as np

prototxt_path = "weights/deploy.prototxt.txt"

model_path = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Load the caffe model
model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Read the image
image = cv2.imread("kids-image3-ssd.png")

# Get image weight and height
h, w = image.shape[:2]

# Preprocessing: resize and subtract the mean
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

# Set the neural network input to the image
model.setInput(blob)

# Perform logic output and get the result
output = np.squeeze(model.forward())

font_scale = 1.0

for i in range(0, output.shape[0]):
    confidence = output[i, 2]

    if confidence > 0.5:
        # get the coordinates of the surrounding block and scale them to the original image
        box = output[i, 3:7] * np.array([w, h, w, h])

        # Convert to integers
        start_x, start_y, end_x, end_y = box.astype(int)

        # Draw a rectangle around the face
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y),
                      color=(255, 0, 0), thickness=2)

        # also draw text
        cv2.putText(image, f"{confidence*100:.2f}%", (start_x, start_y-5), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 0, 0), 2)

# Show the image
cv2.imshow("image", image)
cv2.waitKey(0)

# Save the image with rectangles
cv2.imwrite("kids-image3-ssd-detected.png", image)
