import cv2
import os

#Colors
green = (0,255,0)

# Construct the full path to the XML file
opencv_path = r'C:\Python311\Lib\site-packages\cv2'
xml_file_path = os.path.join(opencv_path, 'data','haarcascade_frontalface_default.xml')

# Load the cascade classifier
trained_face_data = cv2.CascadeClassifier(xml_file_path)

# Load an image
image_path = 'image.jpg'
image = cv2.imread(image_path)
if image is None:
    raise Exception(f"Failed to load image: {image_path}")

# Convert the image to grayscale
grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

print(face_coordinates)
# Your code to draw rectangles around detected faces here
(x,y,w,h) = face_coordinates[0]
cv2.rectangle(image, (x,y), (x+w,y+h),green, 2)
cv2.imshow("Face detected", image)
cv2.waitKey()
cv2.imshow("Grayscaled Image",grayscaled_image)
cv2.waitKey()

