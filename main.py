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
# image_path = 'image.jpg'
# image = cv2.imread(image_path)
# if image is None:
#     raise Exception(f"Failed to load image: {image_path}")

# Convert the image to grayscale
# grayscaled_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform face detection
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

# print(face_coordinates)
# Draw rectangles around detected faces
# for face_coordinate in face_coordinates:
#     (x,y,w,h) = face_coordinate
#     cv2.rectangle(image, (x,y), (x+w,y+h),green, 2)
# cv2.imshow("Face detected", image)
# cv2.waitKey()
# cv2.imshow("Grayscaled Image",grayscaled_image)
# cv2.waitKey()


#Reading from webcam
webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    # Horizontally flip the frame to remove the mirrored effect
    frame_flipped = cv2.flip(frame, 1)

    # Convert the flipped frame to grayscale for face detection
    grayscaled_image = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscaled image
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_image)

    # Draw rectangles around detected faces
    for face_coordinate in face_coordinates:
        (x, y, w, h) = face_coordinate
        cv2.rectangle(frame_flipped, (x, y), (x + w, y + h), green, 2)

    # Display the flipped frame with face detection
    cv2.imshow("Webcam", frame_flipped)

    # Check for a key press and exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()

