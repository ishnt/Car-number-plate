import cv2
import streamlit as st
from PIL import Image
import uuid

# Path to the Haar Cascade classifier XML file for license plate detection
harcascade = "E:\\Car number plate\\JioCloud-Download\\Car-Number-Plates-Detection-main\\model\\haarcascade_russian_plate_number.xml"

# Function to detect license plates
def detect_plates(img, min_area):
    # Creating a license plate classifier
    plate_cascade = cv2.CascadeClassifier(harcascade)
    
    # Converting the frame to grayscale for easier processing
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detecting license plates in the grayscale image
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    # Iterating through the detected plates
    for (x, y, w, h) in plates:
        area = w * h
        # Checking if the detected area is larger than the minimum area
        if area > min_area:
            # Drawing a rectangle around the detected license plate
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Adding a label for the detected plate
            cv2.putText(img, "Number Plate", (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
    return img

# Streamlit App
def main():
    st.title('Number Plate Detection')
    # Minimum area for a detected region to be considered a license plate
    min_area = 500

    # Accessing the webcam (index 0) using VideoCapture
    cap = cv2.VideoCapture(0)

    # Display an empty image to start the Streamlit app
    placeholder = st.empty()
    stop_button_id = str(uuid.uuid4())
    stop_button = st.button("Stop", key=stop_button_id)
    while True:
          # Generating a unique ID for the stop button
       
        
        if stop_button:
            break

        # Reading frames from the webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Unable to read frame from webcam.")
            break

        # Detecting plates in the image
        img_with_plates = detect_plates(frame, min_area)

        # Convert the OpenCV image to RGB format
        img_with_plates = cv2.cvtColor(img_with_plates, cv2.COLOR_BGR2RGB)

        # Displaying the result with detected plates
        placeholder.image(img_with_plates, channels='RGB', use_column_width=True)

    cap.release()

if __name__ == "__main__":
    main()
