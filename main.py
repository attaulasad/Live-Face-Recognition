import threading
import cv2
from deepface import DeepFace

# Initialize video capture from the default webcam (device 0)
cap = cv2.VideoCapture(0)

# Set frame width and height for the video capture
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize a counter to keep track of the number of frames processed
counter = 0

# Initialize a flag to indicate whether a face match is found
face_match = False

# Load the reference image for face comparison
reference_img = cv2.imread("reference.jpg")

# Create a lock object to synchronize access to shared resources between threads
lock = threading.Lock()

def check_face(frame):
    global face_match
    try:
        # Perform face verification between the current frame and the reference image
        match = DeepFace.verify(frame, reference_img.copy())['verified']
        # Update the face_match flag in a thread-safe manner
        with lock:
            face_match = match
    except ValueError:
        # In case of an error, ensure face_match is set to False
        with lock:
            face_match = False

while True:
    # Capture a frame from the video feed
    ret, frame = cap.read()
    
    if ret:
        # Every 38 frames, process the 8th frame for face verification
        if counter % 38 == 8:
            try:
                # Start a new thread to check the face in the current frame
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError:
                # Handle any potential errors starting the thread
                pass
        
        # Increment the frame counter
        counter += 1
        
        # Update the displayed text based on whether a face match was found
        with lock:
            if face_match:
                cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        
        # Display the frame with the overlayed text
        cv2.imshow("video", frame)
    
    # Check if the user pressed the 'q' key to quit the loop
    key = cv2.waitKey(1)
    if key == ord("q"):
        break

# Release the video capture resource and close the display window
cap.release()
cv2.destroyAllWindows()
