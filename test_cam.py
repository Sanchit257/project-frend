import cv2

# Initialize the USB webcam (usually index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Webcam initialized successfully! Press 'q' to exit.")

while True:
    ret, frame = cap.isOpened() and cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # You can process frames here for Project Melanie's hazard detection
    
    # Just a status print to confirm stream is active
    print(f"Captured frame shape: {frame.shape}", end="\r")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

