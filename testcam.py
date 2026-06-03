import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # Load the YOLOv8n model
cap = cv2.VideoCapture(2) # Open the camera (use 0 for default camera, change if you have multiple cameras)

if not cap.isOpened(): # Check if the camera opened successfully
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read() # Read a frame from the camera
    if not ret: # Check if the frame was read successfully
        print("Error: Could not read frame.") 
        break

    results = model.track(frame, stream=True, classes=[0], persist=True, tracker='botsort.yaml') # Run object detection and tracking on the frame, filtering for class ID 0 (person)
    
    #BoTSort is a tracking algorithm that can be used with YOLO models to track detected objects across frames. The 'botsort.yaml' file contains the configuration for the BoTSort tracker.

    for result in results: # Iterate through the results
        people = []
        boxes = result.boxes # Get bounding boxes from the result

        if boxes is None: # If no boxes are detected, skip to the next frame
            continue
        
        for box in boxes: # Iterate through the detected boxes
            if box.id is None: # If no tracking ID is available, skip to the next box
                continue
            if box.conf[0] < 0.5: # Filter out boxes with confidence less than 0.5
                continue

            confidence = box.conf[0] # Get the confidence score of the box
            
            #class_id = int(box.cls[0])  # Get class ID
            
            track_id = int(box.id[0]) if box.id is not None else -1 # Get the tracker id from the box, if id is none, set value to -1 to indicate no tracking ID available.
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            height = y2 - y1 # Calculate the height of the bounding box

            people.append({
                'track_id': track_id,
                'box': (x1, y1, x2, y2), 
                'height': height}) # Append the detected person information to the people list

            if people: # If there are detected people, find the one with the maximum height (closest to the camera)
                tallest = max(people, key=lambda p: p['height']) # Find the person with the maximum height
                target_id = tallest['track_id'] # Get the tracking ID of the tallest person

            for person in people: # Iterate through the detected people
                track_id = person['track_id'] # Get the tracking ID of the person
                x1, y1, x2, y2 = person['box'] # Get the bounding box coordinates of the person
                confidence = confidence # Get the confidence score of the detection

                if track_id == target_id: # Filter for the target ID
                    # Draw green bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'TALLEST - {confidence:.2f} CONFIDENCE' # Create a label with the track ID, class name, and confidence score
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) # Put the label on the frame
                else:
                    # Draw red bounding box and label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f'NON TALLEST - {confidence:.2f} CONFIDENCE' # Create a label with the track ID, class name, and confidence score
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) # Put the label on the frame
           
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit the loop if 'q' is pressed
        break
cap.release() # Release the camera
cv2.destroyAllWindows() # Close all OpenCV windows
#