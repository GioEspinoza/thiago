import cv2
from ultralytics import YOLO
import torch
from torchreid.utils import FeatureExtractor

def get_people_from_boxes(boxes):

    people = []

    for box in boxes:
        if box.id is None:
            continue

        if box.conf[0] < 0.5:
            continue

        track_id = int(box.id[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        height = y2 - y1

        people.append({
            "track_id": track_id,
            "box": (x1, y1, x2, y2),
            "height": height,
            "confidence": confidence
        })

    return people

def draw_person(frame, person, target_id):
    x1, y1, x2, y2 = person["box"]
    confidence = person["confidence"]

    if person["track_id"] == target_id:
        color = (0, 255, 0)
        label = f"TALLEST - {confidence:.2f}"
    else:
        color = (0, 0, 255)
        label = f"NON TALLEST - {confidence:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
def find_tallest_person(people):
    if not people:
        return None

    return max(people, key=lambda p: p["height"])

model = YOLO('yolov8n.pt')  
cap = cv2.VideoCapture(0) # Open the camera (use 0 for default camera, change if you have multiple cameras)
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
) # Initialize the feature extractor for person re-identification using the OSNet model

if not cap.isOpened(): 
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret: 
        print("Error: Could not read frame.") 
        break

    results = model.track(frame, stream=True, classes=[0], persist=True, tracker='botsort.yaml') # Run object detection and tracking on the frame, filtering for class ID 0 (person)
    #BoTSort is a tracking algorithm that can be used with YOLO models to track detected objects across frames. The 'botsort.yaml' file contains the configuration for the BoTSort tracker.

    for result in results: 
        boxes = result.boxes 

        if boxes is None:
            continue
        
        people = get_people_from_boxes(boxes) 
        tallest = find_tallest_person(people) 

        if tallest is None:
            continue

        target_id = tallest['track_id']

        for person in people: 
           
            person_crop = frame[person['box'][1]:person['box'][3], person['box'][0]:person['box'][2]] 
           
            features = extractor(person_crop) #extract features for future reference
                
            draw_person(frame, person, target_id) 
           
    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'): # Exit the loop if 'q' is pressed
        break
cap.release() 
cv2.destroyAllWindows() 

