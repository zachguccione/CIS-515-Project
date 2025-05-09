import cv2
import numpy as np
from ultralytics import YOLO

def draw_roi(frame, roi):
    """Draw ROI rectangle on frame with semi-transparent grey fill"""
    # Create a copy of the frame for the overlay
    overlay = frame.copy()
    
    # Convert ROI coordinates to integers
    x1, y1, x2, y2 = map(int, roi)
    
    # Draw filled rectangle with grey color (BGR: 128, 128, 128)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (128, 128, 128), -1)  # -1 for filled
    
    # Add the overlay with transparency
    alpha = 0.3  # Increased transparency factor for better visibility
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Draw the border with grey color
    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 3)  # Thicker border

def is_in_roi(bbox, roi):
    """Check if bounding box intersects with ROI"""
    x1, y1, x2, y2 = bbox
    roi_x1, roi_y1, roi_x2, roi_y2 = roi
    
    # Check for intersection
    return not (x2 < roi_x1 or x1 > roi_x2 or y2 < roi_y1 or y1 > roi_y2)

def main():
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Initialize webcam (default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Get screen dimensions
    screen_width = 1920  # Default to common screen width
    screen_height = 1080  # Default to common screen height
    
    # Try to get actual screen dimensions
    try:
        import ctypes
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
    except:
        pass
    
    # Create a resizable window
    cv2.namedWindow('Human Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Human Detection', screen_width, screen_height)
    cv2.setWindowProperty('Human Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Define box positions and sizes in pixels
    # Format: (x1, y1, x2, y2) where (x1,y1) is top-left and (x2,y2) is bottom-right
    # Original boxes were 200x300 pixels, now reduced by 50% to 100x150 pixels
    # Centers remain at (200, 250) and (450, 250)
    roi1 = (150, 175, 250, 325)  # First box: 100x150 pixels
    roi2 = (400, 175, 500, 325)  # Second box: 100x150 pixels
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from camera")
            break
            
        # Draw ROIs
        draw_roi(frame, roi1)
        draw_roi(frame, roi2)
        
        # Add instructions to frame
        cv2.putText(frame, 'Press ESC to exit', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Run YOLOv8 inference
        results = model(frame, classes=[0])  # 0 is the class ID for 'person'
        
        # Process detections
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check if detection intersects with any ROI
                if (is_in_roi((x1, y1, x2, y2), roi1) or 
                    is_in_roi((x1, y1, x2, y2), roi2)):
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    # Add confidence score
                    conf = float(box.conf[0])
                    cv2.putText(frame, f'Person {conf:.2f}', (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Display frame
        cv2.imshow('Human Detection', frame)
        
        # Break loop on ESC key
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
