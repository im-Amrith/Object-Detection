from ultralytics import YOLO
import cv2
import numpy as np
from utils.config import VEHICLE_COLOR, LIGHT_COLOR

class ObjectDetector:
    def __init__(self):
        # Load YOLO model
        self.model = YOLO('yolov8n.pt')
        
        # Define classes to detect
        self.vehicle_classes = [2, 3, 5, 7]  # Cars, trucks, buses, bikes
        self.light_classes = [9]             # Traffic lights
        self.person_classes = [0]            # People
        
        # Initialize counters
        self.vehicle_count = 0
        self.person_count = 0
        
        # Initialize statistics
        self.frame_count = 0
        self.detection_stats = {
            'vehicles': [],
            'people': [],
            'traffic_lights': []
        }
    
    def _detect_traffic_light_color(self, frame, x1, y1, x2, y2):
        """Detect the color of a traffic light"""
        # Extract the traffic light region
        light_region = frame[y1:y2, x1:x2]
        
        # Apply some preprocessing to enhance the colors
        light_region = cv2.GaussianBlur(light_region, (5, 5), 0)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(light_region, cv2.COLOR_BGR2HSV)
        
        # Define color ranges with adjusted values
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([160, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        # Adjusted green range to be more sensitive
        green_lower = np.array([35, 50, 50])  # Lowered saturation and value thresholds
        green_upper = np.array([85, 255, 255])  # Widened hue range
        
        # Create masks for each color
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Count pixels of each color
        red_pixels = cv2.countNonZero(red_mask)
        yellow_pixels = cv2.countNonZero(yellow_mask)
        green_pixels = cv2.countNonZero(green_mask)
        
        # Lower the threshold for color detection
        min_pixels = 30  # Reduced from 50
        
        # Determine the dominant color
        max_pixels = max(red_pixels, yellow_pixels, green_pixels)
        if max_pixels < min_pixels:  # Lower threshold to avoid false negatives
            return "Unknown"
        elif max_pixels == red_pixels:
            return "Red"
        elif max_pixels == yellow_pixels:
            return "Yellow"
        else:
            return "Green"
    
    def detect(self, frame):
        # Increment frame counter
        self.frame_count += 1
        
        # Run YOLO with higher confidence threshold
        results = self.model(frame, conf=0.5, verbose=False)
        
        # Create a copy of the frame for drawing
        annotated_frame = frame.copy()
        
        # Reset counters for this frame
        self.vehicle_count = 0
        self.person_count = 0
        
        # Process detections
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Vehicle detection (blue boxes)
                if cls_id in self.vehicle_classes:
                    self.vehicle_count += 1
                    label = f"{self.model.names[cls_id]} {conf:.1f}"
                    color = VEHICLE_COLOR
                
                # Person detection (green boxes)
                elif cls_id in self.person_classes:
                    self.person_count += 1
                    label = f"Person {conf:.1f}"
                    color = (0, 255, 0)  # Green
                
                # Traffic light detection (yellow boxes)
                elif cls_id in self.light_classes:
                    # Detect traffic light color
                    light_color = self._detect_traffic_light_color(frame, x1, y1, x2, y2)
                    label = f"Traffic Light ({light_color}) {conf:.1f}"
                    color = LIGHT_COLOR
                else:
                    continue  # Skip other classes
                
                # Draw bounding box with thicker lines
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add background to text for better visibility
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(annotated_frame, (x1, y1-25), (x1+text_size[0], y1), color, -1)
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Update statistics
        self.detection_stats['vehicles'].append(self.vehicle_count)
        self.detection_stats['people'].append(self.person_count)
        
        # Add statistics overlay
        self._add_stats_overlay(annotated_frame)
        
        return annotated_frame
    
    def _add_stats_overlay(self, frame):
        """Add statistics overlay to the frame"""
        height, width = frame.shape[:2]
        
        # Create a semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        # Add statistics text
        cv2.putText(frame, f"Vehicles: {self.vehicle_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"People: {self.person_count}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {self.frame_count}", (width - 150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def get_statistics(self):
        """Return detection statistics"""
        if not self.detection_stats['vehicles']:
            return {
                'avg_vehicles': 0,
                'avg_people': 0,
                'max_vehicles': 0,
                'max_people': 0
            }
        
        return {
            'avg_vehicles': sum(self.detection_stats['vehicles']) / len(self.detection_stats['vehicles']),
            'avg_people': sum(self.detection_stats['people']) / len(self.detection_stats['people']),
            'max_vehicles': max(self.detection_stats['vehicles']),
            'max_people': max(self.detection_stats['people'])
        }