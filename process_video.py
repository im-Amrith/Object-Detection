import cv2
from utils.object_detector import ObjectDetector

class VideoProcessor:
    def __init__(self):
        self.object_detector = ObjectDetector()
    
    def process(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Detect lanes & objects
            frame = self.lane_detector.detect(frame)
            frame = self.object_detector.detect(frame)
            
            out.write(frame)
            cv2.imshow('Output', frame)
            if cv2.waitKey(1) == ord('q'): break
        
        cap.release()
        out.release()
        cv2.destroyAllWindows()