# Object Detection System

A real-time object detection system using YOLOv8 to identify and track vehicles, people, and traffic lights in videos. The system includes:
- YOLOv8 model for object detection
- Traffic light color detection
- Statistical analysis of detected objects

> **Note**: Full IEEE format documentation is available in [object_detection_ieee_paper.md](object_detection_ieee_paper.md)

## Features

- **Real-time Object Detection**: Identifies vehicles, people, and traffic lights
- **Traffic Light Color Recognition**: Analyzes traffic light status (Red, Yellow, Green)
- **Statistics Tracking**: Counts and records objects per frame
- **Performance Metrics**: Monitors processing speed and detection rates
- **Video Processing**: Processes and saves annotated videos with detection overlays

## Requirements

- Python 3.7+
- PyTorch
- OpenCV 4.5+
- NumPy 1.20+
- Ultralytics YOLO

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/object-detection.git
   cd object-detection
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Ensure the YOLOv8 model file (`yolov8n.pt`) is in the project root directory

## Usage

### Process a video with object detection

```
python main.py --mode process --input your_video.mp4 --output result.mp4
```

### Additional options

```
--no-display     # Disable real-time display during processing
--no-stats       # Disable statistics saving
```

## How It Works

The system uses a deep learning-based approach for object detection:

**Object Detection**:
- Uses YOLOv8 pretrained model 
- Identifies vehicles (cars, trucks, buses, bikes)
- Detects people and traffic lights
- Draws bounding boxes with class labels and confidence scores

**Traffic Light Analysis**:
- Detects traffic light status (Red, Yellow, Green)
- Uses HSV color space analysis for light state detection

**Statistics**:
- Tracks number of vehicles and people per frame
- Calculates average and maximum object counts
- Generates statistics report at completion
- Overlays real-time stats on processed video

## Project Structure

- `main.py`: Main script for video processing
- `utils/object_detector.py`: Object detection implementation
- `utils/config.py`: Configuration settings
- `requirements.txt`: Required Python packages
- `object_detection_ieee_paper.md`: IEEE format documentation

## Examples

Run the detector on included test videos:

```
python main.py --mode process --input test.mp4 --output output.mp4
```

## License

[MIT License](LICENSE)
