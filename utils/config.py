import numpy as np

# Colors (BGR format)
VEHICLE_COLOR = (255, 0, 0)  # Blue
LIGHT_COLOR = (0, 255, 255)  # Yellow
LANE_COLOR = (0, 255, 0)    # Green

# Detection thresholds
VEHICLE_CONFIDENCE = 0.5

# Video processing settings
VIDEO_FPS = 30
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

# Lane detection parameters
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_RHO = 1
HOUGH_THETA = np.pi/180
HOUGH_THRESHOLD = 20
MIN_LINE_LENGTH = 20
MAX_LINE_GAP = 300
MIN_SLOPE = 0.5  # Minimum slope for lane lines

# Region of interest settings
ROI_HEIGHT_RATIO = 0.6  # Height ratio for region of interest
ROI_LEFT_RATIO = 0.25  # Left boundary ratio
ROI_RIGHT_RATIO = 0.75  # Right boundary ratio

# Overlay settings
OVERLAY_ALPHA = 0.3  # Transparency of the lane overlay