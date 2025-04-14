import os
import cv2
import argparse
import time
from utils.object_detector import ObjectDetector

def process_video(input_path, output_path, display=True, save_stats=True):
    """Process a video file with object detection"""
    # Initialize video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return False
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize object detector
    object_detector = ObjectDetector()
    
    # Initialize timing variables
    start_time = time.time()
    frame_count = 0
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect objects in the frame
        processed_frame = object_detector.detect(frame)
        
        # Write frame
        out.write(processed_frame)
        
        # Display frame if requested
        if display:
            cv2.imshow('Object Detection', processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Update progress
        frame_count += 1
        if frame_count % 30 == 0:  # Print progress every second (assuming 30 fps)
            elapsed_time = time.time() - start_time
            fps_processing = frame_count / elapsed_time
            progress = (frame_count / total_frames) * 100
            print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%) - {fps_processing:.1f} fps")
    
    # Clean up
    cap.release()
    out.release()
    if display:
        cv2.destroyAllWindows()
    
    # Calculate and display statistics
    elapsed_time = time.time() - start_time
    processing_fps = frame_count / elapsed_time
    
    print(f"\nVideo processing complete. Output saved to {output_path}")
    print(f"Total frames: {frame_count}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {processing_fps:.1f} fps")
    
    # Get and display detection statistics
    stats = object_detector.get_statistics()
    print("\nDetection Statistics:")
    print(f"Average vehicles per frame: {stats['avg_vehicles']:.1f}")
    print(f"Average people per frame: {stats['avg_people']:.1f}")
    print(f"Maximum vehicles in a frame: {stats['max_vehicles']}")
    print(f"Maximum people in a frame: {stats['max_people']}")
    
    # Save statistics to file if requested
    if save_stats:
        stats_file = os.path.splitext(output_path)[0] + "_stats.txt"
        with open(stats_file, 'w') as f:
            f.write(f"Video: {input_path}\n")
            f.write(f"Output: {output_path}\n")
            f.write(f"Total frames: {frame_count}\n")
            f.write(f"Processing time: {elapsed_time:.2f} seconds\n")
            f.write(f"Average processing speed: {processing_fps:.1f} fps\n\n")
            f.write("Detection Statistics:\n")
            f.write(f"Average vehicles per frame: {stats['avg_vehicles']:.1f}\n")
            f.write(f"Average people per frame: {stats['avg_people']:.1f}\n")
            f.write(f"Maximum vehicles in a frame: {stats['max_vehicles']}\n")
            f.write(f"Maximum people in a frame: {stats['max_people']}\n")
        print(f"Statistics saved to {stats_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Enhanced Object Detection System')
    parser.add_argument('--mode', choices=['process'], required=True,
                      help='Operation mode: process video')
    parser.add_argument('--input', required=True,
                      help='Input video file path')
    parser.add_argument('--output', required=True,
                      help='Output video file path')
    parser.add_argument('--no-display', action='store_true',
                      help='Disable real-time display')
    parser.add_argument('--no-stats', action='store_true',
                      help='Disable statistics saving')
    
    args = parser.parse_args()
    
    if args.mode == 'process':
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist")
            return
        
        # Create output directory if it doesn't exist and if the output path contains a directory
        output_dir = os.path.dirname(args.output)
        if output_dir:  # Only create directory if output path contains a directory
            os.makedirs(output_dir, exist_ok=True)
        
        # Process the video
        success = process_video(
            args.input, 
            args.output, 
            display=not args.no_display,
            save_stats=not args.no_stats
        )
        
        if not success:
            print("Error: Failed to process video")
            return
        
        print("Processing complete!")

if __name__ == '__main__':
    main()