import logging
import os

import cv2

logger = logging.getLogger(__name__)


def extract_frames_by_timestamp(video_path, timestamps, output_folder, output_format='jpg'):
    """
    Extract frames from a video at specific timestamps.
    
    :param video_path: Path to the video file.
    :param timestamps: List of timestamps (in seconds) where frames should be extracted.
    :param output_folder: Folder to save extracted frames.
    :param output_format: Image format (default is 'jpg').
    """
    # Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Cannot open video file {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    logger.debug(f"Video FPS: {fps}, Duration: {duration:.2f}s, Total Frames: {total_frames}")
    
    for timestamp in timestamps:
        if timestamp < 0 or timestamp > duration:
            logger.warning(f"Timestamp {timestamp}s is out of video range.")
            continue
        
        frame_number = int(round(timestamp * fps))
        frame_number = min(frame_number, total_frames - 1)  # Ensure within bounds
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        
        if ret:
            output_path = os.path.join(output_folder, f"frame_{timestamp:.2f}.{output_format}")
            cv2.imwrite(output_path, frame)
            logger.info(f"Saved frame at {timestamp:.2f}s to {output_path}")
        else:
            logger.error(f"Failed to extract frame at {timestamp:.2f}s")
    
    cap.release()
