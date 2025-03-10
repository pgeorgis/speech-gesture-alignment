import json
import logging
import os
from collections import defaultdict
from copy import deepcopy

import cv2
import mediapipe as mp
import numpy as np

from constants import (NEAREST_GESTURE_APEX_KEY, NEAREST_GESTURE_KEY,
                       NEAREST_GESTURE_OFFSET_KEY, TOKEN_END_KEY, TOKEN_KEY,
                       TOKEN_ONSET_KEY)

logger = logging.getLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands


def get_hands_detection_model(**kwargs):
    """Return a new instance of a mediapipe Hands detection model."""
    return mp_hands.Hands(**kwargs)


def get_frame_timestamp_in_seconds(cap):
    """Get capture frame timestamp and convert from seconds to milliseconds."""
    return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000


def convert_frame_to_rgb(frame):
    """Convert a video frame to RGB format."""
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def detect_hands(rgb_image, model: mp.solutions.hands.Hands):
    """Detect hands in an image."""
    return model.process(rgb_image)
    

def detect_video_frames_with_hand_gestures(video_path: str,
                                  hands_detection_model: mp.solutions.hands.Hands,
                                  start_bound: float = 0.0,
                                  end_bound: float=None,
                                  ) -> dict:
    """Detect video frames containing hands and return index of hand gestures with associated hand landmarks and timing information."""
    # Open video file and load as video capture object
    cap = cv2.VideoCapture(video_path)

    # Iterate over video frames and collect frames with hand gestures and their landmarks
    gesture_frames = {}
    frame_n = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_n += 1
        
        # Get frame timestamp, skip if out of bounds
        frame_timestamp = get_frame_timestamp_in_seconds(cap)
        if frame_timestamp < start_bound:
            continue
        if end_bound and frame_timestamp > end_bound:
            continue
        
        # Convert the frame to RGB format
        frame_rgb = convert_frame_to_rgb(frame)
        
        # Process the frame to detect hands
        results = detect_hands(frame_rgb, hands_detection_model)
        
        # Check if hands were detected
        if results.multi_hand_landmarks:
            # Save the hand landmarks and timestamps
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand shape as (x, y, z) coordinates
                hand_shape = [np.array([lm.x, lm.y, lm.z]) for lm in hand_landmarks.landmark]
                gesture_entry = {
                    "timestamp": frame_timestamp,
                    "hand_shape": hand_shape
                }
                gesture_frames[frame_n] = gesture_entry

    return gesture_frames


def combine_overlapping_gestures(gesture_events):
    """Combine lists of potentially overlapping gesture events into non-overlapping gesture events."""
    combined_gestures = defaultdict(list)
    seen_gesture_count = 0
    for gesture_event in gesture_events:
        for _, gesture_entry in gesture_event.items():
            last_gesture_timestamp = combined_gestures[seen_gesture_count][-1]['timestamp'] if seen_gesture_count > 0 else 0
            if seen_gesture_count == 0 or gesture_entry[0]['timestamp'] > last_gesture_timestamp:
                seen_gesture_count += 1
            combined_gestures[seen_gesture_count].extend(
                [
                    gesture_frame
                    for gesture_frame in gesture_entry
                    if gesture_frame["timestamp"] > last_gesture_timestamp
                ]
            )
            combined_gestures[seen_gesture_count].sort(key=lambda x: x['timestamp'])
    return combined_gestures


def find_nearest_gesture_to_words(word_timings: list,
                                  gesture_apices: dict,
                                  key=TOKEN_ONSET_KEY,
                                  max_offset_from_word: float | None = None,
                                  ) -> list:
    """Find nearest gesture (by apex timestamp) to each word's onset."""
    nearest_gestures = []
    for entry in word_timings:
        entry_copy = entry.copy()
        word_time = entry_copy[key]
        nearest_gesture = min(gesture_apices.keys(), key=lambda x: abs(word_time - gesture_apices[x]))
        nearest_gesture_apex = gesture_apices[nearest_gesture]
        nearest_gesture_offset = nearest_gesture_apex - word_time
        if max_offset_from_word and abs(nearest_gesture_offset) > max_offset_from_word:
            entry_copy[NEAREST_GESTURE_KEY] = None
            entry_copy[NEAREST_GESTURE_APEX_KEY] = None
            entry_copy[NEAREST_GESTURE_OFFSET_KEY] = None
        else:
            entry_copy[NEAREST_GESTURE_KEY] = nearest_gesture
            entry_copy[NEAREST_GESTURE_APEX_KEY] = nearest_gesture_apex
            entry_copy[NEAREST_GESTURE_OFFSET_KEY] = nearest_gesture_offset
        nearest_gestures.append(entry_copy)

    return nearest_gestures


class GestureDetector:
    def __init__(self,
                 video_path: str,
                 start_bound: float = 0.0,
                 end_bound: float = None,
                 min_detection_confidence=0.75,
                 **kwargs):
        self.video_path = video_path
        self.fps = self.get_frames_per_second()
        self.frame_interval = self.get_frame_interval()
        self.start_bound = start_bound
        self.end_bound = end_bound
        self.model = get_hands_detection_model(min_detection_confidence=min_detection_confidence, **kwargs)
        self.gesture_frames = self.detect_hand_gestures_in_video()
    
    def set_bounds(self, start_bound: float, end_bound: float | None) -> tuple[float, float]:
        """Adjust bounds to be within preset video bounds."""
        start_bound = max(start_bound, self.start_bound)
        if end_bound is None and self.end_bound is not None:
            end_bound = min(end_bound, self.end_bound)
        return start_bound, end_bound
    
    def get_frames_per_second(self):
        """Get frames per second."""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        return fps
    
    def get_frame_interval(self):
        """Get video frame interval."""
        return 1 / self.fps if self.fps > 0 else None

    def detect_hand_gestures_in_video(self, start_bound: float = 0.0, end_bound=None, **kwargs) -> list:
        """Get list of all video frames with hand gestures."""
        start_bound, end_bound = self.set_bounds(start_bound, end_bound)
        gesture_frames = detect_video_frames_with_hand_gestures(
            self.video_path,
            self.model,
            start_bound=start_bound,
            end_bound=end_bound,
            **kwargs
        )
        return gesture_frames
    
    def extract_gestures_by_time_bounds(self,
                                        start_bound: float = 0.0,
                                        end_bound: float | None = None,
                                        minimum_frames_per_gesture: int = 1,
                                        maximum_frames_per_gesture: int = 100,
                                        ):
        """Extract consecutive video frames containing hand gestures into gesture events."""
        start_bound, end_bound = self.set_bounds(start_bound, end_bound)
        gestures = defaultdict(lambda: [])
        gesture_index = 0
        last_gesture_n = None
        for n, gesture_frame in self.gesture_frames.items():
            timestamp = gesture_frame["timestamp"]
            if timestamp >= start_bound:
                # Break loop once out of bounds
                if end_bound is not None and timestamp > end_bound:
                    break
                
                # Create new gesture if no hand detected in previous frame and maximum_frames_per_gesture has not been reached
                if gesture_index == 0 or (last_gesture_n and (n - last_gesture_n) > 1):
                    gesture_index += 1
                elif (maximum_frames_per_gesture and len(gestures[gesture_index]) >= maximum_frames_per_gesture):
                    gesture_index += 1
                last_gesture_n = n
                
                # Add gesture frame to current gesture
                gestures[gesture_index].append(gesture_frame)
                
        # Filter out gestures appearing in fewer than minimum N frames (potential false positives)
        if minimum_frames_per_gesture > 0:
            gestures = {idx: gesture for idx, gesture in gestures.items() if len(gesture) >= minimum_frames_per_gesture}
        
        return gestures

    def search_for_gestures_near_specific_words(self,
                                                word_timings: list,
                                                max_window: float = 0.5,
                                                maximum_frames_per_gesture: int | None = None,
                                                minimum_frames_per_gesture: int = 10,
                                                combine_overlapping: bool = True,
                                                ):
        """Search for unique gesture events adjacent to specific words in speech recognition results."""
        gesture_events = []
        for i, entry in enumerate(word_timings):
            start_bound = max(0, entry[TOKEN_ONSET_KEY] - max_window)
            end_bound = entry[TOKEN_END_KEY] + max_window
            word = entry[TOKEN_KEY]
            logger.info(f"Searching for gestures near <{word}> (bounds: {start_bound}-{end_bound})")
            gestures = self.extract_gestures_by_time_bounds(
                start_bound=start_bound,
                end_bound=end_bound,
                maximum_frames_per_gesture=maximum_frames_per_gesture,
                minimum_frames_per_gesture=minimum_frames_per_gesture,
            )
            logger.info(f"Found {len(gestures)} gestures within bounds of <{word}> (bounds: {start_bound}-{end_bound})")
            gesture_events.append(gestures)
        if combine_overlapping:
            gesture_events = combine_overlapping_gestures(gesture_events)
        return gesture_events

    def dump_gesture_events(self, outfile, gesture_events=None):
        """Dump gesture event data to json outfile."""
        os.makedirs(os.path.dirname(outfile), exist_ok=True)
        if gesture_events is None:
            gesture_events = self.gesture_frames
        gesture_frames_serializable = deepcopy(gesture_events)
        for _, gesture_event in gesture_frames_serializable.items():
            gesture_event["hand_shape"] = {
                hand_landmark_i: coords.tolist()
                for hand_landmark_i, coords in enumerate(gesture_event["hand_shape"])
            }
        with open(outfile, "w") as outf:
            json.dump(gesture_frames_serializable, outf, indent=4)
