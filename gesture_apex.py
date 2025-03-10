from collections import defaultdict
from statistics import mean

import numpy as np


def compute_velocity(positions: np.array, times: np.array):
    """Compute instantaneous speed at each time step via discrete derivative."""
    velocities = np.linalg.norm(np.diff(positions, axis=0), axis=1) / np.diff(times)[:, np.newaxis]
    return velocities


def compute_acceleration(velocities: np.array, times: np.array):
    """Compute instantaneous acceleration at each time step via derivative of velocity."""
    accelerations = np.diff(velocities) / np.diff(times)[:, np.newaxis]
    return accelerations


def compute_direction_changes(velocities: np.array):
    """Compute angular changes between consecutive velocity vectors."""
    # Normalize velocity vectors
    unit_velocities = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
    dot_products = np.einsum('ij,ij->i', unit_velocities[:-1], unit_velocities[1:])
    dot_products = np.clip(dot_products, -1.0, 1.0)
    # Compute angular difference (radians)
    angles = np.arccos(dot_products)
    return angles


def find_apex(positions: np.array, times: np.array):
    """Identify gestural apex from arrays of 3D hand position coordinates and timestamps."""
    velocities = compute_velocity(positions, times)
    accelerations = compute_acceleration(velocities, times)

    # Identify candidates for the gestural apex
    # Sudden stop (minimum speed)
    speeds = np.linalg.norm(velocities, axis=1)
    min_speed_idx = np.argmin(speeds)

    # Sharp acceleration change
    accel_magnitudes = np.linalg.norm(accelerations, axis=1)  # Compute acceleration magnitudes
    max_accel_idx = np.argmax(accel_magnitudes)
    # Maximum extension of gesture from start
    max_extension_idx = np.argmax(np.linalg.norm(positions - positions[0], axis=(1, 2)))

    # Sharpest directional change
    direction_changes = compute_direction_changes(velocities)
    max_turn_idx = np.argmax(direction_changes) + 1  # Offset by 1 due to diff

    return (
        times[min_speed_idx],
        times[max_accel_idx],
        times[max_extension_idx],
        times[max_turn_idx]
    )


def detect_gesture_apices(gestures: dict, average=False) -> dict:
    """Return a dictionary of gesture indices with their apex timestamps according to 4 criteria:
    - minimum speed (sudden stop or change in direction),
    - maximum acceleration,
    - maximum extension from start,
    - sharpest directional change.
    """
    gesture_apices = defaultdict(dict)
    for idx, gesture_data in gestures.items():
        timestamps = np.array([entry['timestamp'] for entry in gesture_data])
        hand_shapes = np.array([entry['hand_shape'] for entry in gesture_data])
        min_speed_time, max_accel_time, max_extension_time, max_turn_time = find_apex(hand_shapes, timestamps)
        gesture_apices[idx]["min_speed_timestamp"] = min_speed_time
        gesture_apices[idx]["max_acceleration_timestamp"] = max_accel_time
        gesture_apices[idx]["max_extension_timestamp"] = max_extension_time
        gesture_apices[idx]["sharpest_direction_change_timestamp"] = max_turn_time

    if average:
        gesture_apices = {idx: mean(gesture_apex.values()) for idx, gesture_apex in gesture_apices.items()}

    return gesture_apices
