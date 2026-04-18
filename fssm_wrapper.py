"""Integration of F-SSM with the existing environment.

This module integrates the Fast Spectral Scan Matching (F-SSM) algorithm
with the existing environment that uses a Lidar sensor model and Unicycle motion model.
"""

import numpy as np
from fssm import FSSM

class FSsmWrapper:
    """Wrapper class to integrate F-SSM with the existing environment."""
    
    def __init__(self, bin_width=30, sigma_d=30):
        """Initialize the F-SSM wrapper.
        
        Args:
            bin_width (float): Width of distance bins in mm
            sigma_d (float): Parameter to control matching sensitivity in mm
        """
        self.fssm = FSSM(bin_width=bin_width, sigma_d=sigma_d)
    
    def convert_lidar_to_points(self, lidar_data):
        """Convert Lidar scan data to a 2D point array.
        
        Args:
            lidar_data (dict or np.ndarray): Lidar measurement data
                If dict: Expected to contain 'ranges' and 'angles' keys
                If ndarray: Expected to be Cartesian coordinates (x, y)
                
        Returns:
            np.ndarray: Array of 2D points in Cartesian coordinates, shape (n, 2)
        """
        if isinstance(lidar_data, dict):
            # If input is a dictionary, extract ranges and angles
            ranges = np.array(lidar_data.get('ranges', []))
            angles = np.array(lidar_data.get('angles', []))
            
            # Convert to Cartesian coordinates
            x = ranges * np.cos(angles)
            y = ranges * np.sin(angles)
            
            return np.column_stack((x, y))
        elif isinstance(lidar_data, np.ndarray) and lidar_data.ndim == 2:
            # If input is already a 2D array of points, return it
            return lidar_data
        else:
            raise ValueError("Unsupported lidar data format. Expected dict with 'ranges' and 'angles' or 2D point array")
    
    def match_scans(self, current_scan, reference_scan):
        """Match two Lidar scans using F-SSM.
        
        Args:
            current_scan (dict or np.ndarray): Current Lidar scan data
            reference_scan (dict or np.ndarray): Reference Lidar scan data
            
        Returns:
            tuple: (R, t, success)
                - R (np.ndarray): Rotation matrix (2x2)
                - t (np.ndarray): Translation vector (2,)
                - success (bool): Whether matching was successful
        """
        # Convert scans to point arrays
        current_points = self.convert_lidar_to_points(current_scan)
        reference_points = self.convert_lidar_to_points(reference_scan)
        
        # Filter out invalid points (e.g., max range or zero range)
        current_points = self.filter_invalid_points(current_points)
        reference_points = self.filter_invalid_points(reference_points)
        
        # Skip matching if not enough points
        if len(current_points) < 5 or len(reference_points) < 5:
            return np.eye(2), np.zeros(2), False
        
        # Perform scan matching
        try:
            R, t, correspondences = self.fssm.match(current_points, reference_points)
            success = len(correspondences) >= 5  # At least 5 correspondences for reliability
            return R, t, success
        except Exception as e:
            print(f"Error in scan matching: {e}")
            return np.eye(2), np.zeros(2), False
    
    def filter_invalid_points(self, points, min_dist=0.1, max_dist=30.0):
        """Filter out invalid points from a scan.
        
        Args:
            points (np.ndarray): Array of 2D points, shape (n, 2)
            min_dist (float): Minimum valid distance
            max_dist (float): Maximum valid distance
            
        Returns:
            np.ndarray: Filtered array of 2D points
        """
        # Calculate distances from origin for each point
        distances = np.linalg.norm(points, axis=1)
        
        # Keep points within valid range
        valid_idx = (distances >= min_dist) & (distances <= max_dist)
        return points[valid_idx]
    
    def estimate_pose_change(self, current_scan, reference_scan, prior_pose=None):
        """Estimate pose change between two scans.
        
        Args:
            current_scan (dict or np.ndarray): Current Lidar scan data
            reference_scan (dict or np.ndarray): Reference Lidar scan data
            prior_pose (np.ndarray, optional): Prior pose estimate [x, y, theta]
            
        Returns:
            tuple: (delta_pose, confidence)
                - delta_pose (np.ndarray): Estimated pose change [delta_x, delta_y, delta_theta]
                - confidence (float): Confidence value between 0 and 1
        """
        # Match scans
        R, t, success = self.match_scans(current_scan, reference_scan)
        
        if not success:
            # Return zero change with low confidence if matching failed
            return np.zeros(3), 0.0
        
        # Extract rotation angle from rotation matrix
        delta_theta = np.arctan2(R[1, 0], R[0, 0])
        
        # Create delta pose
        delta_pose = np.array([t[0], t[1], delta_theta])
        
        # Use prior pose if available to validate the match
        confidence = 0.8  # Default confidence
        if prior_pose is not None:
            prior_delta = prior_pose[:2] - t
            prior_confidence = np.exp(-np.linalg.norm(prior_delta) / 2.0)
            confidence = min(confidence, prior_confidence)
        
        return delta_pose, confidence
    
    def integrate_with_unicycle(self, motion_model, current_state, current_scan, reference_scan, dt):
        """Integrate F-SSM scan matching with unicycle motion model.
        
        Args:
            motion_model: Unicycle motion model object
            current_state (np.ndarray): Current robot state [x, y, theta]
            current_scan (dict or np.ndarray): Current Lidar scan data
            reference_scan (dict or np.ndarray): Reference Lidar scan data
            dt (float): Time step
            
        Returns:
            np.ndarray: Updated robot state [x, y, theta]
        """
        # Estimate pose change using scan matching
        delta_pose, confidence = self.estimate_pose_change(current_scan, reference_scan, current_state)
        
        # Skip update if confidence is too low
        if confidence < 0.3:
            return current_state
        
        # Extract pose components
        delta_x, delta_y, delta_theta = delta_pose
        
        # Calculate velocity and angular velocity based on pose change
        v = np.sqrt(delta_x**2 + delta_y**2) / dt
        omega = delta_theta / dt
        
        # Create action for unicycle model
        action = np.array([v, omega])
        
        # Update state using unicycle model
        updated_state = motion_model.step(current_state, action, dt)
        
        return updated_state
