"""Demonstration of F-SSM with the NEU racing environment.

This script demonstrates how to use the F-SSM implementation with
the NEU racing environment that uses a Lidar sensor and Unicycle motion model.
"""

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym

import gym_neu_racing.motion_models as motion_models
import gym_neu_racing.sensor_models as sensor_models
from fssm_wrapper import FSsmWrapper

def visualize_scan_matching(reference_scan, current_scan, R, t):
    """Visualize scan matching results.
    
    Args:
        reference_scan (np.ndarray): Reference scan points
        current_scan (np.ndarray): Current scan points
        R (np.ndarray): Rotation matrix
        t (np.ndarray): Translation vector
    """
    plt.figure(figsize=(10, 8))
    
    # Plot reference scan
    plt.scatter(reference_scan[:, 0], reference_scan[:, 1], c='blue', label='Reference Scan', alpha=0.5)
    
    # Plot current scan before transformation
    plt.scatter(current_scan[:, 0], current_scan[:, 1], c='red', label='Current Scan', alpha=0.5)
    
    # Transform current scan and plot
    transformed_scan = (R @ current_scan.T).T + t
    plt.scatter(transformed_scan[:, 0], transformed_scan[:, 1], c='green', label='Transformed Scan', alpha=0.5)
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.title('F-SSM Scan Matching Result')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.show()

def simulate_lidar_scan(env_map, pose, num_beams=360, max_range=10.0):
    """Simulate a Lidar scan at the given pose.
    
    Args:
        env_map: Map object of the environment
        pose (np.ndarray): Robot pose [x, y, theta]
        num_beams (int): Number of Lidar beams
        max_range (float): Maximum sensing range
        
    Returns:
        dict: Simulated Lidar scan data
    """
    angles = np.linspace(0, 2*np.pi, num_beams, endpoint=False)
    ranges = np.full(num_beams, max_range)
    
    # Cast rays
    for i, angle in enumerate(angles):
        # Calculate absolute angle
        abs_angle = pose[2] + angle
        
        # Calculate ray direction
        dir_x = np.cos(abs_angle)
        dir_y = np.sin(abs_angle)
        
        # Ray origin
        ray_origin = pose[:2]
        
        # Perform ray casting
        for dist in np.arange(0.1, max_range, 0.1):
            # Calculate point along ray
            point_x = ray_origin[0] + dist * dir_x
            point_y = ray_origin[1] + dist * dir_y
            point = np.array([point_x, point_y])
            
            # Check if point is in map and occupied
            grid_coords, in_map = env_map.world_coordinates_to_map_indices(point)
            
            if not in_map:
                # Ray has left the map
                ranges[i] = dist
                break
            
            if env_map.static_map[grid_coords[0], grid_coords[1]]:
                # Ray has hit an obstacle
                ranges[i] = dist
                break
    
    return {'ranges': ranges, 'angles': angles}

def extract_scan_points(lidar_data):
    """Extract 2D points from Lidar data.
    
    Args:
        lidar_data (dict): Lidar data with 'ranges' and 'angles'
        
    Returns:
        np.ndarray: Array of 2D points, shape (n, 2)
    """
    ranges = lidar_data['ranges']
    angles = lidar_data['angles']
    
    # Convert to Cartesian coordinates
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)
    
    return np.column_stack((x, y))

def demo_scan_matching_with_environment():
    """Demonstrate F-SSM scan matching with the NEU racing environment."""
    # Create environment
    env = gym.make('gym_neu_racing/NEURacing-v0')
    
    # Reset environment
    observation, _ = env.reset()
    
    # Get map from environment
    env_map = env.map
    
    # Create FSSM wrapper
    fssm_wrapper = FSsmWrapper(bin_width=30, sigma_d=30)
    
    # Simulate a reference scan at initial pose
    initial_pose = env.state.copy()
    reference_scan_data = simulate_lidar_scan(env_map, initial_pose)
    reference_scan_points = extract_scan_points(reference_scan_data)
    
    # Simulate movement with unicycle model
    motion_model = motion_models.Unicycle()
    action = np.array([0.5, 0.2])  # Forward velocity and angular velocity
    dt = 0.1
    new_pose = motion_model.step(initial_pose, action, dt)
    
    # Simulate a new scan at the new pose
    current_scan_data = simulate_lidar_scan(env_map, new_pose)
    current_scan_points = extract_scan_points(current_scan_data)
    
    # Perform scan matching
    R, t, success = fssm_wrapper.match_scans(current_scan_points, reference_scan_points)
    
    if success:
        print("Scan matching successful!")
        print(f"Rotation matrix:\n{R}")
        print(f"Translation vector: {t}")
        
        # Calculate estimated pose change
        delta_theta = np.arctan2(R[1, 0], R[0, 0])
        estimated_change = np.array([t[0], t[1], delta_theta])
        
        # Calculate actual pose change
        actual_change = new_pose - initial_pose
        
        print("\nPose Change Comparison:")
        print(f"Estimated: {estimated_change}")
        print(f"Actual: {actual_change[:3]}")
        print(f"Error: {np.linalg.norm(estimated_change - actual_change[:3])}")
        
        # Visualize scan matching
        visualize_scan_matching(reference_scan_points, current_scan_points, R, t)
    else:
        print("Scan matching failed")
    
    # Close environment
    env.close()

def demo_localization_with_fssm():
    """Demonstrate localization using F-SSM in the NEU racing environment."""
    # Create environment
    env = gym.make('gym_neu_racing/NEURacing-v0')
    
    # Reset environment
    observation, _ = env.reset()
    
    # Get map from environment
    env_map = env.map
    
    # Create motion and sensor models
    motion_model = motion_models.Unicycle()
    lidar_sensor = sensors.Lidar2D(env_map)
    
    # Create FSSM wrapper
    fssm_wrapper = FSsmWrapper(bin_width=30, sigma_d=30)
    
    # Initial pose and localization state
    true_pose = env.state.copy()
    estimated_pose = true_pose.copy()
    
    # Initial scan as reference
    reference_scan_data = simulate_lidar_scan(env_map, true_pose)
    
    # Create figure for visualization
    plt.figure(figsize=(12, 10))
    
    # Simulation steps
    num_steps = 20
    
    # Arrays to store data
    true_poses = [true_pose.copy()]
    estimated_poses = [estimated_pose.copy()]
    
    # Simulation loop
    for step in range(num_steps):
        # Apply a random action
        action = np.array([0.5, 0.1 * np.sin(step * 0.5)])  # Simple sinusoidal steering
        dt = 0.1
        
        # Update true pose
        true_pose = motion_model.step(true_pose, action, dt)
        
        # Simulate Lidar scan at true pose
        current_scan_data = simulate_lidar_scan(env_map, true_pose)
        
        # Estimate pose change using F-SSM
        delta_pose, confidence = fssm_wrapper.estimate_pose_change(
            current_scan_data, reference_scan_data, estimated_pose)
        
        # Update estimated pose
        estimated_pose[0] += delta_pose[0]
        estimated_pose[1] += delta_pose[1]
        estimated_pose[2] += delta_pose[2]
        
        # Normalize angle
        estimated_pose[2] = (estimated_pose[2] + np.pi) % (2 * np.pi) - np.pi
        
        # Store poses for plotting
        true_poses.append(true_pose.copy())
        estimated_poses.append(estimated_pose.copy())
        
        # Use current scan as reference for next iteration
        reference_scan_data = current_scan_data
        
        print(f"Step {step+1}/{num_steps}")
        print(f"True Pose: {true_pose}")
        print(f"Estimated Pose: {estimated_pose}")
        print(f"Estimation Error: {np.linalg.norm(true_pose - estimated_pose)}")
        print(f"Confidence: {confidence}")
        print("---")
    
    # Convert to numpy arrays for plotting
    true_poses = np.array(true_poses)
    estimated_poses = np.array(estimated_poses)
    
    # Plot result
    plt.subplot(1, 1, 1)
    plt.plot(true_poses[:, 0], true_poses[:, 1], 'b-', label='True Path')
    plt.plot(estimated_poses[:, 0], estimated_poses[:, 1], 'r--', label='Estimated Path')
    plt.scatter(true_poses[0, 0], true_poses[0, 1], c='g', s=100, marker='o', label='Start')
    plt.scatter(true_poses[-1, 0], true_poses[-1, 1], c='r', s=100, marker='x', label='End')
    
    # Draw environment map
    env_map.draw_map(show=False, ax=plt.gca())
    
    plt.legend()
    plt.title('Localization using F-SSM')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.show()
    
    # Close environment
    env.close()

if __name__ == "__main__":
    print("Running F-SSM scan matching demonstration...")
    demo_scan_matching_with_environment()
    
    try:
        import gym_neu_racing.sensors as sensors
        print("\nRunning F-SSM localization demonstration...")
        demo_localization_with_fssm()
    except ImportError:
        print("\nSkipping localization demo due to missing sensors module.")
        print("Please ensure you have the appropriate sensor model set up.")
