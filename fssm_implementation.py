"""Fast Spectral Scan Matching (F-SSM) implementation.

This module implements the F-SSM algorithm described in the paper:
'Coarse-to-Fine Localization for a Mobile Robot Based on Place Learning with a 2-D Range Scan'
by Soonyong Park and Kyung Shik Roh.

The F-SSM algorithm is an adaptation of the Fast Spectral Graph Matching (FASM) method
for scan matching. It approximates the affinity matrix using the linear combination of
Kronecker products of basis and index matrices for efficient computation.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

class FSSM:
    """Fast Spectral Scan Matching implementation."""
    
    def __init__(self, bin_width=30, sigma_d=30):
        """Initialize F-SSM with parameters.
        
        Args:
            bin_width (float): Width of distance bins in mm for approximate matching (w in paper)
            sigma_d (float): Parameter to control matching sensitivity in mm (σd in paper)
        """
        self.bin_width = bin_width
        self.sigma_d = sigma_d
    
    def approximate_distance(self, distance):
        """Approximate a distance value using bin width.
        
        This implements equation (2) from the paper.
        
        Args:
            distance (float): Original distance value
            
        Returns:
            float: Approximated distance value
        """
        bin_idx = int(distance / self.bin_width)
        return self.bin_width * (bin_idx + 0.5)
    
    def compute_affinity_score(self, d_ij, d_i_j_prime):
        """Compute affinity score between two pairs of points.
        
        This implements equation (4) from the paper.
        
        Args:
            d_ij (float): Distance between points i and j in first scan
            d_i_j_prime (float): Distance between points i' and j' in second scan
            
        Returns:
            float: Affinity score (0 if pairs are not compatible)
        """
        diff = abs(d_ij - d_i_j_prime)
        if diff < 3 * self.sigma_d:
            return 4.5 - (diff ** 2) / (2 * (self.sigma_d ** 2))
        return 0
    
    def create_basis_matrices(self, scan1, scan2):
        """Create basis matrices for the approximate affinity matrix.
        
        Args:
            scan1 (np.ndarray): First scan points, shape (n1, 2)
            scan2 (np.ndarray): Second scan points, shape (n2, 2)
            
        Returns:
            tuple: (bases, index_matrices)
                - bases: List of basis matrices
                - index_matrices: List of index matrices corresponding to bases
        """
        n1 = scan1.shape[0]
        n2 = scan2.shape[0]
        
        # Calculate pairwise distances within each scan
        # For scan1
        d1 = np.zeros((n1, n1))
        for i in range(n1):
            for j in range(n1):
                if i != j:
                    d1[i, j] = np.linalg.norm(scan1[i] - scan1[j])
        
        # For scan2
        d2 = np.zeros((n2, n2))
        for i in range(n2):
            for j in range(n2):
                if i != j:
                    d2[i, j] = np.linalg.norm(scan2[i] - scan2[j])
        
        # Approximate distances
        d1_approx = np.zeros_like(d1)
        for i in range(n1):
            for j in range(n1):
                if i != j:
                    d1_approx[i, j] = self.approximate_distance(d1[i, j])
        
        # Find unique approximated distances in scan1
        unique_distances = np.unique(d1_approx[d1_approx > 0])
        
        # Create bases and index matrices
        bases = []
        index_matrices = []
        
        for dist in unique_distances:
            # Create basis matrix B_i for this distance
            B_i = np.zeros((n2, n2))
            for i in range(n2):
                for j in range(n2):
                    if i != j:
                        B_i[i, j] = self.compute_affinity_score(dist, d2[i, j])
            
            # Create index matrix H_i for this distance
            H_i = np.zeros((n1, n1))
            for i in range(n1):
                for j in range(n1):
                    if i != j and d1_approx[i, j] == dist:
                        H_i[i, j] = 1
            
            # Only add non-zero bases and corresponding index matrices
            if np.sum(B_i) > 0:
                bases.append(B_i)
                index_matrices.append(H_i)
        
        return bases, index_matrices
    
    def bases_power_method(self, bases, index_matrices, max_iter=100, tol=1e-6):
        """Implement the bases power method with bistochastic normalization.
        
        This implements Algorithm 1 from the paper.
        
        Args:
            bases (list): List of basis matrices
            index_matrices (list): List of index matrices
            max_iter (int): Maximum number of iterations
            tol (float): Convergence tolerance
            
        Returns:
            np.ndarray: Assignment matrix V
        """
        n1 = index_matrices[0].shape[0]
        n2 = bases[0].shape[0]
        
        # Initialize v with random values and normalize
        v = np.random.rand(n1 * n2)
        v = v / np.linalg.norm(v)
        
        # Power method iterations
        for _ in range(max_iter):
            # Matrix-vector multiplication
            z = np.zeros_like(v)
            for basis, index_matrix in zip(bases, index_matrices):
                # Implement equation (6) from the paper
                for j in range(n1):
                    for x in range(n1):
                        if index_matrix[x, j] > 0:
                            v_j = v[j*n2:(j+1)*n2]
                            z_x = z[x*n2:(x+1)*n2]
                            z[x*n2:(x+1)*n2] = z_x + index_matrix[x, j] * basis @ v_j
            
            # Normalize the vector
            z_norm = np.linalg.norm(z)
            if z_norm < tol:
                break
            z = z / z_norm
            
            # Bistochastic normalization (Sinkhorn algorithm)
            V = z.reshape(n1, n2)
            
            # Apply bistochastic normalization until convergence
            prev_V = None
            for _ in range(50):  # Limit iterations for normalization
                # Normalize rows
                row_sums = V.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # Avoid division by zero
                V = V / row_sums
                
                # Normalize columns
                col_sums = V.sum(axis=0, keepdims=True)
                col_sums[col_sums == 0] = 1  # Avoid division by zero
                V = V / col_sums
                
                # Check convergence
                if prev_V is not None and np.allclose(V, prev_V, atol=tol):
                    break
                prev_V = V.copy()
            
            # Reshape V back to vector v
            v = V.flatten()
            
            # Check convergence
            if np.allclose(v, z, atol=tol):
                break
        
        # Return assignment matrix V
        return v.reshape(n1, n2)
    
    def match(self, scan1, scan2):
        """Match two range scans using F-SSM.
        
        Args:
            scan1 (np.ndarray): First scan points, shape (n1, 2)
            scan2 (np.ndarray): Second scan points, shape (n2, 2)
            
        Returns:
            tuple: (R, t, correspondences)
                - R (np.ndarray): Rotation matrix (2x2)
                - t (np.ndarray): Translation vector (2,)
                - correspondences (list): List of tuples (i, j) representing correspondences
                  between points in scan1 and scan2
        """
        # Create basis and index matrices
        bases, index_matrices = self.create_basis_matrices(scan1, scan2)
        
        # Compute assignment matrix using bases power method
        assignment_matrix = self.bases_power_method(bases, index_matrices)
        
        # Apply Hungarian algorithm to find discrete assignments
        row_indices, col_indices = linear_sum_assignment(-assignment_matrix)
        
        # Create correspondence pairs
        correspondences = [(i, j) for i, j in zip(row_indices, col_indices)]
        
        # Estimate transformation using RANSAC
        R, t = self.ransac_estimation(scan1, scan2, correspondences)
        
        return R, t, correspondences
    
    def ransac_estimation(self, scan1, scan2, correspondences, max_iter=100, threshold=0.1):
        """RANSAC-based estimation of transformation between scans.
        
        Args:
            scan1 (np.ndarray): First scan points, shape (n1, 2)
            scan2 (np.ndarray): Second scan points, shape (n2, 2)
            correspondences (list): List of correspondence pairs
            max_iter (int): Maximum number of RANSAC iterations
            threshold (float): Inlier threshold distance
            
        Returns:
            tuple: (R, t)
                - R (np.ndarray): Rotation matrix (2x2)
                - t (np.ndarray): Translation vector (2,)
        """
        best_R = np.eye(2)
        best_t = np.zeros(2)
        max_inliers = 0
        
        for _ in range(max_iter):
            # Randomly select two correspondence pairs
            if len(correspondences) < 2:
                continue
                
            idx1, idx2 = np.random.choice(len(correspondences), 2, replace=False)
            i1, j1 = correspondences[idx1]
            i2, j2 = correspondences[idx2]
            
            # Get the corresponding points
            p1 = scan1[i1]
            q1 = scan2[j1]
            p2 = scan1[i2]
            q2 = scan2[j2]
            
            # Compute tentative transformation
            # This implements equation (7) from the paper
            # First, calculate the rotation angle
            dp = p2 - p1
            dq = q2 - q1
            
            if np.linalg.norm(dp) < 1e-6 or np.linalg.norm(dq) < 1e-6:
                continue
                
            theta = np.arctan2(dp[1], dp[0]) - np.arctan2(dq[1], dq[0])
            
            # Compute rotation matrix
            c, s = np.cos(theta), np.sin(theta)
            R = np.array([[c, -s], [s, c]])
            
            # Compute translation
            t = p1 - R @ q1
            
            # Count inliers
            inliers = 0
            for i, j in correspondences:
                error = np.linalg.norm(scan1[i] - (R @ scan2[j] + t))
                if error < threshold:
                    inliers += 1
            
            # Update best transformation if we have more inliers
            if inliers > max_inliers:
                max_inliers = inliers
                best_R = R
                best_t = t
        
        # Refine transformation using all inliers
        inlier_src = []
        inlier_dst = []
        
        for i, j in correspondences:
            error = np.linalg.norm(scan1[i] - (best_R @ scan2[j] + best_t))
            if error < threshold:
                inlier_src.append(scan1[i])
                inlier_dst.append(scan2[j])
        
        if len(inlier_src) >= 2:
            # Recalculate transformation with all inliers using SVD
            inlier_src = np.array(inlier_src)
            inlier_dst = np.array(inlier_dst)
            
            # Center the point sets
            src_mean = np.mean(inlier_src, axis=0)
            dst_mean = np.mean(inlier_dst, axis=0)
            
            src_centered = inlier_src - src_mean
            dst_centered = inlier_dst - dst_mean
            
            # Compute rotation using SVD
            H = src_centered.T @ dst_centered
            U, _, Vt = np.linalg.svd(H)
            R = U @ Vt
            
            # Ensure proper rotation matrix (determinant 1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = U @ Vt
            
            # Compute translation
            t = src_mean - R @ dst_mean
            
            best_R = R
            best_t = t
        
        return best_R, best_t
    
    def transform_scan(self, scan, R, t):
        """Transform a scan using rotation and translation.
        
        Args:
            scan (np.ndarray): Scan points, shape (n, 2)
            R (np.ndarray): Rotation matrix (2x2)
            t (np.ndarray): Translation vector (2,)
            
        Returns:
            np.ndarray: Transformed scan, shape (n, 2)
        """
        return (R @ scan.T).T + t
