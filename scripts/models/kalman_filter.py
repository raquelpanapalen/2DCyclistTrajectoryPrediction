import numpy as np

class KalmanFilter:
    def __init__(self, dt, initial_coords, stdacc, xstdmeas, ystdmeas):
        """
        # stdacc: process noise
        # xstdmeas: standard deviation of x
        # ystdmeas: standard deviation of y
        """
        # state is [x, y, dx, dy]
        self.x = np.matrix([[initial_coords[0]], [initial_coords[1]], [0], [0]])
        self.dt = dt
        """
        Matrix A represents the fact that the position (x and y) of the bounding box 
        changes linearly with time at a constant velocity (vx and vy), 
        while the velocity itself remains constant.
        """
        # Transition Matrix
        self.A = np.array([
            [1, 0, self.dt, 0],  # x 0 dx 0
            [0, 1, 0, self.dt],  # 0 y 0 dy
            [0, 0, 1, 0],  # 0 0 vx 0
            [0, 0, 0, 1]   # 0 0 0 vy
        ]) 
        
        # Observation Matrix
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Noise
        self.Q = np.matrix([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, (self.dt**2), 0],
                            [0, (self.dt**3)/2, 0, (self.dt**2)]]) * stdacc**2
        
        # Noise Covariance Matrix
        self.R = np.matrix([[xstdmeas**2, 0],
                            [0, ystdmeas**2]])
        
        #Initial Process Covariance Matrix
        self.P = np.eye(self.A.shape[1])
        

    def predict(self):
        # Compute state t=k:  x_k = A*x_{k-1} + B*a_{k-1}
        self.x = np.dot(self.A, self.x)
        
        # Compute error covariance P = A*P*A' + Q
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        
        return self.x, self.P

    
    def update(self, z):
        # z is the observation
        z = [[z[0]], [z[1]]]
        
        # S = H*P*H' + R       
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        
        # Kalman Gain K = P*H'*S^(-1)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # Update state
        y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.x.shape[0])

        #Update error covariance
        self.P = (I - (K * self.H)) * self.P
