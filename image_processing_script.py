import cv2
import numpy as np
import numpy.linalg as la
from scipy.spatial import distance as dist
from icecream import ic
import timeit




class AR():
    def __init__(self):
        # Create KLT tracker with default parameters
        self.tracker = cv2.SparsePyrLKOpticalFlow_create()
        self.qr_detector = cv2.QRCodeDetector()
        self.prev_points = None
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.tracking_points = None
        self.tracking_status = None
        self.pts_src = None
        
        self.K= 1000 * np.array([[1.1211 , 0       , 0.3587],
                            [0      , 1.1236  , 0.6379],
                            [0      , 0       , 0.0010]])

        #  Square coordiantes in real world
        # Assuming SQUARE is ordered as [TL, TR, BR, BL]
        self.SQUARE = 0.01 * np.array([[0, 0, 0],
                                [14.5, 0, 0],
                                [14.5, 14.5, 0],
                                [0, 14.5, 0]])

        self.HEIGHT = 0.005 * 14.5
        #      7------6
        #     /|     /|
        #    / |    / |
        #   4------5  |
        #   |  3---|--2
        #   | /    | /
        #   |/     |/
        #   0------1
        # 3D coordinates of the cube in its local coordinate system (assuming it starts at the origin)

        self.CUBE_POINTS_3D = np.array([
            [0, 0, 0],        # Point 0: Origin
            [self.HEIGHT, 0, 0],   # Point 1: Base corner on the x-axis
            [self.HEIGHT, self.HEIGHT, 0], # Point 2: Base corner diagonally opposite the origin on the x-y plane
            [0, self.HEIGHT, 0],   # Point 3: Base corner on the y-axis
            [0, 0, self.HEIGHT],   # Point 4: Top corner above the origin
            [self.HEIGHT, 0, self.HEIGHT], # Point 5: Top corner above point 1
            [self.HEIGHT, self.HEIGHT, self.HEIGHT], # Point 6: Top corner diagonally opposite the origin in x-y-z space
            [0, self.HEIGHT, self.HEIGHT]  # Point 7: Top corner above point 3
        ])

                
    def process_frame(self, image):
        self.image = image  # Set the current frame to self.image
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.tracking_points is not None and len(self.tracking_points) >= 4:
            self.update_tracking(gray_frame)
            self.pts_src = self.tracking_points
            if self.pts_src is not None:
                H = self.compute_homography()
                R, t = self.decompose_homography(H)
                self.Cube_points_2D = self.project_points(R, t)
                return self.draw_cube()

        detected_points = self.detect(gray_frame)
        if detected_points is not None:
            self.tracking_points = detected_points
            self.prev_gray = gray_frame.copy()

        return image




            
        
    def compute_homography(self):

        self.pts_src_normalized = [([x, y, 1]) for (x, y) in self.pts_src]
        
        # Set up matrix A using the correspondences
        A = []
        for i in range(4):
            X, Y, Z = self.SQUARE[i]
            x, y = self.pts_src_normalized[i][0], self.pts_src_normalized[i][1]
            A.append([-X, -Y, -1, 0, 0, 0, x*X, x*Y, x])
            A.append([0, 0, 0, -X, -Y, -1, y*X, y*Y, y])
        A = np.array(A)



        try:
            A= np.append(A, np.zeros(len(A[0])))
            A = A.reshape((9,9))
            A[-1,-1] = 1
            p = np.zeros(len(A)-1)
            p= np.append(p, 1)
            H = la.inv(A).dot(p)
            H = H.reshape(3,3)
        except:
                    # Compute SVD of A
            U, S, Vt = la.svd(A)
            # The homography is the last column of Vt (or V in np.linalg.svd)
            h = Vt[-1]
            H = h.reshape((3, 3))
        H = H / H[2, 2]



        return H


    def decompose_homography(self,H):
        # Invert the camera calibration matrix
        K_inv = la.inv(self.K)
        
        # Multiply by the homography to get the initial rotation and translation matrix
        Rt = K_inv @ H
        
        # Normalize the first two columns to get the first two rotation vectors
        r1 = Rt[:, 0] / la.norm(Rt[:, 0])
        r2 = Rt[:, 1] / la.norm(Rt[:, 1])
        
        # The third rotation vector is the cross product of the first two
        r3 = np.cross(r1, r2)
        r3 = r3 / la.norm(r3)  # Ensure it is a unit vector
        
        # Construct the initial rotation matrix
        R = np.column_stack([r1, r2, r3])
        
        # Correct the rotation matrix to ensure it is a proper rotation matrix
        # by making the determinant +1
        if la.det(R) < 0:
            R[:, 2] = -R[:, 2]  # Negate the third column if determinant is negative
        
        # Scale the rotation matrix if the determinant is not 1
        det_R = la.det(R)
        if not np.isclose(det_R, 1) and det_R != 0:
            scale_factor = np.cbrt(1 / det_R)
            R = R * scale_factor
        
        # Extract the translation vector, and normalize it
        t = Rt[:, 2]
        t = t / la.norm(R @ t)  # Scale translation to match the scale of rotation
        
        return R, t




    def preprocess_image_for_detection(self):
        """Apply basic preprocessing steps such as conversion to grayscale and blurring."""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return blurred
    
    
    
    def update_tracking(self, gray_frame):
        new_points, status, _ = self.tracker.calc(self.prev_gray, gray_frame, self.tracking_points, None)
        if new_points is not None and status is not None:
            new_points = new_points[status.flatten() == 1]
            if len(new_points) >= 4:
                self.tracking_points = new_points
            else:
                self.tracking_points = None
        self.prev_gray = gray_frame.copy()

    def detect(self, gray_frame):
        _, _, points, _ = self.qr_detector.detectAndDecodeMulti(gray_frame)
        if points is not None:
            return np.float32(points).reshape(-1, 2)
        return None





    def project_points(self,R, t):
        # Transform 3D points to 2D using the camera intrinsic matrix, rotation and translation
        projected_points = []
        for point in   self.CUBE_POINTS_3D:
            # Convert to homogeneous coordinates
            point_hom = np.append(point, 1)

            point_img_hom = self.K @ np.column_stack([R, t]) @ point_hom
            point_img = point_img_hom[:2] / point_img_hom[2]
            projected_points.append(point_img)
        self.Cube_points_2D = np.array(projected_points)

        return self.Cube_points_2D







    def draw_cube(self):
        cube_points = self.Cube_points_2D
        # Draw the base of the cube
        for i in range(4):
            point1 = tuple(cube_points[i % 4][:2].astype(int))
            point2 = tuple(cube_points[(i + 1) % 4][:2].astype(int))
            cv2.line(self.image, point1, point2, (0, 255, 0), 2)

        # Draw the top of the cube
        for i in range(4):
            point1 = tuple(cube_points[4 + i % 4][:2].astype(int))
            point2 = tuple(cube_points[4 + (i + 1) % 4][:2].astype(int))
            cv2.line(self.image, point1, point2, (0, 255, 0), 2)

        # Draw vertical lines of the cube
        for i in range(4):
            point1 = tuple(cube_points[i][:2].astype(int))
            point2 = tuple(cube_points[i + 4][:2].astype(int))
            cv2.line(self.image, point1, point2, (0, 255, 0), 2)

        # Draw the square in red
        self.pts_src_int = self.pts_src.astype(int)
        cv2.polylines(self.image, [self.pts_src_int], True, (0, 0, 255), 2)

        return self.image


def main():
    ar = AR()
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_image = ar.process_frame(frame)
        timeit.timeit()
        cv2.imshow('QR Code Detection and Tracking', processed_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()