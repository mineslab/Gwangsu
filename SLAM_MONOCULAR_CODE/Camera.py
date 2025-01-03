#Camera.py
import numpy as np
import cv2

def featureMapping(image):
    orb = cv2.ORB_create()
    # Convert the image to grayscale and detect good features to track
    gray_image = np.mean(image, axis=2).astype(np.uint8)
    pts = cv2.goodFeaturesToTrack(gray_image, 1000, qualityLevel=0.01, minDistance=7)
    if pts is None:
        # Handle cases where no features are detected
        return np.array([]), None

    # Create KeyPoint objects from detected points
    key_pts = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in pts]
    key_pts, descriptors = orb.compute(image, key_pts)

    # Return Key_points and ORB_descriptors
    return np.array([(kp.pt[0], kp.pt[1]) for kp in key_pts]), descriptors

def normalize(count_inv, pts):
    return np.dot(count_inv, np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1).T).T[:, 0:2]

def denormalize(count, pt):
    ret = np.dot(count, np.array([pt[0], pt[1], 1.0]))
    ret /= ret[2]
    return int(round(ret[0])), int(round(ret[1]))

Identity = np.eye(4)

class Camera(object):
    def __init__(self, desc_dict, image, count):
        self.count = count
        self.count_inv = np.linalg.inv(self.count)
        self.pose = Identity
        self.h, self.w = image.shape[0:2]
        
        # Generate keypoints and descriptors
        key_pts, self.descriptors = featureMapping(image)
        
        # Normalize keypoints
        self.key_pts = normalize(self.count_inv, key_pts) if key_pts.size > 0 else []
        self.pts = [None] * len(self.key_pts)
        self.id = len(desc_dict.frames)
        desc_dict.frames.append(self)
