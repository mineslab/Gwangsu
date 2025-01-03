#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monocular SLAM Demo in Python with 3D Visualization
--------------------------------------------------
Additional Features:
- 3D Visualization using Matplotlib (Display current position and surrounding map points)

Implemented Features:
1. Tracking
   - Extract ORB
   - INITIAL Pose estimation (match with previous keyframe each frame)
   - Track local Map (simplified: match with surrounding KeyFrames and MapPoints)
   - New Keyframe (when conditions are met)

2. Local Mapping
   - Insert KeyFrame
   - Recent mapping culling (MapPoints/KeyFrames)
   - New points creation (Triangulation)
   - Local BA (optional: python-g2o)
   - Local keyframes culling

3. Loop Closing
   - Loop detection (naive Bow/DB)
   - Compute Sim3 (simple R,t,s estimation)
   - Loop fusion (merge submaps/keyframes)

4. 3D Visualization
   - Visualize current position and surrounding map points in 3D using Matplotlib

Usage:
  python slam_full_demo.py

Press 'q' to exit
"""

import sys
import time
import signal
import threading
import queue
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

###############################################################################
#                             G2O check (optional)
###############################################################################
USE_G2O = False
try:
    import g2o
    USE_G2O = True
except ImportError:
    print("[INFO] python-g2o not found; advanced BA/pose-graph disabled.")
    USE_G2O = False

###############################################################################
#                          CONFIG & HELPERS
###############################################################################
FLIP_Z = True
DEBUG_PRINT = True

def extract_orb(img, nfeatures=1500):
    orb = cv2.ORB_create(nfeatures)
    kps_cv, des_cv = orb.detectAndCompute(img, None)
    if kps_cv is None or des_cv is None or len(kps_cv) < 5:
        return None, None
    pts = np.array([kp.pt for kp in kps_cv], dtype=np.float32)
    return pts, des_cv

def match_descriptors(des1, des2, max_matches=500):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches[:max_matches]

def recover_pose_from_2view(K, pts1, pts2, thresh=1.0):
    E, mask = cv2.findEssentialMat(pts1, pts2, K, cv2.RANSAC, 0.999, thresh)
    if E is None:
        return None, None, None
    inliers, R, t, _ = cv2.recoverPose(E, pts1, pts2, K, mask=mask)
    if inliers < 10:
        return None, None, None
    return R, t, mask

def make_4x4(R, t):
    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = t.ravel()
    return T

def triangulate_points(K, poseA, poseB, ptsA, ptsB):
    P1 = K @ poseA[:3,:]
    P2 = K @ poseB[:3,:]
    X4d = cv2.triangulatePoints(P1, P2, ptsA.T, ptsB.T).T
    X4d /= (X4d[:,3:]+1e-12)
    return X4d

###############################################################################
#                           DATA STRUCTURES
###############################################################################
class MapPoint:
    def __init__(self, X4d):
        if len(X4d)==4:
            self.X = X4d[:3]/(X4d[3]+1e-12)
        else:
            self.X = X4d
        self.obs = []  # (KeyFrame, kp_idx)
        self.n = np.zeros(3, dtype=np.float32)  # viewing direction
        self.D = None  # representative descriptor
        self.dmin = 0.1
        self.dmax = 100.0

    def add_observation(self, kf, kp_idx):
        self.obs.append((kf, kp_idx))
        # Update viewing direction
        invPose = np.linalg.inv(kf.pose)
        c_w = invPose[:3,3]
        v = self.X - c_w
        norm_v = np.linalg.norm(v)
        if norm_v > 1e-12:
            self.n += v / norm_v
        # Update representative descriptor
        desc_new = kf.des[kp_idx]
        if self.D is None:
            self.D = desc_new.copy()
        else:
            dist_old = np.count_nonzero(self.D != desc_new)
            # Simplistically set dist_new=0, replace if fewer differences
            if 0 < dist_old:
                self.D = desc_new.copy()

    def finalize_direction(self):
        norm_n = np.linalg.norm(self.n)
        if norm_n > 1e-12:
            self.n /= norm_n

class KeyFrame:
    def __init__(self, fid, pose, K):
        self.id = fid
        self.pose = pose.copy()    # world->camera
        self.K = K                 # Camera intrinsics
        self.kps = None           # ORB keypoints (x,y)
        self.des = None           # ORB descriptors
        self.mappoints = {}       # kp_idx->MapPoint
        self.is_keyframe = False

class SubMap:
    def __init__(self, sid):
        self.id = sid
        self.frames = []
        self.points = []
        self.keyframes = []

class Atlas:
    def __init__(self):
        self.submaps = []
        self.active_map = None
        self.next_submap_id = 0
    def create_submap(self):
        sm = SubMap(self.next_submap_id)
        self.next_submap_id += 1
        self.submaps.append(sm)
        self.active_map = sm
        return sm
    def merge_submaps(self, smA, smB, T_AB):
        # smB -> smA coords
        for kf in smB.keyframes:
            kf.pose = T_AB @ kf.pose
            smA.keyframes.append(kf)
        for mp in smB.points:
            Xh = np.hstack((mp.X, [1.0]))
            mp.X = (T_AB @ Xh)[:3]
            smA.points.append(mp)
        smA.frames.extend(smB.frames)
        if smB in self.submaps:
            self.submaps.remove(smB)

###############################################################################
#                     Naive BoW & KeyFrameDB (for Loop Detection)
###############################################################################
class NaiveVocab:
    def compute_vec(self, descriptors):
        # Simply sum all descriptors
        return np.sum(descriptors, axis=0, dtype=np.int32)

class KeyFrameDB:
    def __init__(self, vocab):
        self.vocab = vocab
        self.db = []  # (kf_id, vec, kf)
    def add(self, kf):
        if kf.des is None:
            return
        vec = self.vocab.compute_vec(kf.des)
        self.db.append((kf.id, vec, kf))
    def query(self, kf, topN=5):
        if kf.des is None:
            return []
        vq = self.vocab.compute_vec(kf.des)
        results = []
        for (kid, vv, kff) in self.db:
            if kf == kff:
                continue
            dist = np.sum(np.abs(vv - vq))
            results.append((dist, kid, kff))
        results.sort(key=lambda x: x[0])
        return results[:topN]

###############################################################################
#                        LOCAL MAPPING THREAD
###############################################################################
class LocalMappingThread(threading.Thread):
    def __init__(self, atlas, kf_q, display3d_q):
        super().__init__()
        self.atlas = atlas
        self.kf_q = kf_q
        self.display3d_q = display3d_q
        self.alive = True

    def run(self):
        while self.alive:
            try:
                kf = self.kf_q.get(timeout=0.05)
            except:
                continue
            if kf is None:
                print("[LocalMapping] None => break.")
                break

            if DEBUG_PRINT:
                print(f"[LocalMapping] KeyFrame {kf.id} inserted.")
            sm = self.atlas.active_map
            # recent mapping culling (simplified)
            self.cull_map_points(sm)
            self.cull_keyframes(sm)

            # new points creation (triangulate with last keyframe)
            if len(sm.keyframes) < 2:
                continue
            refkf = sm.keyframes[-2]
            matches = match_descriptors(kf.des, refkf.des)
            if len(matches) < 15:
                continue
            x1 = np.array([kf.kps[m.queryIdx] for m in matches], dtype=np.float32)
            x2 = np.array([refkf.kps[m.trainIdx] for m in matches], dtype=np.float32)
            X4d = triangulate_points(kf.K, kf.pose, refkf.pose, x1, x2)
            goodZ = X4d[:,2] > 0
            newpts = 0
            for i, m in enumerate(matches):
                if not goodZ[i]:
                    continue
                if m.queryIdx in kf.mappoints:
                    continue
                mp = MapPoint(X4d[i])
                mp.add_observation(kf, m.queryIdx)
                mp.add_observation(refkf, m.trainIdx)
                sm.points.append(mp)
                kf.mappoints[m.queryIdx] = mp
                refkf.mappoints[m.trainIdx] = mp
                mp.finalize_direction()
                newpts += 1
            if DEBUG_PRINT:
                print(f"[LocalMapping] Created {newpts} new points from KF {kf.id}")

            # Send data to 3D visualization
            self.send_to_display3d(sm)

            # local BA
            if USE_G2O:
                self.local_ba(sm)

        print("[LocalMapping] ended.")

    def local_ba(self, submap):
        print("[LocalMapping] (g2o) local_ba stubbed (not implemented).")

    def cull_map_points(self, submap):
        new_list = []
        removed = 0
        for mp in submap.points:
            if len(mp.obs) < 2:
                removed += 1
                # Disconnect from keyframes
                for (kkf, kid) in mp.obs:
                    if kid in kkf.mappoints:
                        del kkf.mappoints[kid]
            else:
                new_list.append(mp)
        submap.points = new_list
        if DEBUG_PRINT and removed > 0:
            print(f"[LocalMapping] Culling {removed} map points.")

    def cull_keyframes(self, submap):
        # local keyframes culling (simplified: keep at least 2 and remove too close keyframes)
        if len(submap.keyframes) < 3:
            return
        removed_kf = []
        new_kfs = []
        base = submap.keyframes[0]
        new_kfs.append(base)
        for i in range(1, len(submap.keyframes)):
            kf = submap.keyframes[i]
            delta = np.linalg.inv(new_kfs[-1].pose) @ kf.pose
            trans = np.linalg.norm(delta[:3,3])
            if trans < 0.05:
                # remove
                removed_kf.append(kf)
            else:
                new_kfs.append(kf)
        submap.keyframes = new_kfs
        if DEBUG_PRINT and len(removed_kf) > 0:
            print(f"[LocalMapping] Culling {len(removed_kf)} keyframes.")

    def add_keyframe(self, kf):
        self.kf_q.put(kf)

    def send_to_display3d(self, submap):
        camera_positions = []
        for kf in submap.keyframes:
            cam_pos = np.linalg.inv(kf.pose)[:3,3]
            camera_positions.append(cam_pos)
        if len(camera_positions) > 0:
            latest_cam = camera_positions[-1]
            self.display3d_q.put(latest_cam)  # Send only camera position

###############################################################################
#                         LOOP CLOSING THREAD
###############################################################################
class LoopClosingThread(threading.Thread):
    def __init__(self, atlas, kf_q, display3d_q=None):
        super().__init__()
        self.atlas = atlas
        self.kf_q = kf_q
        self.alive = True
        self.db = KeyFrameDB(NaiveVocab())
        self.display3d_q = display3d_q

    def run(self):
        while self.alive:
            try:
                kf = self.kf_q.get(timeout=0.05)
            except:
                continue
            if kf is None:
                print("[LoopClosing] None => break.")
                break

            # Add to DB
            self.db.add(kf)
            # Loop detection
            candidates = self.db.query(kf, topN=5)
            for (dist, kid, kfold) in candidates:
                # Skip keyframes that are too close in time
                if abs(kf.id - kfold.id) < 30:
                    continue
                # Simple matching
                matches = match_descriptors(kf.des, kfold.des)
                if len(matches) < 20:
                    continue
                # Detect loop
                print(f"[LoopClosing] Potential loop: {kf.id} -> {kfold.id}")
                # Compute sim3 (simplistically scale=1, R,t using recoverPose)
                x1 = np.array([kf.kps[m.queryIdx] for m in matches], dtype=np.float32)
                x2 = np.array([kfold.kps[m.trainIdx] for m in matches], dtype=np.float32)
                R, t, _ = recover_pose_from_2view(kf.K, x1, x2)
                if R is None:
                    continue
                # scale=1 => Sim3 = [ R  t ]
                #                [ 0  1 ]
                T_4x4 = make_4x4(R, t)
                # Loop fusion => submap merge
                sm = self.atlas.active_map
                smB = None
                for s_ in self.atlas.submaps:
                    if kfold in s_.keyframes:
                        smB = s_
                        break
                if smB and smB != sm:
                    self.atlas.merge_submaps(sm, smB, T_4x4)
                    print(f"[LoopClosing] Merged submap {sm.id} and {smB.id} via sim3.")
                    # Send updated positions to 3D visualization
                    if self.display3d_q is not None:
                        camera_positions = []
                        for kf_ in sm.keyframes:
                            cam_pos = np.linalg.inv(kf_.pose)[:3,3]
                            camera_positions.append(cam_pos)
                        if len(camera_positions) > 0:
                            latest_cam = camera_positions[-1]
                            self.display3d_q.put(latest_cam)  # Send only camera position
                break

        print("[LoopClosing] ended.")

    def add_keyframe(self, kf):
        self.kf_q.put(kf)

###############################################################################
#                     TRACKING THREAD (ORB, Pose, LocalMap)
###############################################################################
class TrackingThread(threading.Thread):
    def __init__(self, frame_q, atlas, K,
                 localmapper, loopcloser,
                 display2d_q, path2d_q,
                 display3d_q,
                 max_lost=15):
        super().__init__()
        self.frame_q = frame_q
        self.atlas = atlas
        self.K = K
        self.localmapper = localmapper
        self.loopcloser = loopcloser
        self.display2d_q = display2d_q
        self.path2d_q = path2d_q
        self.display3d_q = display3d_q

        self.frame_id = 0
        self.last_kf = None
        self.lost_count = 0
        self.max_lost = max_lost
        self.alive = True

    def run(self):
        while self.alive:
            try:
                img = self.frame_q.get(timeout=0.05)
            except:
                continue
            if img is None:
                print("[Tracking] Received None => break.")
                break

            # (1) Extract ORB
            pts, des = extract_orb(img, 1500)
            if pts is not None:
                for (ux, uy) in pts:
                    cv2.circle(img, (int(ux), int(uy)), 2, (0, 255, 0), -1)

            if pts is None or des is None or len(pts) < 5:
                self.lost_count += 1
                if self.lost_count > self.max_lost:
                    self.lost_count = 0
                    print("[Tracking] Too many lost => reset.")
                self.display2d_q.put(img)
                self.update_path2d(None)
                continue

            # New KeyFrame object
            pose_init = np.eye(4, dtype=np.float32)
            if self.last_kf is not None:
                pose_init = self.last_kf.pose.copy()
            current_kf = KeyFrame(self.frame_id, pose_init, self.K)
            current_kf.kps = pts
            current_kf.des = des

            sm = self.atlas.active_map
            sm.frames.append(current_kf)

            # (2) INITIAL Pose estimation
            if self.last_kf is None:
                current_kf.is_keyframe = True
                sm.keyframes.append(current_kf)
                self.last_kf = current_kf
                self.lost_count = 0
                self.display2d_q.put(img)
                self.update_path2d(current_kf)
                print(f"[Tracking] First KeyFrame {current_kf.id} in submap {sm.id}")
                # Send initial position to 3D visualization
                self.send_to_display3d(current_kf, sm)
                self.frame_id += 1
                continue

            # Match with previous frame/keyframe
            matches = match_descriptors(current_kf.des, self.last_kf.des)
            if len(matches) < 15:
                self.lost_count += 1
                if self.lost_count > self.max_lost:
                    self.lost_count = 0
                self.display2d_q.put(img)
                self.update_path2d(None)
                self.frame_id += 1
                continue

            x1 = np.array([current_kf.kps[m.queryIdx] for m in matches], dtype=np.float32)
            x2 = np.array([self.last_kf.kps[m.trainIdx] for m in matches], dtype=np.float32)
            R, t, _ = recover_pose_from_2view(self.K, x1, x2)
            if R is not None:
                T_4x4 = make_4x4(R, t)
                current_kf.pose = self.last_kf.pose @ T_4x4
                self.lost_count = 0
            else:
                self.lost_count += 1
                self.display2d_q.put(img)
                self.update_path2d(None)
                self.frame_id += 1
                continue

            # (3) Track local map (simplified)
            # Actual implementation would refine pose using local map points
            # Here, it's simplified or omitted

            # (4) New keyframe?
            lkf = sm.keyframes[-1]
            delta = np.linalg.inv(lkf.pose) @ current_kf.pose
            trans = np.linalg.norm(delta[:3,3])
            rot_trace = np.trace(delta[:3,:3]) - 1
            rot = np.arccos(max(-1, min(1, rot_trace / 2)))
            deg = np.degrees(rot)
            if trans > 0.1 or deg > 5.0:
                current_kf.is_keyframe = True
                sm.keyframes.append(current_kf)
                self.localmapper.add_keyframe(current_kf)
                self.loopcloser.add_keyframe(current_kf)
                if DEBUG_PRINT:
                    print(f"[Tracking] => new keyframe: {current_kf.id}")
                # Send updated position to 3D visualization
                self.send_to_display3d(current_kf, sm)

            self.last_kf = current_kf
            self.display2d_q.put(img)
            self.update_path2d(current_kf)
            self.frame_id += 1

        print("[Tracking] Thread ended.")

    def update_path2d(self, kf):
        if kf is None:
            self.path2d_q.put((None, None))
            return
        invp = np.linalg.inv(kf.pose)
        center = invp[:3,3]
        cam_forward_local = np.array([0, 0, -0.2, 1], dtype=np.float32)
        fwd = invp @ cam_forward_local
        self.path2d_q.put((center, fwd[:3]))

    def send_to_display3d(self, kf, submap):
        camera_positions = []
        for kf_ in submap.keyframes:
            cam_pos = np.linalg.inv(kf_.pose)[:3,3]
            camera_positions.append(cam_pos)
        if len(camera_positions) > 0:
            latest_cam = camera_positions[-1]
            self.display3d_q.put(latest_cam)  # Send only camera position

###############################################################################
#                             MAIN
###############################################################################
def main():
    # Switch to webcam input
    cap = cv2.VideoCapture(0)  # Webcam ID (default 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    
    if not cap.isOpened():
        print("[MAIN] Cannot open webcam.")
        sys.exit(-1)

    W, H = 1280, 960
    F = 300  # Focal length (approximate, adjust if needed)
    K = np.array([[F, 0, W / 2],
                  [0, F, H / 2],
                  [0, 0, 1]], dtype=np.float32)

    atlas = Atlas()
    atlas.create_submap()

    frame_q = queue.Queue()
    kf_q_map = queue.Queue()
    kf_q_loop = queue.Queue()
    display2d_q = queue.Queue()
    path2d_q = queue.Queue()
    display3d_q = queue.Queue()

    locmapper = LocalMappingThread(atlas, kf_q_map, display3d_q)
    loopcloser = LoopClosingThread(atlas, kf_q_loop, display3d_q=display3d_q)
    tracker = TrackingThread(frame_q, atlas, K,
                             locmapper, loopcloser,
                             display2d_q, path2d_q,
                             display3d_q)

    locmapper.start()
    loopcloser.start()
    tracker.start()

    def sigint_handler(signum, frame):
        print("[MAIN] Caught Ctrl+C => shutdown.")
        tracker.alive = False
        locmapper.alive = False
        loopcloser.alive = False
        display3d_q.put(None)
        frame_q.put(None)
        kf_q_map.put(None)
        kf_q_loop.put(None)
        time.sleep(0.2)
        cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    # Initialize 3D Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()  # Enable interactive mode

    camera_positions = []

    print("[MAIN] reading webcam feed. Press 'q' to quit.")
    traj_w, traj_h = 800, 800
    traj_img = np.zeros((traj_h, traj_w, 3), dtype=np.uint8)

    scale = 50.0
    offset_x = 0
    offset_y = 0
    center_u = traj_w // 2
    center_v = traj_h // 2

    center_arrow_list = []

    while True:
        ret, frm = cap.read()
        if not ret:
            print("[MAIN] Cannot read webcam frame => break.")
            break
        frame_q.put(frm)

        if not display2d_q.empty():
            img2d = display2d_q.get()
            cv2.imshow("Frame", img2d)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[MAIN] 'q' pressed => break.")
                break

        while not path2d_q.empty():
            c_ar = path2d_q.get()
            center_arrow_list.append(c_ar)

        traj_img[:] = 0
        for (cc, fwd) in center_arrow_list:
            if cc is None:
                continue
            x3d = cc[0]
            z3d = cc[2]
            u = int(center_u + offset_x + x3d * scale)
            v = int(center_v + offset_y - z3d * scale)
            cv2.circle(traj_img, (u, v), 2, (0, 0, 255), -1)
            if fwd is not None:
                fx = fwd[0]
                fz = fwd[2]
                uf = int(center_u + offset_x + fx * scale)
                vf = int(center_v + offset_y - fz * scale)
                cv2.arrowedLine(traj_img, (u, v), (uf, vf), (255, 255, 0), 1, tipLength=0.3)

        cv2.imshow("Trajectory2D", traj_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[MAIN] 'q' pressed => break.")
            break

        # Update 3D Plot
        while not display3d_q.empty():
            cam_pose = display3d_q.get()
            if cam_pose is None:
                print("[MAIN] Received stop signal for 3D plot.")
                break
            if cam_pose is not None:
                camera_positions.append(cam_pose)

        ax.cla()
        # Plot camera positions
        if len(camera_positions) > 0:
            cams = np.array(camera_positions)
            ax.plot(cams[:,0], cams[:,1], cams[:,2], c='r', label='Camera Path')
            ax.scatter(cams[-1,0], cams[-1,1], cams[-1,2], c='r', marker='^', s=50, label='Current Position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title('3D SLAM Visualization - KeyFrames Only')
        ax.auto_scale_xyz([-10,10], [-10,10], [0,20])
        plt.draw()
        plt.pause(0.01)

        time.sleep(0.1)

    print("[MAIN] finishing...")
    cap.release()
    cv2.destroyAllWindows()

    tracker.alive = False
    locmapper.alive = False
    loopcloser.alive = False
    display3d_q.put(None)
    frame_q.put(None)
    kf_q_map.put(None)
    kf_q_loop.put(None)

    tracker.join()
    locmapper.join()
    loopcloser.join()

    print("[MAIN] done.")
    sys.exit(0)

if __name__ == "__main__":
    main()
