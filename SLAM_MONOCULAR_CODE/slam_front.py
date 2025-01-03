import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def extract_features_and_keypoints(img, detector):
    """
    ORB(또는 다른 특징 추출기)를 사용해 특징점과 디스크립터를 추출
    """
    keypoints, descriptors = detector.detectAndCompute(img, None)
    return keypoints, descriptors

def match_features(desc1, desc2, matcher):
    """
    두 프레임의 디스크립터를 매칭
    (가장 간단히 BFMatcher 또는 FLANN 등을 사용)
    """
    matches = matcher.knnMatch(desc1, desc2, k=2)

    # ratio test로 매칭 걸러내기
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    return good_matches

def get_essential_matrix(kps1, kps2, K):
    """
    매칭된 특징점들로부터 에센셜 매트릭스를 추정
    kps1, kps2: 매칭된 keypoints (numpy array of shape (N,2))
    K: 카메라 내외부 파라미터(3x3)
    """
    # opencv의 findEssentialMat을 이용
    E, mask = cv2.findEssentialMat(kps1, kps2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    return E, mask

def recover_camera_pose(E, kps1, kps2, K):
    """
    에센셜 매트릭스로부터 R, t를 추정
    """
    # recoverPose는 E, 특징점들, 카메라내부행렬을 입력으로 R, t 추정
    _, R, t, mask = cv2.recoverPose(E, kps1, kps2, K)
    return R, t, mask

def triangulate_points(kps1, kps2, R, t, K):
    """
    두 프레임 간 매칭된 특징점들을 3D 좌표로 삼각측량
    - P1 = K[I|0], P2 = K[R|t]
    """
    # 투영행렬 구성
    P1 = np.hstack((np.eye(3), np.zeros((3,1))))      # [I | 0]
    P2 = np.hstack((R, t))                           # [R | t]
    P1 = K @ P1
    P2 = K @ P2

    # Homogeneous 좌표로 결과가 나오므로 (X, Y, Z, W), 나중에 나누기
    points_4d_hom = cv2.triangulatePoints(P1, P2, kps1.T, kps2.T)
    points_4d = points_4d_hom / points_4d_hom[3]      # (4 x N)
    return points_4d[:3, :].T  # (N, 3)

def transform_points(points_3d, R, t):
    """
    points_3d (N,3)에 대해, R, t 변환 적용
    X_world = R * X_local + t
    """
    return (R @ points_3d.T).T + t.ravel()

def main():
    if len(sys.argv) < 2:
        print("사용법: python3 slam_front.py <video_path>")
        sys.exit(1)
    video_path = sys.argv[1]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("비디오를 열 수 없습니다.")
        sys.exit(1)

    # --- 파라미터 설정 ---
    # 간단히 ORB 사용
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # 카메라 내부 파라미터 (예시 값)
    # fx, fy = 800, 800
    # cx, cy = 320, 240
    K = np.array([[800,   0, 320],
                  [  0, 800, 240],
                  [  0,   0,   1]], dtype=np.float64)
    
    # 전역 좌표계에서의 카메라 위치(초기값): (R=I, t=0)
    # world_T_camera = [R|t] 형식으로 저장
    global_R = np.eye(3, dtype=np.float64)
    global_t = np.zeros((3,1), dtype=np.float64)
    
    # 카메라가 지나온 경로를 저장 (x, z)만 저장해서 2D 플롯에 표시
    trajectory = []
    
    # 전역 맵 포인트를 저장
    map_points = []
    
    # 첫 프레임 처리
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 번째 프레임을 읽을 수 없습니다.")
        sys.exit(1)
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_kps, prev_desc = extract_features_and_keypoints(prev_gray, orb)
    
    # Matplotlib으로 경로 시각화할 figure 생성
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title("Camera Trajectory and Map Points (2D top-down view)")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True)
    
    # 초기 카메라 위치(0,0)로 가정
    trajectory.append((0.0, 0.0))
    
    frame_idx = 1
    while True:
        # 매 프레임을 위해 사용자가 키를 눌러야 함
        print(f"[INFO] Frame {frame_idx}: 다음 프레임으로 넘어가려면 아무 키나 누르세요.")
        key = cv2.waitKey(0)  # 키 입력 대기
        if key == 27:  # ESC 누르면 종료
            print("사용자가 ESC를 눌러 종료합니다.")
            break
    
        ret, frame = cap.read()
        if not ret:
            print("모든 프레임을 재생했습니다.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kps, desc = extract_features_and_keypoints(gray, orb)
    
        if len(prev_kps) < 5 or len(kps) < 5:
            # 특징점이 너무 적으면 다음으로
            prev_kps, prev_desc = kps, desc
            continue
    
        # 매칭
        good_matches = match_features(prev_desc, desc, bf)
        if len(good_matches) < 5:
            print("유효한 매칭이 적어 Pose 추정이 불가합니다.")
            prev_kps, prev_desc = kps, desc
            continue
    
        # 매칭된 keypoint 좌표 뽑기
        matched_pts1 = np.array([prev_kps[m.queryIdx].pt for m in good_matches], dtype=np.float32)
        matched_pts2 = np.array([kps[m.trainIdx].pt for m in good_matches], dtype=np.float32)
        
        # 에센셜 행렬 추정 -> R, t 추정
        E, mask_E = get_essential_matrix(matched_pts1, matched_pts2, K)
        if E is None:
            print("에센셜 매트릭스 추정 실패.")
            prev_kps, prev_desc = kps, desc
            continue
        
        R, t, mask_pose = recover_camera_pose(E, matched_pts1, matched_pts2, K)
        
        # inlier만 추출 (RANSAC mask)
        inlier_idx = (mask_pose.ravel() == 1)
        matched_pts1_inlier = matched_pts1[inlier_idx]
        matched_pts2_inlier = matched_pts2[inlier_idx]

        if len(matched_pts1_inlier) < 5:
            print("인라이어 매칭이 너무 적어 Pose 추정이 불가합니다.")
            prev_kps, prev_desc = kps, desc
            continue
    
        # 삼각측량
        points_3d_local = triangulate_points(matched_pts1_inlier, matched_pts2_inlier, R, t, K)
    
        # 전역 좌표 업데이트
        # 새로 얻은 R, t는 이전 프레임(=카메라 좌표계) 기준
        # 전역 좌표계에서의 R, t로 변환: 
        #   global_T_new = global_T_old * cam_T_new
        #   cam_T_new = [R|t] (로컬)
        #   world_T_new = [global_R * R | global_R * t + global_t]
        new_global_R = global_R @ R
        new_global_t = global_R @ t + global_t

        # 3D 포인트를 월드 좌표로도 변환
        points_3d_world = transform_points(points_3d_local, new_global_R, new_global_t)

        # 카메라 위치(world 좌표계)
        camera_pos_world = new_global_t.ravel()
        # 2D 플롯에선 (x,z)만 사용
        trajectory.append((camera_pos_world[0], camera_pos_world[2]))

        # 맵 포인트에 추가
        map_points.extend(points_3d_world)

        # 그래프 갱신
        ax.clear()
        ax.set_title("Camera Trajectory and Map Points (2D top-down view)")
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.grid(True)

        # 카메라 궤적
        traj_x = [p[0] for p in trajectory]
        traj_z = [p[1] for p in trajectory]
        ax.plot(traj_x, traj_z, '-ro', label='Camera Trajectory')

        # 맵 포인트
        if len(map_points) > 0:
            map_points_arr = np.array(map_points)
            ax.scatter(map_points_arr[:,0], map_points_arr[:,2], s=2, c='b', alpha=0.5, label='Map Points')

        ax.legend()
        plt.draw()
        plt.pause(0.01)

        # 상태 업데이트
        global_R = new_global_R
        global_t = new_global_t
        prev_kps, prev_desc = kps, desc
        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
