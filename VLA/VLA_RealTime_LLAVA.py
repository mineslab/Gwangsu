#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import time
import os
import sys
import re
import termios
import tty
import select
import datetime
import ollama  # pip install ollama

from hiwonder_servo_msgs.msg import *

def getKey():
    """
    터미널에서 non-blocking 방식으로 키 입력을 받는 함수.
    입력이 없으면 빈 문자열을 반환합니다.
    """
    tty.setraw(sys.stdin.fileno())
    r, w, e = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if r else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def capture_image(image_path, cap):
    """
    카메라 캡처 객체를 사용해 이미지를 캡처하고 지정된 경로에 저장합니다.
    """
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
        rospy.loginfo(f"이미지 저장: {image_path}")
    else:
        rospy.logerr("이미지 캡처 실패")

def generate_movement_command(image_filename, previous_log, user_command):
    """
    현재 이미지와 이전 움직임 기록, 그리고 사용자가 입력한 명령을 참고하여 llava에게
    (서보아이디, 움직임량) 형식의 명령을 요청합니다.
    
    출력 예시: (6,+1),(5,-1)
    참고: 사용 가능한 서보는 id 1, 2, 3, 4, 5, 6이며, 각 서보의 위치 범위는 0~1000, 초기 위치는 500입니다.
    """
    available_servos_info = "사용 가능한 서보: 1, 2, 3, 4, 5, 6 (위치 범위: 0~1000, 초기값: 500)"
    
    prompt = (
        f"초기 명령: {user_command}\n"
        f"현재 이미지 파일: {image_filename}\n"
        f"이전 움직임 기록:\n{previous_log}\n"
        f"{available_servos_info}\n"
        "주어진 명령 수행을 위한 적절한 움직임을, 사진과 과거 움직임을 참고하여서 "
        "다음 형식으로 출력해 주세요: (서보아이디,움직임량) 예: (6,+1),(5,-1)"
    )
    
    messages = [
        {
            "role": "user",
            "content": prompt,
            "images": [image_filename]
        }
    ]
    
    # llava 모델에 요청 (모델이 다운로드되어 있어야 합니다: 'ollama pull llava')
    res = ollama.chat(model="llava", messages=messages)
    
    # 모델 응답에서 텍스트 명령 추출
    movement_command = res.get("message", {}).get("content", "").strip()
    return movement_command

def rotate(duration, id_pos_s):
    """
    ROS 메시지를 통해 서보 모터를 회전시키는 함수.
    
    Parameters:
      - duration: 동작 시간 (ms)
      - id_pos_s: (서보ID, 목표 위치) 쌍의 튜플들. 예: ((6, 200), (5, 800))
    """
    id_pos_dur_list = list(map(lambda x: RawIdPosDur(x[0], x[1], duration), id_pos_s))
    msg = MultiRawIdPosDur(id_pos_dur_list=id_pos_dur_list)
    
    time_sec = round(duration / 1000, 2)
    rospy.loginfo(f"Rotate servo(s) for {time_sec} second(s).")
    joints_pub.publish(msg)
    rospy.sleep(time_sec)

if __name__ == '__main__':
    # 터미널 설정 백업
    settings = termios.tcgetattr(sys.stdin)
    
    # ROS 초기화 및 퍼블리셔 생성
    rospy.init_node('vla_controller', anonymous=True)
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur',
                                 MultiRawIdPosDur,
                                 queue_size=1)
    
    # 로그 파일 및 이미지 저장 디렉토리 설정
    log_file_path = "movement_log.txt"
    image_log_dir = "image_logs"
    if not os.path.exists(image_log_dir):
        os.makedirs(image_log_dir)
    
    previous_log = ""
    step = 1

    # 초기 명령 입력 받기 (명령이 없으면 입력할 때까지 대기)
    user_command = input("초기 명령을 입력하세요: ").strip()
    while user_command == "":
        user_command = input("명령이 비어있습니다. 초기 명령을 입력하세요: ").strip()
    
    # llava 모델 초기화 (필요 시, ollama 문서 참조)
    # 예: ollama.init_model("llava")
    
    # 카메라 캡처 객체 생성 (기본 웹캠 사용)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("웹캠을 열 수 없습니다.")
        sys.exit(1)
    
    rospy.loginfo("=== VLA Controller Start (종료: 'q' 키) ===")
    
    try:
        while not rospy.is_shutdown():
            # 사용자가 'q' 키를 누르면 종료 (목표 달성 대신 임시 종료 조건)
            key = getKey()
            if key == 'q':
                rospy.loginfo("사용자에 의해 종료 요청됨.")
                break
            
            rospy.loginfo(f"Step {step} 시작")
            # 이미지 파일명 생성 (이미지 저장 경로)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = os.path.join(image_log_dir, f"step_{step}_{timestamp}.png")
            
            # Step 1: 현재 이미지 캡처
            capture_image(image_filename, cap)
            
            # Step 2: llava를 호출하여 움직임 명령 생성 (초기 명령 포함)
            movement_command = generate_movement_command(image_filename, previous_log, user_command)
            rospy.loginfo("생성된 움직임 명령: " + movement_command)
            
            # 로그 기록: step 번호, 이미지 파일명, 명령 내용
            log_entry = f"Step {step} [{timestamp}]: ({image_filename}) movement: {movement_command}\n"
            with open(log_file_path, "a") as log_file:
                log_file.write(log_entry)
            previous_log += log_entry
            
            # Step 3: llava의 출력값 파싱 후 ROS 액션 수행
            try:
                # 예상 출력 형식: (6,+1),(5,-1)
                pairs = re.findall(r'\((\d+),([+-]?\d+)\)', movement_command)
                servo_commands = []
                for servo_id_str, movement_str in pairs:
                    servo_id = int(servo_id_str)
                    movement_value = int(movement_str)
                    servo_commands.append((servo_id, movement_value))
                
                if servo_commands:
                    # 고정 동작 시간 (500ms)으로 서보 제어
                    rotate(500, tuple(servo_commands))
                else:
                    rospy.logwarn("움직임 명령 파싱 실패: 유효한 명령이 없습니다.")
            except Exception as e:
                rospy.logerr("움직임 명령 파싱 중 에러 발생: " + str(e))
            
            step += 1
            # 루프 간 짧은 대기 (필요에 따라 조정)
            time.sleep(0.5)
    
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("KeyboardInterrupt로 종료합니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rospy.loginfo("VLA Controller Finished.")
