#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import sys, select, termios, tty
import cv2
import os
import re
import datetime

# llava 모델과 ollama 라이브러리 사용 (llava 모델의 세부 구현은 ollama 라이브러리 내에 있다고 가정)
import ollama  # ollama 라이브러리를 통해 llava 모델 제어

from hiwonder_servo_msgs.msg import *

def generate_movement_command(current_image_path, previous_movements, user_command):
    """
    현재 캡처된 이미지와 과거의 움직임 기록, 그리고 사용자 명령을 입력받아
    llava 모델을 통해 (서보아이디, 움직임량) 형식의 움직임 명령을 생성합니다.
    
    Parameters:
      - current_image_path: 현재 캡처된 이미지의 파일명
      - previous_movements: 지금까지의 움직임 로그 (문자열 리스트)
      - user_command: 사용자로부터 입력받은 명령 (텍스트)
    
    Returns:
      - llava 모델이 생성한 움직임 명령 문자열, 예: "(6,+1),(5,-1)"
    """
    prompt = (
        f"현재 사진: {current_image_path}\n"
        f"과거의 움직임:\n{''.join(previous_movements)}\n"
        f"명령: {user_command}\n"
        "주어진 명령 수행을 위한 적절한 움직임을 사진과 과거의 움직임을 참고하여서 "
        "다음 형식으로 출력하십시오: (서보아이디,움직임량), 예: (6,+1),(5,-1)"
    )
    
    # ollama 라이브러리를 통해 llava 모델에 프롬프트 전달 (실제 함수명은 ollama 라이브러리 문서를 참고)
    result = ollama.generate(prompt)
    return result

def capture_image(save_path):
    """
    카메라에서 이미지를 캡처하여 지정된 경로에 저장합니다.
    """
    cap = cv2.VideoCapture(0)  # 기본 카메라 사용 (필요시 인덱스 변경)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(save_path, frame)
        rospy.loginfo(f"이미지 저장: {save_path}")
    else:
        rospy.logerr("이미지 캡처 실패")
    cap.release()

def getKey():
    """
    non-blocking 방식으로 터미널에서 키 입력을 받습니다.
    """
    tty.setraw(sys.stdin.fileno())
    r, w, e = select.select([sys.stdin], [], [], 0.1)
    if r:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def rotate(duration, id_pos_s):
    """
    ROS 메시지를 통해 서보 모터들을 회전시키는 함수입니다.
    
    Parameters:
      - duration: 동작 지속 시간 (ms)
      - id_pos_s: (서보ID, 목표 위치) 쌍의 튜플들. 예: ((6, 200), (5, 800))
    """
    id_pos_dur_list = list(map(lambda x: RawIdPosDur(x[0], x[1], duration), id_pos_s))
    msg = MultiRawIdPosDur(id_pos_dur_list=id_pos_dur_list)
    time_sec = round(duration / 1000, 2)
    rospy.loginfo('Rotate servo(s) for {} second(s).'.format(time_sec))
    joints_pub.publish(msg)
    rospy.sleep(time_sec)

if __name__ == '__main__':
    # 터미널 설정 백업
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('vla_controller', anonymous=True)
    
    # ROS 서보 퍼블리셔 (실제 토픽 이름과 포트 확인 필요)
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur',
                                 MultiRawIdPosDur,
                                 queue_size=1)
    
    # 로그 파일 및 이미지 저장 폴더 설정
    log_file_path = "movement_log.txt"
    image_log_dir = "image_logs"
    if not os.path.exists(image_log_dir):
        os.makedirs(image_log_dir)
    
    previous_movements = []  # 이전 단계의 로그를 저장할 리스트

    # llava 모델 초기화 (ollama 라이브러리 사용)
    ollama.init_model("llava")  # 실제 초기화 방식은 ollama 문서 참조
    
    # Step 2: 사용자로부터 명령 입력 받기
    user_command = input("명령을 입력하세요: ")
    
    step = 1
    goal_achieved = False
    
    while not rospy.is_shutdown() and not goal_achieved:
        rospy.loginfo("Step {} 시작".format(step))
        
        # Step 3: 현재 사진 캡처
        image_filename = f"image_step_{step}.png"
        image_path = os.path.join(image_log_dir, image_filename)
        capture_image(image_path)
        
        # llava 모델을 통해 움직임 명령 생성
        movement_output = generate_movement_command(image_filename, previous_movements, user_command)
        rospy.loginfo("생성된 움직임 명령: " + movement_output)
        
        # 로그 기록 (예: step 2 : (image2.png) movement : (6,+1),(5,-1))
        log_entry = f"Step {step}: ({image_filename}) movement: {movement_output}\n"
        with open(log_file_path, "a") as log_file:
            log_file.write(log_entry)
        previous_movements.append(log_entry)
        
        # Step 4: 생성된 명령을 파싱하여 실제 서보 액션 수행
        try:
            # 예상 출력 형식: (6,+1),(5,-1)
            pairs = re.findall(r'\((\d+),([+-]?\d+)\)', movement_output)
            servo_commands = []
            for servo_id_str, movement_str in pairs:
                servo_id = int(servo_id_str)
                movement_value = int(movement_str)
                # 여기서는 movement_value를 서보의 절대 위치값으로 가정합니다.
                servo_commands.append((servo_id, movement_value))
            
            if servo_commands:
                # 고정된 지속 시간 (500ms)으로 서보 움직임 수행
                rotate(500, tuple(servo_commands))
            else:
                rospy.logwarn("움직임 명령 파싱 실패: 명령이 비어있습니다.")
        except Exception as e:
            rospy.logerr("움직임 명령 파싱 중 에러 발생: " + str(e))
        
        # Step 5: 목표 달성 여부 확인 (사용자에게 입력 요청)
        response = input("목표 달성하였습니까? (y/n): ")
        if response.lower() == 'y':
            goal_achieved = True
        
        step += 1

    # 종료 시 터미널 설정 복구
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    rospy.loginfo("VLA operation finished.")
