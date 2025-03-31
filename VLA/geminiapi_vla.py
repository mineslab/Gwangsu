#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import cv2
import time
import os
import sys
import termios
import tty
import select
import datetime
import re
import pathlib
import google.generativeai as genai
from hiwonder_servo_msgs.msg import MultiRawIdPosDur, RawIdPosDur
from std_msgs.msg import String

def getKey():
    """
    터미널에서 non-blocking 방식으로 키 입력을 받는 함수.
    입력이 없으면 빈 문자열을 반환합니다.
    """
    tty.setraw(sys.stdin.fileno())
    r, _, _ = select.select([sys.stdin], [], [], 0.1)
    key = sys.stdin.read(1) if r else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

def capture_image(cap, image_path):
    """
    카메라 캡처 객체를 사용해 이미지를 캡처하고 지정된 경로에 저장합니다.
    저장된 이미지 파일의 경로를 반환합니다.
    """
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(image_path, frame)
        rospy.loginfo("이미지 저장: " + image_path)
        return image_path
    else:
        rospy.logerr("이미지 캡처 실패")
        return None

def get_image_data(image_path):
    """
    지정된 이미지 경로에서 bytes 데이터를 읽어 딕셔너리 형태로 반환합니다.
    """
    return {
        'mime_type': 'image/png',
        'data': pathlib.Path(image_path).read_bytes()
    }

def parse_movement_command(text):
    """
    텍스트에서 (서보아이디,움직임량) 형식의 명령을 추출하여 리스트로 반환합니다.
    예: "(6,+1),(5,-1)" -> [(6, +1), (5, -1)]
    """
    pairs = re.findall(r'\((\d+),([+-]?\d+)\)', text)
    return [(int(sid), int(mv)) for sid, mv in pairs]

def rotate(duration, servo_commands):
    """
    ROS 메시지를 통해 서보 모터를 회전시키는 함수.
    
    Parameters:
      - duration: 동작 시간 (ms)
      - servo_commands: (서보ID, 움직임량) 쌍의 튜플 리스트, 예: [(6, +1), (5, -1)]
    """
    id_pos_dur_list = [RawIdPosDur(sid, mv, duration) for sid, mv in servo_commands]
    msg = MultiRawIdPosDur(id_pos_dur_list=id_pos_dur_list)
    time_sec = round(duration / 1000.0, 2)
    rospy.loginfo("서보 모터를 {}초 동안 회전합니다.".format(time_sec))
    joints_pub.publish(msg)
    rospy.sleep(time_sec)

if __name__ == '__main__':
    # 터미널 설정 백업
    settings = termios.tcgetattr(sys.stdin)
    
    # ROS 노드 초기화 및 퍼블리셔 생성
    rospy.init_node('robot_movement_controller', anonymous=True)
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur',
                                 MultiRawIdPosDur,
                                 queue_size=1)
    command_pub = rospy.Publisher('/robot_movement_command', String, queue_size=1)
    
    # 로그 파일 및 이미지 저장 디렉토리 설정
    log_file_path = "movement_log.txt"
    image_dir = "image_logs"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
    
    # 초기 사용자 명령 입력
    user_command = input("초기 명령을 입력하세요: ").strip()
    while not user_command:
        user_command = input("명령이 비어있습니다. 초기 명령을 입력하세요: ").strip()
    
    # 1. API 키 설정 및 모델/채팅 세션 시작
    genai.configure(api_key="AIzaSyBR0GWqGa_wS1zl1TJPzKBjSLqy3qhH7mg")
    rospy.loginfo("사용 중인 API Key: " + "AIzaSyBR0GWqGa_wS1zl1TJPzKBjSLqy3qhH7mg")
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    chat = model.start_chat()
    
    # 첫 번째 프롬트 정의 (초기 대화)
    first_prompt = (
        "* 서보 1: 로봇 암의 수직 이동 (높이)\n"
        "* 서보 2: 로봇 암의 수평 이동 (앞뒤)\n"
        "* 서보 3: 로봇 암의 수평 이동 (좌우)\n"
        "* 서보 4: 로봇 암의 회전 (손목)\n"
        "* 서보 5: 그리퍼의 열림/닫힘\n"
        "* 서보 6: 로봇 암의 기울기 (상하)\n"
        "사용 가능한 서보: 1, 2, 3, 4, 5, 6 (위치 범위: 0~1000, 초기값: 500)\n"
        "사진과 과거 움직임 기록을 참고하여 주어진 초기 명령을 수행하기 위한\n"
        "앞에 있는 것을 잡아주라.\n"
        "적절한 로봇 이동 명령을 다음 형식으로 출력해 주세요: (서보아이디,움직임량) 예: (6,+1),(5,-1)\n"
        "위를 이해했으면 예라고 대답해."
    )
    
    # 첫 번째 이미지 캡처 및 전송
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(image_dir, f"step_0_{timestamp}.png")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        rospy.logerr("웹캠을 열 수 없습니다.")
        sys.exit(1)
    
    first_image_path = capture_image(cap, image_path)
    if first_image_path is None:
        rospy.logerr("첫 이미지 캡처 실패")
        sys.exit(1)
    
    image_data = get_image_data(first_image_path)
    rospy.loginfo("첫 번째 프롬트 전송 중...")
    response = chat.send_message([first_prompt, image_data])
    rospy.loginfo("초기 응답: " + response.text)
    
    # 반복 프롬트 정의 (대화 후속)
    repeated_prompt = (
        "이전의 명령을 따르기 위한 명령을 내려줘.\n"
        "현재 상태를 고려하여 명령을 만들어주라.\n"
        "사진과 과거 움직임 기록을 참고하여 주어진 초기 명령을 수행하기 위한\n"
        "적절한 로봇 이동 명령을 다음 형식으로 출력해 주세요: (서보아이디,움직임량) 예: (6,+1),(5,-1)"
    )
    
    i = 0
    rospy.loginfo("=== Robot Movement Controller Start (종료: 'q' 키) ===")
    
    try:
        while not rospy.is_shutdown():
            key = getKey()
            if key == 'q':
                rospy.loginfo("사용자에 의해 종료 요청됨.")
                break
            
            # 매 반복마다 새 이미지 캡처
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(image_dir, f"step_{i+1}_{timestamp}.png")
            captured_image_path = capture_image(cap, image_path)
            if captured_image_path is None:
                continue
            
            image_data = get_image_data(captured_image_path)
            start_time = time.time()
            # 반복 프롬트와 새 이미지 전송 (대화 세션 유지)
            response = chat.send_message([repeated_prompt, image_data])
            end_time = time.time()
            elapsed_time = end_time - start_time
            rospy.loginfo(f"{i} {response.text} (응답 시간: {elapsed_time:.2f}초)")
            
            # 응답 텍스트에서 로봇 이동 명령 파싱
            servo_commands = parse_movement_command(response.text)
            if servo_commands:
                rotate(500, servo_commands)
            else:
                rospy.logwarn("움직임 명령 파싱 실패: 유효한 명령이 없습니다.")
            
            # ROS 토픽에 명령 발행 및 로그 기록
            command_pub.publish(response.text)
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Step {i+1} [{timestamp}]: {response.text} (응답 시간: {elapsed_time:.2f}초)\n")
            
            i += 1
            time.sleep(0.5)
    
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("KeyboardInterrupt로 종료합니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rospy.loginfo("Robot Movement Controller Finished.")



앞에 있는 박스를 잡아서 동쪽 위에 박스를 냅두도록 명령을 내려주라. 과거의 움직임들을 참고해서 적절한 명령을 내려주라.


예: [0.2371 -0.0200 0.0542 -2.2272 0.0013 2.2072 -0.9670 0.04 0.04]
 "이전의 명령을 따르기 위한 명령을 내려줘.\n"
        "현재 상태를 고려하여 명령을 만들어주라.\n"
        "사진과 과거 움직임 기록을 참고하여 주어진 초기 명령을 수행하기 위한\n"
        "적절한 로봇 이동 명령을 다음 형식으로 출력해 주세요: (서보아이디,움직임량) 예: [0.2371 -0.0200 0.0542 -2.2272 0.0013 2.2072 -0.9670 0.04 0.04]"
