#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import sys, select, termios, tty
from hiwonder_servo_msgs.msg import *

def getKey():
    """
    터미널에서 키보드 입력을 non-blocking 방식으로 받아오는 함수.
    입력이 없으면 ''(빈 문자열)을 반환.
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
    서보 모터를 회전시키는 함수.
    - duration : 서보 모터의 회전 시간 (범위 : 0 ms ~ 30000 ms)
    - id_pos_s: ((id, position), (id, position), ...) 튜플
       * id : 서보 모터의 ID
       * position : 서보 모터의 위치 (범위 : 0 ~ 1000)
    """
    id_pos_dur_list = list(map(lambda x: RawIdPosDur(x[0], x[1], duration), id_pos_s))
    msg = MultiRawIdPosDur(id_pos_dur_list=id_pos_dur_list)

    time_sec = round(duration / 1000, 2)  # 단위: s
    rospy.loginfo('Rotate servo(s) for {} second(s).'.format(time_sec))

    joints_pub.publish(msg)
    rospy.sleep(time_sec)

if __name__ == '__main__':
    # 키보드 세팅(터미널 설정) 백업
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('servo_test_arm_only', anonymous=True)
    
    # 서보 퍼블리셔 (사용하는 포트와 토픽 이름 확인 필요)
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur',
                                 MultiRawIdPosDur,
                                 queue_size=1)
    
    # 굴착기(포크레인) 작동 방식을 본딴 '팔(Arm)' 제어 예시
    # 실제 서보 ID와 동작 범위(0~1000) 값에 맞춰 수정 필수
    commands = {
        # Base(하부 회전) 예시
        'q': {
            'duration': 500,
            'id_pos_s': ((6, 200),)  # 예: 왼쪽 회전
        },
        'e': {
            'duration': 500,
            'id_pos_s': ((6, 800),)  # 예: 오른쪽 회전
        },

        # Shoulder / Boom(어깨/붐) 예시
        'w': {
            'duration': 500,
            'id_pos_s': ((2, 200),)  # 붐/어깨 올리기
        },
        's': {
            'duration': 500,
            'id_pos_s': ((2, 800),)  # 붐/어깨 내리기
        },

        # Arm / Elbow(암/팔꿈치) 예시
        'a': {
            'duration': 500,
            'id_pos_s': ((3, 200),)  # 팔꿈치 접기
        },
        'd': {
            'duration': 500,
            'id_pos_s': ((3, 800),)  # 팔꿈치 펴기
        },

        # Bucket(버켓) 예시
        'z': {
            'duration': 500,
            'id_pos_s': ((4, 200),)  # 버켓 닫기
        },
        'x': {
            'duration': 500,
            'id_pos_s': ((4, 800),)  # 버켓 열기
        },
    }

    rospy.loginfo("=== ROS Arm Teleop Start ===")
    rospy.loginfo("조작 키:")
    rospy.loginfo("  q/e : 베이스 좌/우 회전")
    rospy.loginfo("  w/s : 어깨(붐) 올리기/내리기")
    rospy.loginfo("  a/d : 팔꿈치(암) 굽히기/펴기")
    rospy.loginfo("  z/x : 버켓 닫기/열기")
    rospy.loginfo("Ctrl + C로 종료")

    try:
        while not rospy.is_shutdown():
            key = getKey()

            if key in commands:
                # 해당 키에 맞는 서보 이동 명령 수행
                rotate(commands[key]['duration'], commands[key]['id_pos_s'])
            elif key == '\x03':  # Ctrl+C 처리
                break

    except rospy.ROSInterruptException:
        pass
    finally:
        # 프로그램 종료 시, 터미널 설정 복구
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rospy.loginfo("Teleop Finished.")
