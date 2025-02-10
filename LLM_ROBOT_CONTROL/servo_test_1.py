#!/usr/bin/python3
# -*- coding: utf-8 -*-

import rospy
import sys, select, termios, tty
from hiwonder_servo_msgs.msg import *

def getKey():
    """
    터미널에서 non-blocking 방식으로 키 입력을 받아오는 함수.
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

def send_command(servo_id, position, duration=20):
    """
    단일 서보의 위치 명령을 보내는 함수.
      - servo_id : 서보의 ID (1~6)
      - position : 이동할 목표 위치 (0~1000)
      - duration : 동작 시간 (단위: ms; 여기서는 짧게 20ms 사용)
    """
    # RawIdPosDur(msg_id, msg_pos, msg_dur)
    msg_item = RawIdPosDur(servo_id, position, duration)
    msg = MultiRawIdPosDur(id_pos_dur_list=[msg_item])
    # 변경된 값을 보내고 duration에 해당하는 시간만큼 대기
    joints_pub.publish(msg)
    rospy.sleep(duration/1000.0)

if __name__ == '__main__':
    # 터미널 설정 백업
    settings = termios.tcgetattr(sys.stdin)

    rospy.init_node('servo_teleop_continuous', anonymous=True)
    
    # 서보 명령을 보내는 퍼블리셔 (포트와 토픽 이름은 사용 환경에 맞게 수정)
    joints_pub = rospy.Publisher('/servo_controllers/port_id_1/multi_id_pos_dur',
                                 MultiRawIdPosDur,
                                 queue_size=1)
    
    # 서보 1 ~ 6의 초기 위치를 500으로 설정 (필요시 조정)
    servo_positions = { i: 500 for i in range(1, 7) }
    
    # 각 서보의 증감키 매핑 (예시)
    # 예) 서보1: +키 'a', -키 's'
    key_mapping = {
        'a': (1, +1),   # 서보 1 위치 +1
        's': (1, -1),   # 서보 1 위치 -1
        'd': (2, +1),   # 서보 2 위치 +1
        'f': (2, -1),   # 서보 2 위치 -1
        'g': (3, +1),   # 서보 3 위치 +1
        'h': (3, -1),   # 서보 3 위치 -1
        'j': (4, +1),   # 서보 4 위치 +1
        'k': (4, -1),   # 서보 4 위치 -1
        'l': (5, +1),   # 서보 5 위치 +1
        ';': (5, -1),   # 서보 5 위치 -1
        'z': (6, +1),   # 서보 6 위치 +1
        'x': (6, -1)    # 서보 6 위치 -1
    }
    
    rospy.loginfo("=== Continuous Servo Teleop Start ===")
    rospy.loginfo("키 조작 방법:")
    rospy.loginfo("  서보 1: a (증가), s (감소)")
    rospy.loginfo("  서보 2: d (증가), f (감소)")
    rospy.loginfo("  서보 3: g (증가), h (감소)")
    rospy.loginfo("  서보 4: j (증가), k (감소)")
    rospy.loginfo("  서보 5: l (증가), ; (감소)")
    rospy.loginfo("  서보 6: z (증가), x (감소)")
    rospy.loginfo("Ctrl+C를 눌러 종료합니다.")
    
    try:
        while not rospy.is_shutdown():
            key = getKey()
            if key == '\x03':  # Ctrl+C 입력 시 종료
                break

            # key_mapping에 정의된 키가 입력된 경우
            if key in key_mapping:
                servo_id, delta = key_mapping[key]
                # 현재 서보 위치에서 delta 만큼 이동 (범위 0 ~ 1000으로 제한)
                new_pos = servo_positions[servo_id] + delta
                new_pos = max(0, min(1000, new_pos))
                servo_positions[servo_id] = new_pos

                rospy.loginfo("서보 {}: 새 위치 = {}".format(servo_id, new_pos))
                send_command(servo_id, new_pos, duration=20)
            # 입력이 없으면 아무 동작도 하지 않음

    except rospy.ROSInterruptException:
        pass
    finally:
        # 터미널 설정 복구
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        rospy.loginfo("Teleop Finished.")
