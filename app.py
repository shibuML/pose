import asyncio
import logging
import queue
import threading
import urllib.request
from pathlib import Path
from typing import List, NamedTuple

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore

import av
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydub
import streamlit as st
from aiortc.contrib.media import MediaPlayer
from mediapipe.python.solutions.hands import Hands
import numpy as np
import mediapipe as mp
import cv2
import math
import time

from streamlit_webrtc import (
    AudioProcessorBase,
    RTCConfiguration,
    VideoProcessorBase,
    WebRtcMode,
    webrtc_streamer,
)

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


def main():
    st.header("Pose Detection Demo")
    st.subheader("Right Hand Points")
    right_1, right_2, right_3 = st.columns(3)
    right_a = right_1.text_input("Enter First Right Point")
    right_b = right_2.text_input("Enter Second Right Point")
    right_c = right_3.text_input("Enter Third Right Point")
    st.subheader("Left Hand Points")
    left_1, left_2, left_3 = st.columns(3)
    left_a = left_1.text_input("Enter First Left Point")
    left_b = left_2.text_input("Enter Second Left Point")
    left_c = left_3.text_input("Enter Third Left Point")
    st.subheader("Maximum and minimum Angle")
    max_1, max_2 = st.columns(2)
    max_angle = max_1.text_input("Enter Maximum Angle")
    min_angle = max_2.text_input("Enter Minimum Angle")

    object_detection_page = "Real time Pose Detection"
    # right_a, right_b, right_c, left_a, left_b, left_c
    app_object_detection(right_a, right_b, right_c, left_a, left_b, left_c, max_angle, min_angle)

    logger.debug("=== Alive threads ===")
    for thread in threading.enumerate():
        if thread.is_alive():
            logger.debug(f"  {thread.name} ({thread.ident})")


def app_loopback():
    """ Simple video loopback """
    webrtc_streamer(key="loopback")


def find_angle(lm, img, p1, p2, p3, max_angle, min_angle, draw=True):
    p1 = tuple(lm[p1][1:])
    p2 = tuple(lm[p2][1:])
    p3 = tuple(lm[p3][1:])

    angle = math.degrees(math.atan2((p3[1] - p2[1]), (p3[0] - p2[0])) - math.atan2((p2[1] - p1[1]), (p2[0] - p1[0])))
    if angle < 0:
        angle += 360
    angle2 = angle
    if angle2 > 115:
        angle2 = 170 - (angle2 - 205)
    # angle2 = 105-(angle2-130)
    elif angle2 < 35:
        angle2 = 35
    # per = np.interp(angle2, (35, 170), (0, 100))  # 35,155
    # bar = np.interp(angle2, (35, 170), (320, 30))  # 18,105
    per = np.interp(angle2, (min_angle, max_angle), (0, 100))  # 35,155
    bar = np.interp(angle2, (min_angle, max_angle), (320, 30))  # 18,105
    # print(angle2,per)
    if draw:
        cv2.line(img, p1, p2, (255, 0, 255), 6)
        cv2.line(img, p2, p3, (255, 0, 255), 6)

        cv2.circle(img, p1, 11, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, p1, 15, (255, 255, 50), 2)

        # cv2.putText(img,"Percentage "+str(int(per)),(20,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,255),4)
        # cv2.putText(img,str(int(per)),p2,cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),4)
        # cv2.putText(img,str(int(angle)),p2,cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),4)
        cv2.circle(img, p2, 11, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, p2, 15, (255, 255, 50), 2)

        cv2.circle(img, p3, 11, (255, 255, 0), cv2.FILLED)
        cv2.circle(img, p3, 15, (255, 255, 50), 2)
    return per, bar


def app_object_detection(right_a, right_b, right_c, left_a, left_b, left_c, max_angle, min_angle):
    """Object detection demo with MobileNet SSD.
    This model and code are based on
    https://github.com/robmarkcole/object-detection-app
    """

    class PoseDetection(VideoProcessorBase):
        left_hand = bool

        def __init__(self) -> None:
            self.dir = 0
            self.c = 0
            self.per = 0
            self.bar = 330
            self.start = False
            self.ptime = 0
            self.left_hand = True

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            img = cv2.flip(image, 1)
            keypoints = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if keypoints.pose_landmarks:
                lm = []
                mp_drawing.draw_landmarks(img, keypoints.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                for idx, landmark in enumerate(keypoints.pose_landmarks.landmark):
                    lm.append([int(idx), int(landmark.x * img.shape[1]), int(landmark.y * img.shape[0])])
                # print(lm)

                if self.left_hand:
                    # self.per, self.bar = find_angle(lm, img, 11, 13, 15)
                    self.per, self.bar = find_angle(lm, img, int(left_a), int(left_b), int(left_c), max_angle,
                                                    min_angle)
                else:
                    # self.per, self.bar = find_angle(lm, img, 16, 14, 12)
                    self.per, self.bar = find_angle(lm, img, int(right_a), int(right_b), int(right_c), max_angle,
                                                    min_angle)

                if len(lm) > 0:
                    self.start = True
                if self.per >= 95.0 and self.start == True:
                    if self.dir == 1:
                        self.c += 0.5
                        self.dir = 0
                if self.per <= 25 and self.start == True:
                    if self.dir == 0:
                        self.c += 0.5
                        self.dir = 1
            cv2.putText(img, "Count " + str(int(self.c)), (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
            cv2.rectangle(img, (570, 30), (600, 330), (255, 200, 0), 3)
            color1 = (255, 200, 0)
            if self.per >= 95 and self.start == True:
                color1 = (0, 200, 0)
            elif self.per <= 25 and self.start == True:
                color1 = (0, 200, 0)
            cv2.putText(img, str(int(self.per)), (565, 23), cv2.FONT_HERSHEY_PLAIN, 2, color1, 2)
            cv2.rectangle(img, (570, int(self.bar)), (600, 330), color1, cv2.FILLED)

            ctime = time.time()
            fps = 1 / (ctime - self.ptime)
            self.ptime = ctime

            cv2.putText(img, "FPS " + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return av.VideoFrame.from_ndarray(img, format="rgb24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=PoseDetection,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    option = st.sidebar.radio('Select the angle',
                              ['Left',
                               'Right'], )
    # option = st.radio('Select the angle',
    #                   ['Left',
    #                    'Right'])
    if option == "Left":
        res = True
    else:
        res = False
    if webrtc_ctx.video_processor:
        webrtc_ctx.video_processor.left_hand = res


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
               "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
