#! python3
# coding:utf-8
import datetime
import os
import statistics
from collections import deque
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image


class KivyCamera(Image):
    def __init__(self, capture, config, **kwargs):
        super(KivyCamera, self).__init__(**kwargs)
        self.config = config
        self.capture = capture
        self.last_frames = deque(maxlen=10)
        self.last_real_frame = None
        self.alarm_triggered = False
        self.alarm_active = False
        self.fps = config.getint("camera", "fps")
        self.alarm_threshold = config.getfloat("detection", "threshold")
        self.fps_counter = 0
        path = Path(self.config.get("reactions", "image_dir"))
        path.mkdir(parents=True, exist_ok=True)
        Clock.schedule_interval(self.update, 1.0 / self.fps)
        Clock.schedule_once(
            partial(self.set_alarm, state=True),
            self.config.getfloat("alarm", "warmup"),
        )

    def shall_raise_alarm(self, frame) -> bool:
        if not self.last_frames:
            return False
        h, w = frame.shape
        mses = []
        for last_frame in self.last_frames:
            diff = cv2.subtract(last_frame, frame)
            err = np.sum(diff**2)
            mses.append(err / (float(h * w)))
        mse_avg = statistics.mean(mses)
        if self.fps_counter == 0:
            print("mse detected", mse_avg, "threshold", self.alarm_threshold)
        if not self.alarm_active:
            return False
        if mse_avg <= self.alarm_threshold:
            return False
        if not self.alarm_triggered:
            print(
                "mse detected triggering alarm",
                mse_avg,
                "threshold",
                self.alarm_threshold,
            )
        return True

    def set_alarm(self, dt=None, *, state):
        self.alarm_active = state
        print("alarm is now:", "active" if state else "deactivated")

    def set_alarm_triggered_off(self, dt=None):
        self.alarm_triggered = False

    def raise_alarm(self, dt, *, count):
        if count == 0:
            print("alarm activated")
        path = Path(self.config.get("reactions", "image_dir"))
        cv2.imwrite(
            f"{path}{os.sep}{datetime.datetime.now(tz=datetime.UTC).isoformat()}.jpg",
            self.last_real_frame,
        )
        if count < self.config.getint("alarm", "repeats"):
            Clock.schedule_once(
                partial(self.raise_alarm, count=count + 1),
                self.config.getfloat("alarm", "repeat_delay"),
            )
        else:
            Clock.schedule_once(
                self.set_alarm_triggered_off,
                self.config.getfloat("alarm", "cooldown"),
            )

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.last_real_frame = frame
            # convert it to texture
            buf = cv2.flip(frame, 0).tostring()
            image_texture = Texture.create(
                size=(frame.shape[1], frame.shape[0]), colorfmt="bgr"
            )
            image_texture.blit_buffer(buf, colorfmt="bgr", bufferfmt="ubyte")
            # display image from the texture
            self.texture = image_texture
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.blur(gray_frame, (5, 5))
            if self.shall_raise_alarm(gray_frame):
                if not self.alarm_triggered:
                    Clock.schedule_once(
                        partial(self.raise_alarm, count=0),
                        self.config.getfloat("alarm", "delay"),
                    )
                self.alarm_triggered = True
            self.last_frames.append(gray_frame)
            self.fps_counter = (self.fps_counter + 1) % self.fps


class IntrusionApp(App):
    def build_config(self, config):
        config.setdefaults("camera", {"source": "0", "fps": "30"})
        config.setdefaults("detection", {"threshold": "10"})
        config.setdefaults(
            "alarm",
            {
                "delay": "2",
                "repeat_delay": "3",
                "repeats": "1",
                "warmup": "10",
                "cooldown": "10",
            },
        )
        config.setdefaults("reactions", {"image_dir": "detections"})

    def build(self):
        config = self.config
        source = config.get("camera", "source")
        try:
            source = int(source)
        except Exception:
            pass
        self.capture = cv2.VideoCapture(source)
        self.motion_camera = KivyCamera(capture=self.capture, config=config)
        return self.motion_camera

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == "__main__":
    IntrusionApp().run()
