#! python3
# coding:utf-8
import datetime
import hmac
import os
import smtplib
import statistics
from collections import deque
from contextlib import contextmanager

# Import the email modules we'll need
from email.message import EmailMessage
from functools import partial
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
from kivy.uix.image import Image


class KivyCamera(Image):
    def __init__(self, capture, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.capture = capture
        self.last_frames = deque(maxlen=10)
        self.last_real_frame = None
        self.alarm_triggered = False
        self.alarm_active = False
        self.warmup_active = True
        self.fps = config.getint("camera", "fps")
        self.alarm_threshold = config.getfloat("detection", "threshold")
        self.fps_counter = 0
        path = Path(self.config.get("reactions", "image_dir"))
        path.mkdir(parents=True, exist_ok=True)
        email_recipients = self.config.get("reactions", "email_recipients").strip()
        email_server = self.config.get("reactions", "email_server").strip()
        if email_server and email_recipients:
            with self.create_smtp_context(timeout=4) as s:
                s.noop()
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
            Logger.debug("mse detected", mse_avg, "threshold", self.alarm_threshold)
        if not self.alarm_active:
            return False
        if mse_avg <= self.alarm_threshold:
            return False
        if not self.alarm_triggered:
            Logger.debug(
                "mse detected triggering alarm",
                mse_avg,
                "threshold",
                self.alarm_threshold,
            )
        return True

    def set_alarm(self, dt=None, *, state):
        self.alarm_active = state
        if not state:
            self.warmup_active = False
        Logger.debug("alarm is now:", "active" if state else "deactivated")

    def set_alarm_triggered_off(self, dt=None):
        self.alarm_triggered = False

    @contextmanager
    def create_smtp_context(self, timeout=60):
        email_server = self.config.get("reactions", "email_server").strip()
        email_url = urlsplit(email_server)
        if email_url.scheme == "smtp":
            smtp_class = smtplib.SMTP

        else:
            smtp_class = smtplib.SMTP_SSL

        class SMTP_fixed(smtp_class):
            def auth_cram_md5(self, challenge=None):
                if challenge is None:
                    return None
                return (
                    self.user
                    + " "
                    + hmac.HMAC(
                        self.password.encode("utf8"), challenge, "md5"
                    ).hexdigest()
                )

        try:
            with SMTP_fixed(
                email_url.hostname, port=email_url.port or 0, timeout=timeout
            ) as s:
                if email_url.scheme == "smtp":
                    s.starttls()
                if email_url.username and email_url.password:
                    s.login(
                        email_url.username,
                        email_url.password,
                    )
                yield s
        except TimeoutError as exc:
            Logger.error(
                "timeout for",
                email_url.hostname,
                email_url.port,
                "original",
                email_server,
            )
            raise exc

    def raise_alarm(self, dt, *, count):
        now = datetime.datetime.now(tz=datetime.UTC)
        path = Path(self.config.get("reactions", "image_dir"))
        full_path = f"{path}{os.sep}{now.isoformat()}.jpg"
        try:
            cv2.imwrite(
                full_path,
                self.last_real_frame,
            )
        except Exception as exc:
            Logger.exception("ERROR writing file:", exc)
        email_recipients = self.config.get("reactions", "email_recipients").strip()
        email_server = self.config.get("reactions", "email_server").strip()
        if email_recipients and email_server:
            try:
                email_url = urlsplit(email_server)
                msg = EmailMessage()
                msg["Subject"] = f"Intrusion alert: {now}"
                sender = self.config.get("reactions", "email_sender").strip()
                if not sender:
                    sender = f"{email_url.username or 'intrusiondetection'}@{email_url.hostname.removeprefix('smtp.')}"
                msg["From"] = sender
                msg["To"] = email_recipients
                msg.set_content(f"Intrusion alert triggered at {now}")
                with open(full_path, "rb") as rob:
                    msg.add_attachment(
                        rob.read(),
                        maintype="image",
                        subtype="jpg",
                        filename=f"{now.isoformat()}.jpg",
                    )

                with self.create_smtp_context() as s:
                    s.send_message(msg)
            except Exception as exc:
                Logger.exception("ERROR sending email:", exc)
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

    def draw_cv2_frame(self, frame):
        # convert it to texture
        image_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt="bgr", bufferfmt="ubyte"
        )
        image_texture.blit_buffer(frame.tostring(), colorfmt="bgr")
        image_texture.flip_vertical()
        # display image from the texture
        self.texture = image_texture

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            self.last_real_frame = frame
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.blur(gray_frame, (5, 5))
            if self.shall_raise_alarm(gray_frame):
                if not self.alarm_triggered:
                    Clock.schedule_once(
                        partial(self.raise_alarm, count=0),
                        self.config.getfloat("alarm", "delay"),
                    )
                self.alarm_triggered = True
            shown_frame = frame.copy()
            if not self.alarm_active:
                shown_frame = cv2.putText(
                    shown_frame,
                    f"Alarm: {'warming up' if self.warmup_active else 'paused'}",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (153, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif self.alarm_triggered:
                params = {
                    "text": "Alarm",
                    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                    "fontScale": 3,
                    "thickness": 3,
                }
                text_size, baseline = cv2.getTextSize(**params)
                shown_frame = cv2.putText(
                    shown_frame,
                    **params,
                    # x, y
                    org=(
                        max(0, shown_frame.shape[1] // 2 - text_size[0] // 2),
                        shown_frame.shape[0] // 2,
                    ),
                    color=(0, 0, 255),
                )
            self.draw_cv2_frame(shown_frame)
            self.last_frames.append(gray_frame)
            self.fps_counter = (self.fps_counter + 1) % self.fps


class IntrusionDetectionApp(App):
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
        config.setdefaults(
            "reactions",
            {
                "image_dir": "detections",
                "email_recipients": "",
                "email_sender": "",
                "email_server": "localhost",
            },
        )

    def build(self):
        config = self.config
        source = config.get("camera", "source")
        try:
            source = int(source)
        except Exception:
            pass
        self.capture = cv2.VideoCapture(source)
        self.motion_camera = KivyCamera(
            capture=self.capture, config=config, fit_mode="contain"
        )
        return self.motion_camera

    def on_stop(self):
        # without this, app will not exit even if the window is closed
        self.capture.release()


if __name__ == "__main__":
    IntrusionDetectionApp().run()
