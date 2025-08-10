import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from intrusiondetection.main import IntrusionDetectionApp

IntrusionDetectionApp().run()
