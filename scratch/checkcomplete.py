# ============ Base imports ======================
import os
import cv2
# ====== External package imports ================
# ====== Internal package imports ================
from src.modules.data.video_file import VideoFile
# ============== Logging  ========================
import logging
from src.modules.utils.setup import setup, IndentLogger
logger = IndentLogger(logging.getLogger(''), {})
# =========== Config File Loading ================
from src.modules.utils.config_loader import get_config
conf, confp = get_config()
# ================================================

directory_str = 'data2/'
directory = os.fsencode(directory_str)

f = open('incomplete_first10files','w')

for filename in os.listdir(directory)[:10]:
	filename = filename.decode('utf-8')
	print(filename)
	if filename[-4:] == '.mkv':
		cap = cv2.VideoCapture(directory_str + filename)
		count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		cap.set(1, count-1)
		_, frame = cap.read()
		if frame is None:
			f.write(filename+'\n')
		cap.release()

f.close()
