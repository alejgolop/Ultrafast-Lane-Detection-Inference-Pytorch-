import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

#model_path = "models/tusimple_18.pth"
model_path = "../Ultra-Fast-Lane-Detection-v2/weights/ufldv2_tusimple_res34_320x800.onnx"
model_type = ModelType.TUSIMPLE
use_gpu = False

# Initialize video

videoUrl = '../../INPUT_VIDEOS/v3.mp4'

cap = cv2.VideoCapture(videoUrl)

# Initialize lane detection model
lane_detector = UltrafastLaneDetector(model_path, model_type, use_gpu)

cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)	

while cap.isOpened():
	try:
		# Read frame from the video
		ret, frame = cap.read()
	except:
		continue

	if ret:	

		# Detect the lanes
		output_img = lane_detector.detect_lanes(frame)

		cv2.imshow("Detected lanes", output_img)

	else:
		break

	# Press key q to stop
	if cv2.waitKey(1) == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()