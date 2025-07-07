import cv2, time
from threading import Thread

class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self,resolution=(640,480),framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
#         ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])

        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

	# Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True



config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'
# MIN_CONF_LEVEL=0.5
MIN_CONF_LEVEL = float(input("Enter MIN_CONF_LEVEL: "))
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classes_90 = ("background", "person", "bicycle", "car", "motorcycle","airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant","unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse","sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
            "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
            "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
            "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
            "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
            "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
            "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
            "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" )

model.setInputSize(320, 320) #greater this value better the reults tune it for best output
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)


cap = VideoStream(resolution=(640,480),framerate=30).start()

while True:
    time_begin=time.perf_counter()

    img=cap.read()
    classIndex, confidence, bbox = model.detect(img, confThreshold=MIN_CONF_LEVEL) #tune confThreshold for best results

    if len(classIndex)!=0:
        for classInd, conf, boxes in zip(classIndex.flatten(), confidence.flatten(), bbox):
#             print(classes_90[classInd], round(conf,2),boxes)
            print(classes_90[classInd])

#     cv2.imshow('result', img)

    time_delta=time.perf_counter()-time_begin
    FPS=round(1/time_delta,1)
    print(f'FPS={FPS} {round(time_delta,3)}s Detections={len(classIndex)}')
    print('-----------------------------------------------------------------')
#     if cv2.waitKey(1) == ord('q'):
#         break

cap.stop()
cv2.destroyAllWindows()
