import cv2, time

cap=cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

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

while True:
    time_begin=time.perf_counter()

    success, img=cap.read()
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

cap.release()
cv2.destroyAllWindows()

