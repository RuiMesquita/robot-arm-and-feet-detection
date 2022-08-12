import cv2
import numpy as np
import depthai as dai
import torch
import functions as fnc

from model import build_unet


def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask


""" Hyperparameters """
H = 512
W = 512
size = (W, H)
checkpoint_path = "../target/model_1.3.0.pth"

torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
color = (0, 255, 0)

model = build_unet()
model = model.to(torch_device)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device))
model.eval()

# Step size for sliding window
stepSize = 0.01

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define RGB camera
rgbCam = pipeline.createColorCamera()

# Define a source - two mono (grayscale) cameras and a RGB camera
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

# Create Xlinks for devices
xoutRgb = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

# Define stream names
xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# Camera properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
rgbCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
rgbCam.setBoardSocket(dai.CameraBoardSocket.RGB)

rgbCam.setPreviewSize(512, 512)
rgbCam.preview.link(xoutRgb.input)

outputDepth = True
outputRectified = False
lrcheck = False
subpixel = False

# StereoDepth
stereo.setOutputDepth(outputDepth)
stereo.setOutputRectified(outputRectified)
stereo.setConfidenceThreshold(255)

stereo.setLeftRightCheck(lrcheck)
stereo.setSubpixel(subpixel)

monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

spatialLocationCalculator.passthroughDepth.link(xoutDepth.input)
stereo.depth.link(spatialLocationCalculator.inputDepth)

topLeft = dai.Point2f(0.28, 0.48)
bottomRight = dai.Point2f(0.32, 0.52)

spatialLocationCalculator.setWaitForConfigInput(False)
config = dai.SpatialLocationCalculatorConfigData()
config.depthThresholds.lowerThreshold = 100
config.depthThresholds.upperThreshold = 10000
config.roi = dai.Rect(topLeft, bottomRight)

spatialLocationCalculator.initialConfig.addROI(config)
spatialLocationCalculator.out.link(xoutSpatialData.input)
xinSpatialCalcConfig.out.link(spatialLocationCalculator.inputConfig)

# Pipeline is defined, now we can connect to the device
with dai.Device(pipeline) as device:
    device.startPipeline()

    # Output queue will be used to get the depth frames from the outputs defined above
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    spatialCalcQueue = device.getOutputQueue(name="spatialData", maxSize=4, blocking=False)
    spatialCalcConfigInQueue = device.getInputQueue("spatialCalcConfig")
    rgbQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)

    color = (0, 255, 0)

    while True:
        inDepth = depthQueue.get()
        inDepthAvg = spatialCalcQueue.get()
        inRgb = rgbQueue.get()
        rgbFrame = inRgb.getCvFrame()

        # get depth frame and convert to color image
        depthFrame = inDepth.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_INFERNO)

        spatialData = inDepthAvg.getSpatialLocations()

        image = cv2.resize(rgbFrame, size)
        x = np.transpose(image, (2, 0, 1))
        x = x / 255.0
        x = np.expand_dims(x, axis=0)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(torch_device)

        with torch.no_grad():
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            pred_y = pred_y[0].cpu().numpy()
            pred_y = np.squeeze(pred_y, axis=0)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        pred_y = mask_parse(pred_y)
        pred_y = pred_y * 255

        # Overlay the mask on top of the image
        image_overlayed = cv2.addWeighted(image, 1, pred_y, 1, 0)

        # Create annotations
        keypoints = fnc.get_image_keypoints(pred_y)

        for keypoint in keypoints:
            x = int(keypoint.pt[0])
            y = int(keypoint.pt[1])
            s = keypoint.size

            cv2.putText(image_overlayed, f"X: {int(x)} mm", (x - 10, y - 10), cv2.FONT_ITALIC, 0.3, color)
            cv2.putText(image_overlayed, f"Y: {int(y)} mm", (x - 10, y - 20), cv2.FONT_ITALIC, 0.3, color)

        # Display the resulting frame
        cv2.imshow("image", image_overlayed)
        cv2.imshow("depth", depthFrameColor)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break
