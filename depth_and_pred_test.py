import cv2
import depthai as dai
import torch
import numpy as np
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
checkpoint_path = "target/unet_9p_20.pth"
stepSize = 0.01
torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = build_unet()
model = model.to(torch_device)
model.load_state_dict(torch.load(checkpoint_path, map_location=torch_device))
model.eval()

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define RGB camera
rgbCam = pipeline.createColorCamera()

# Define a source - two mono (grayscale) cameras
monoLeft = pipeline.createMonoCamera()
monoRight = pipeline.createMonoCamera()
stereo = pipeline.createStereoDepth()
spatialLocationCalculator = pipeline.createSpatialLocationCalculator()

xoutRgb = pipeline.createXLinkOut()
xoutDepth = pipeline.createXLinkOut()
xoutSpatialData = pipeline.createXLinkOut()
xinSpatialCalcConfig = pipeline.createXLinkIn()

xoutRgb.setStreamName("rgb")
xoutDepth.setStreamName("depth")
xoutSpatialData.setStreamName("spatialData")
xinSpatialCalcConfig.setStreamName("spatialCalcConfig")

# MonoCamera
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# RGB Camera
rgbCam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
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

topLeft = dai.Point2f(0.45, 0.81)
bottomRight = dai.Point2f(0.47, 0.83)

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

        # get queues for different cameras
        inDepthAvg = spatialCalcQueue.get()
        inDepth = depthQueue.get()
        inRgb = rgbQueue.get()

        # get frames for both images that we want to display
        rgbFrame = inRgb.getCvFrame()
        depthFrame = inDepth.getFrame()

        # apply a color filter to depth frame
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_INFERNO)
        depthFrameColor = cv2.resize(depthFrameColor, size)

        # get spatial location data
        spatialData = inDepthAvg.getSpatialLocations()

        for depthData in spatialData:
            
            # define roi dimensions
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            # output roi to frame and spatial coordinates
            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(depthFrameColor, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymax + 20), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymax + 35), fontType, 0.5, color)
            cv2.putText(depthFrameColor, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymax + 50), fontType, 0.5, color)

        # operations on rgb frame
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

            cv2.putText(image_overlayed, f"({x}, {y})", (x-10, y-10), cv2.FONT_HERSHEY_TRIPLEX, 0.3, color=(255, 255, 255))

        cv2.imshow("Depth image", depthFrameColor)
        cv2.imshow("Real image", image_overlayed)

        newConfig = False
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                newConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                newConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                newConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                newConfig = True
        elif key == ord('e'):
            topLeft.x += 0.01
            topLeft.y += 0.01
            bottomRight.x -= 0.01
            bottomRight.y -= 0.01
            newConfig = True
        elif key == ord('r'):
            topLeft.x -= 0.01
            topLeft.y -= 0.01
            bottomRight.x += 0.01
            bottomRight.y += 0.01
            newConfig = True
        if newConfig:
            config.roi = dai.Rect(topLeft, bottomRight)
            cfg = dai.SpatialLocationCalculatorConfig()
            cfg.addROI(config)
            spatialCalcConfigInQueue.send(cfg)
