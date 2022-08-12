import cv2
import depthai as dai

stepSize = 0.01
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
        inDepthAvg = spatialCalcQueue.get()
        inDepth = depthQueue.get()
        inRgb = rgbQueue.get()

        rgbFrame = inRgb.getCvFrame()

        # get frame and apply a color filter to it
        depthFrame = inDepth.getFrame()
        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_INFERNO)

        spatialData = inDepthAvg.getSpatialLocations()
        #print(spatialData)

        for depthData in spatialData:
            roi = depthData.config.roi
            roi = roi.denormalize(width=depthFrameColor.shape[1], height=depthFrameColor.shape[0])
            xmin = int(roi.topLeft().x)
            ymin = int(roi.topLeft().y)
            xmax = int(roi.bottomRight().x)
            ymax = int(roi.bottomRight().y)

            fontType = cv2.FONT_HERSHEY_TRIPLEX
            cv2.rectangle(rgbFrame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(rgbFrame, f"X: {int(depthData.spatialCoordinates.x)} mm", (xmin + 10, ymax + 20), fontType, 0.5, color)
            cv2.putText(rgbFrame, f"Y: {int(depthData.spatialCoordinates.y)} mm", (xmin + 10, ymax + 35), fontType, 0.5, color)
            cv2.putText(rgbFrame, f"Z: {int(depthData.spatialCoordinates.z)} mm", (xmin + 10, ymax + 50), fontType, 0.5, color)

        cv2.imshow("depth", depthFrameColor)
        cv2.imshow("rgb", rgbFrame)

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
