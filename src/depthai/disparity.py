from sys import maxsize
import cv2
import depthai as dai
import numpy as np


def getFrame(queue):
    # Get frame from queue
    frame = queue.get()
    # Convert frame to OpenCV format and return
    return frame.getCvFrame()


def getMonoCamera(pipeline, isLeft):
    # Configure mono camera
    mono = pipeline.createMonoCamera()

    # Set Camera Resolution
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    if isLeft:
        # Get left camera
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        # Get right camera
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono


def getStereoPair(pipeline, mono_left, mono_right):
    stereo = pipeline.createStereoDepth()
    stereo.setLeftRightCheck(True)

    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    return stereo


if __name__ == '__main__':
    pipeline = dai.Pipeline()

    # Set up left and right cameras
    monoLeft = getMonoCamera(pipeline, isLeft=True)
    monoRight = getMonoCamera(pipeline, isLeft=False)

    stereo = getStereoPair(pipeline, monoLeft, monoRight)

    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")

    xoutDisp = pipeline.createXLinkOut()
    xoutDisp.setStreamName("disparity")

    xoutRectifiedLeft = pipeline.createXLinkOut()
    xoutRectifiedLeft.setStreamName("rectifiedLeft")

    xoutRectifiedRight = pipeline.createXLinkOut()
    xoutRectifiedRight.setStreamName("rectifiedRight")

    stereo.disparity.link(xoutDisp.input)

    stereo.rectifiedLeft.link(xoutRectifiedLeft.input)
    stereo.rectifiedRight.link(xoutRectifiedRight.input)

    with dai.Device(pipeline) as device:
        # Get output queues.
        disparityQueue = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)
        rectifiedLeftQueue = device.getOutputQueue(name="rectifiedLeft", maxSize=1, blocking=False)
        rectifiedRightQueue = device.getOutputQueue(name="rectifiedRight", maxSize=1, blocking=False)

        disparityMultiplier = 255 / stereo.initialConfig.getMaxDisparity()

        # Set display window name
        cv2.namedWindow("Stereo Pair")
        # Variable used to toggle between side by side view and one frame view.
        sideBySide = True

        while True:
            disparity = getFrame(disparityQueue)

            disparity = (disparity * disparityMultiplier).astype(np.uint8)
            disparity = cv2.applyColorMap(disparity, cv2.COLORMAP_JET)

            # Get left and right frame
            leftFrame = getFrame(rectifiedLeftQueue)
            rightFrame = getFrame(rectifiedRightQueue)

            if sideBySide:
                # Show side by side view
                imOut = np.hstack((leftFrame, rightFrame))
            else:
                # Show overlapping frames
                imOut = np.uint8(leftFrame / 2 + rightFrame / 2)

            imOut = cv2.cvtColor(imOut, cv2.COLOR_GRAY2RGB)
            # Display output image
            cv2.imshow("Stereo Pair", imOut)
            cv2.imshow("Disparity", disparity)

            # Check for keyboard input
            key = cv2.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide
            elif key == ord('c'):
                cv2.imwrite('depthai/captures/c1.png', disparity)
                print("Screenshot saved in captures/")
