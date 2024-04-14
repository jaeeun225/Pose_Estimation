import cv2 as cv
import numpy as np

def select_img(video_file, board_pattern, select_all=False):
    # Open a video
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    # Select images
    img_select = []
    while True:
        # Grab an images from the video
        valid, img = video.read()
        if not valid:
            break

        if select_all:
            img_select.append(img)
        else:
            # Show the image
            display = img.copy()
            cv.putText(display, f'Selected img: {len(img_select)}', (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0))
            cv.imshow('Camera Calibration', display)

            # Process the key event
            key = cv.waitKey(10)
            if key == ord(' '):             # Space: Pause and show corners
                # 'complete' is assigned a boolean value (True or False) indicating whether all corners are found.
                # 'pts' is assigned the coordinates of the corners of the chessboard.
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                complete, pts = cv.findChessboardCorners(gray, board_pattern)
                cv.drawChessboardCorners(display, board_pattern, pts, complete)
                cv.imshow('Camera Calibration', display)
                key = cv.waitKey()
                if key == ord('\r'):
                    img_select.append(img) # Enter: Select the image
            if key == 27:                  # ESC: Exit (Complete image selection)
                break

    cv.destroyAllWindows()
    return img_select

def calib_chessboard(img_select, board_pattern, board_cellsize, K=None, dist_coeff=None, calib_flags=None):
    # Find 2D corner points from given images
    # `img_points` represent the positions on the image
    img_points = []
    for img in img_select:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, pts = cv.findChessboardCorners(gray, board_pattern)
        if complete:
            img_points.append(pts)
    assert len(img_points) > 0

    # Prepare 3D points of the chess board
    # `obj_points` represent the positions in the actual 3D space
    obj_pts = [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])]
    obj_points = [np.array(obj_pts, dtype=np.float32) * board_cellsize] * len(img_points) # Must be `np.float32`

    # Calibrate the camera
    # `K`: Camera matrix
    # `dist_coeff`: Distortion coefficients
    # `calib_flags`: Calibration procedure control flags, specifying optional settings
    return cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], K, dist_coeff, flags=calib_flags)