import numpy as np
import cv2 as cv
import component.chessboard_calibration as cc

video_file = 'example/chessboard.mp4'
board_pattern = (10, 7)
board_cellsize = 0.025

img_select = cc.select_img(video_file, board_pattern)
assert len(img_select) > 0, 'There is no selected images!'
rms, K, dist_coeff, rvecs, tvecs = cc.calib_chessboard(img_select, board_pattern, board_cellsize)

# Print calibration results
print('## Camera Calibration Results')
print(f'* The number of selected images = {len(img_select)}')
print(f'* RMS error = {rms}')
print(f'* Camera matrix (K) = \n{K}')
print(f'* Distortion coefficient (k1, k2, p1, p2, k3, ...) = {dist_coeff.flatten()}')

board_criteria = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK

# Open a video
video = cv.VideoCapture(video_file)
assert video.isOpened(), 'Cannot read the given input, ' + video_file

# Define 3D points for S and T
S_lower = board_cellsize * np.array([
    [[8, 5, 0]], [[5, 5, 0]], [[5, 4, 0]],
    [[7, 4, 0]], [[7, 3, 0]], [[5, 3, 0]],
    [[5, 0, 0]], [[8, 0, 0]], [[8, 1, 0]],
    [[6, 1, 0]], [[6, 2, 0]], [[8, 2, 0]],
])

S_upper = board_cellsize * np.array([
    [[8, 5, -1]], [[5, 5, -1]], [[5, 4, -1]],
    [[7, 4, -1]], [[7, 3, -1]], [[5, 3, -1]],
    [[5, 0, -1]], [[8, 0, -1]], [[8, 1, -1]],
    [[6, 1, -1]], [[6, 2, -1]], [[8, 2, -1]],
])

T_lower = board_cellsize * np.array([
    [[4, 5, 0]], [[1, 5, 0]], [[1, 4, 0]],
    [[2, 4, 0]], [[2, 0, 0]], [[3, 0, 0]],
    [[3, 4, 0]], [[4, 4, 0]]
])

T_upper = board_cellsize * np.array([
    [[4, 5, -1]], [[1, 5, -1]], [[1, 4, -1]],
    [[2, 4, -1]], [[2, 0, -1]], [[3, 0, -1]],
    [[3, 4, -1]], [[4, 4, -1]]
])

# Prepare 3D points on a chessboard 
obj_points = board_cellsize * np.array([[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])])

# Run pose estimation
while True:
    # Read an image from the video
    valid, img = video.read()
    if not valid:
        break

    # Estimate the camera pose
    success, img_points = cv.findChessboardCorners(img, board_pattern, board_criteria)
    if success:
        ret, rvec, tvec = cv.solvePnP(obj_points, img_points, K, dist_coeff)

        # Draw S on the image
        s_lower, _ = cv.projectPoints(S_lower, rvec, tvec, K, dist_coeff)
        s_upper, _ = cv.projectPoints(S_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(s_lower)], True, (0, 0, 255), 2)
        cv.polylines(img, [np.int32(s_upper)], True, (0, 0, 255), 2)
        for b, t in zip(s_lower, s_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (128, 128, 128), 2)

        t_lower, _ = cv.projectPoints(T_lower, rvec, tvec, K, dist_coeff)
        t_upper, _ = cv.projectPoints(T_upper, rvec, tvec, K, dist_coeff)
        cv.polylines(img, [np.int32(t_lower)], True, (255, 0, 0), 2)
        cv.polylines(img, [np.int32(t_upper)], True, (255, 0, 0), 2)
        for b, t in zip(t_lower, t_upper):
            cv.line(img, np.int32(b.flatten()), np.int32(t.flatten()), (128, 128, 128), 2)

        # Print the camera position
        R, _ = cv.Rodrigues(rvec) # Alternative) `scipy.spatial.transform.Rotation`
        p = (-R.T @ tvec).flatten()
        info = f'XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]'
        cv.putText(img, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0))

    # Show the image and process the key event
    cv.imshow('Pose Estimation (Chessboard)', img)
    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27: # ESC
        break

video.release()
cv.destroyAllWindows()