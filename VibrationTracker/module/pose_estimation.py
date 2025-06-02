import cv2
import json
from VibrationTracker.module.camera_calibration import CalibrateCamera
import numpy as np
import os
from PyQt5.QtWidgets import QMessageBox
import matplotlib.pyplot as plt

class EstimatePose(CalibrateCamera):
    
    def __init__(self, node=None):

        if node is not None:
            self.node = node
            self.figure = node.mainWidget.figure
            self.ax = node.mainWidget.ax
            self.canvas = node.mainWidget.canvas
        self.cid = None
        # self.numberS 
        self.numberCornersX = 9
        self.numberCornersY = 5
        self.sizeSquare = 50
        self.numberCirclesX = 11
        self.numberCirclesY = 8
        self.distanceBetweenCircles = 15

        self.params = {"numberCornersX": self.numberCornersX, "numberCornersY": self.numberCornersY, "sizeSquare": self.sizeSquare, "numberCirclesX": self.numberCirclesX, "numberCirclesY": self.numberCirclesY, "distanceBetweenCircles": self.distanceBetweenCircles}

    def estimatePose_PointCorrespondences(self, img_path, calib_path, result_path, ind_img, points_2D, points_3D):

        imageNames = self.readImageNamesFromJson(img_path)
        img = cv2.imread(imageNames[ind_img])

        calibResult = self.readCalibrationResults(calib_path)
        cameraMatrix = calibResult["cameraMatrix"]
        distortionCoefficients = calibResult["distortionCoefficients"]

        print("Estimate Pose")

        img_undistorted, newCameraMatrix = self.undistortImage(img, cameraMatrix, distortionCoefficients, flag_newCameraMatrix=True)

        points2D = np.array(points_2D, dtype=np.float32)
        points3D = np.array(points_3D, dtype=np.float32)

        ret, rvec, tvec = cv2.solvePnP(points3D, points2D, newCameraMatrix, np.zeros((5,1)))

        img_res = self.drawResultAxis(img_undistorted, newCameraMatrix, rvec, tvec)
        homographyMatrix = np.zeros((3,3))
        self.savePoseEstimationResults(result_path, rvec, tvec, newCameraMatrix, homographyMatrix, img_res)

        return img_res

    def estimatePose(self, img_path, calib_path, result_path, params, ind_img = 2, type_pose = "Circle Grid"):
        #                      typeSelector.addItem('Charuco Pattern')
        # typeSelector.addItem('Chessboard')
        # typeSelector.addItem('Circle Grid')
        # Load the camera matrix and distortion coefficients
        if calib_path is None:
            calibResult = None
            cameraMatrix = None
            distortionCoefficients = None
        else:
            calibResult = self.readCalibrationResults(calib_path)
            cameraMatrix = calibResult["cameraMatrix"]
            distortionCoefficients = calibResult["distortionCoefficients"]
        
        print("Estimate Pose")
        
        # Load the image
        imageNames = self.readImageNamesFromJson(img_path)

        img = cv2.imread(imageNames[ind_img])

        # undistort the image
        if calib_path is None:
            img_undistorted = img
            newCameraMatrix = None
        else:
            img_undistorted, newCameraMatrix = self.undistortImage(img, cameraMatrix, distortionCoefficients, flag_newCameraMatrix=True)
        
        # Estimate the pose of the charuco board
        if type_pose == "Charuco Pattern":
            img_res, rvec, tvec, homographyMatrix = self.estimatePoseCharucoBoard(img_undistorted, newCameraMatrix, params)
        elif type_pose == "Chessboard":
            img_res, rvec, tvec, homographyMatrix = self.estimatePoseChessBoard(img_undistorted, newCameraMatrix, params)
        elif type_pose == "Circle Grid":
            img_res, rvec, tvec, homographyMatrix = self.estimatePoseCircleGrid(img_undistorted, newCameraMatrix, params)

        self.savePoseEstimationResults(result_path, rvec, tvec, newCameraMatrix, homographyMatrix, img_res)

        # cv2.imshow("Result", img_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img_res
    
    def getPerspectiveTransformCharucoBoard(self, allCorners, allIds, arucoBoard):

        points2D = allCorners
        print("points2D: ", points2D)
        print("allIds: ", allIds)
        print("arucoBoard: ", arucoBoard)
        points3D = (arucoBoard.getChessboardCorners()[np.array(allIds)]).reshape(-1, 3)
        print("points3D: ", points3D)
        points3D = np.array(points3D, dtype=np.float32).reshape(-1, 3)
        points2D = np.array(points2D, dtype=np.float32).reshape(-1, 2)
        # Estimate the Homography matrix
        H, mask = cv2.findHomography(points2D, points3D, cv2.RANSAC, 5.0)
        print("Homography matrix: ", H)

        return H

    def estimatePoseCharucoBoard(self, img, newCameraMatrix, params):

        img_temp = img.copy()
        allCorners, allIds, imSize = self.readCharucoBoard([img], params)
        newDistCoeffs = np.zeros((5,1))
        rvec = np.zeros((3,1))
        tvec = np.zeros((3,1))
        img_res = img_temp

        if newCameraMatrix is not None:
            res, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(allCorners[0], allIds[0], self.arucoBoard, newCameraMatrix, newDistCoeffs, rvec, tvec)

            img_res = self.drawResultAxis(img_temp, newCameraMatrix, rvec, tvec)

        homographyMatrix = self.getPerspectiveTransformCharucoBoard(allCorners[0],allIds[0], self.arucoBoard)
        
        # find homography matrix

        return img_res, rvec, tvec, homographyMatrix
    
    def estimatePoseChessBoard(self, img, newCameraMatrix, params):

        img_temp = img.copy()

        # Find the chessboard corners
        threedpoints, twodpoints, imsize = self.readChessboard([img], params)
        order = params["order"]
        rvec = np.zeros((3,1))
        tvec = np.zeros((3,1))
        if order == True:
            threedpoints[0] = threedpoints[0][::-1]
            # twodpoints[0] = twodpoints[0][::-1]
        # Estimate the pose of the chessboard
        img_res = img_temp
        try:
            if newCameraMatrix is not None:

                ret, rvec, tvec = cv2.solvePnP(threedpoints[0], twodpoints[0], newCameraMatrix, np.zeros((5,1))) 
                img_res = self.drawResultAxis(img_temp, newCameraMatrix, rvec, tvec)

            # find homography matrix
            H, mask = cv2.findHomography(twodpoints[0], threedpoints[0])

            return img_res, rvec, tvec, H
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Control points are not detected: {e}")

            print(f"Error message: {e}")



            

        # cv2.imshow("Result", img_res)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    
    def estimatePoseCircleGrid(self, img, newCameraMatrix, params):

        img_temp = img.copy()

        # Find the circle grid
        threedpoints, twodpoints, imsize = self.readCirclesGrid([img], params)

        order = params["order"]
        print("threedpoints: ", threedpoints)
        print("twodpoints: ", twodpoints)
        if order == True:
            threedpoints[0] = threedpoints[0][::-1]
            # twodpoints[0] = twodpoints[0][::-1]
        img_res = img_temp

        try: 
            # Estimate the pose of the circle grid
            if newCameraMatrix is not None:
                ret, rvec, tvec = cv2.solvePnP(threedpoints[0], twodpoints[0], newCameraMatrix, np.zeros((5,1)))
                img_res = self.drawResultAxis(img_temp, newCameraMatrix, rvec, tvec)

            H, mask = cv2.findHomography(twodpoints[0], threedpoints[0])

            # cv2.imshow("Result", img_res)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return img_res, rvec, tvec, H
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Control points are not detected: {e}")

            print(f"Error message: {e}")

    def resetImages(self, node):
    
        self.figure = node.mainWidget.figure
        self.ax = node.mainWidget.ax
        self.canvas = node.mainWidget.canvas

    def selectPointsInImage(self, images, index, calibResult = None):
        # if images is list of imageNames
        # if images is a path to a folder
        if type(images) == str:
            images = self.readImageNamesFromJson(images)    
        else:
            images = images
        if self.node is not None:
            self.resetImages(self.node)
        
        print("Select points in image by right-clicking")
        self.points = []
        point_index = 0

        def onclick(event, points, point_index):

            if event.button == 3:  # Right mouse button clicked
                # Save the coordinates of the clicked point
                self.points.append((event.xdata, event.ydata))
                point_index = len(self.points)-1
                # Display a red dot and the selected order on the image
                self.ax.plot(event.xdata, event.ydata, 'ro')  # Display red dot
                self.ax.text(event.xdata, event.ydata, str(point_index), color='r', fontsize=12)  # Display selected order
                self.canvas.draw()
                # Print coordinates of the selected point
                print(f"Point {point_index}: ({event.xdata}, {event.ydata})")
                print("points: ", self.points)
                self.points_array = np.array(self.points)
                self.node.configWidget.points_2D = self.points_array

        # fig = plt.figure(figsize=(20, 20))
        ref_image = cv2.imread(images[index])
        if calibResult is not None:
            ref_image = self.undistortImage(ref_image, calibResult[0], calibResult[1])
        self.ax.imshow(ref_image)
        
        self.ax.set_title("Select points by right-clicking")

        # Connect mouse click event callback
        # check if it is already connected
        if self.cid is not None:
            self.canvas.mpl_disconnect(self.cid)
        self.cid = self.canvas.mpl_connect('button_press_event', lambda event: onclick(event, self.points, point_index))
        self.canvas.draw()

        return None

    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "EstimatePose_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath

    def savePoseEstimationResults(self, resultFolderPath, rvec, tvec, newCameraMatrix, homography, imgRes):
        if newCameraMatrix is None:
            newCameraMatrix = np.zeros((3,3))
        result = {"rvec": rvec.tolist(), "tvec": tvec.tolist(), "newCameraMatrix": newCameraMatrix.tolist(), "homographyMatrix": homography.tolist()}
        
        imgResName = os.path.join(resultFolderPath, 'imgRes.png')
        cv2.imwrite(imgResName, imgRes)
        
        self.outputName = os.path.join(resultFolderPath, 'poseEstimationResults.json')

        with open(self.outputName, 'w') as f:
            json.dump(result, f)
        return
    
    def readPoseEstimationResults(self, jsonPath):
        with open(jsonPath, 'r') as f:
            result = json.load(f)
        rvec, tvec, newCameraMatrix = result["rvec"], result["tvec"], result["newCameraMatrix"]
        imgRes = cv2.imread(os.path.join(os.path.dirname(jsonPath), 'imgRes.png'))
    
        return rvec, tvec, newCameraMatrix, imgRes

    def drawResultAxis(self, img, newCameraMatrix, rvec, tvec):

        # draw X Y Z axis in the image R, G , B axis
        axisLength = 1000
        axisPoints = np.float32([[0,0,0], [axisLength,0,0], [0,axisLength,0], [0,0,axisLength]]).reshape(-1,3)
        axisPoints, _ = cv2.projectPoints(axisPoints, rvec, tvec, newCameraMatrix, np.zeros((5,1)))

        OriginPoints = axisPoints[0].ravel().astype(int)
        XaxisPoints = axisPoints[1].ravel().astype(int)
        YaxisPoints = axisPoints[2].ravel().astype(int)
        ZaxisPoints = axisPoints[3].ravel().astype(int)

        cv2.line(img, tuple(OriginPoints), tuple(XaxisPoints), (255,0,0), 5)
        cv2.line(img, tuple(OriginPoints), tuple(YaxisPoints), (0,255,0), 5)
        cv2.line(img, tuple(OriginPoints), tuple(ZaxisPoints), (0,0,255), 5)   
        
        return img
    
        

if __name__ == '__main__':
    
    EP = EstimatePose()
    img_path = "example_all4wall\ImportImages_2631899500896\imagesNames.json"
    calib_path = "example_all4wall\CalibrateCamera_2632046682752\calibrationResults.json"
    EP.filePath = img_path

    resultFolderPath = EP.createResultFolder()
    EP.estimatePose(img_path, calib_path)


    

