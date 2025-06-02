import json
import cv2
import os, sys
import PyQt5.QtWidgets as QtGui
from PyQt5.QtWidgets import QApplication
import numpy as np

class CalibrateCamera:
    """
    Class to calibrate camera using different calibration patterns
    """
    def openFileDialog(self):
        """ Open file dialog to select the working directory
        Returns:
            filePath (str): path of the working directory
        """
        startingDir = './'
        dialog = QtGui.QFileDialog()
        dialog.setFileMode( QtGui.QFileDialog.FileMode() )
        
        self.filePath = dialog.getOpenFileName( None, 'Open working directory', startingDir )[0]

        return self.filePath
    

    def readImageNamesFromJson(self, jsonPath):
        """ Read image names from json file

        Args:
            jsonPath (str): path of the json file

        Returns:
            imagesNames (list): list of image names
        """

        with open(jsonPath) as f:
            imagesNames = json.load(f)
        return imagesNames
    
############################################################################################################
## CHARUCO CALIBRATION
############################################################################################################

    def readCharucoBoard(self, imageNames, params):
        """ Read charuco board from images

        Args:
            imageNames (list): list of image names or image arrays
            params (dict): dictionary of parameters of charuco board (numberSquaresX, numberSquaresY, sizeSquare, sizeMarker, markerType, legacyPattern)

        Returns:
            allCorners (list): list of corners
            allIds (list): list of ids
            imSize (tuple): image size (height, width)
        """

        print("DETECTION OF CHARUCO CONTROL POINT:")

        numberSquareX = params['numberSquaresX']
        numberSquareY = params['numberSquaresY']
        sizeSquare = params['sizeSquare']
        sizeMarker = params['sizeMarker']
        markerType = params['markerType']
        legacyPattern = params['legacyPattern']

        allCorners = []
        allIds = []
        decimator = 0
        #sub pixel corner detection criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if markerType == "DICT_5X5_1000":
            arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        else:
            arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

        detectorParams = cv2.aruco.DetectorParameters()
        charucoParams = cv2.aruco.CharucoParameters()

        self.arucoBoard = cv2.aruco.CharucoBoard((numberSquareX, numberSquareY), sizeSquare, sizeMarker, arucoDict)
        self.arucoBoard.setLegacyPattern(legacyPattern)
        
        charucoDetector = cv2.aruco.CharucoDetector(board = self.arucoBoard, charucoParams=charucoParams, detectorParams=detectorParams)
        charucoImage = self.arucoBoard.generateImage((1000, 1000))
        
        cv2.imshow("charuco", charucoImage)
        cv2.waitKey(5)
        cv2.destroyAllWindows()

        for imageName in imageNames:
            print("=> Processing image: ", imageName)
            # if imageName is string
            if isinstance(imageName, str):
                frame = cv2.imread(imageName)
            # if imageName is numpy array
            elif isinstance(imageName, np.ndarray): 
                frame = imageName
            frame_temp = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            charucoCorners, charucoIds, markerCorners, markerIds = charucoDetector.detectBoard(gray)
            print("charucoCorners: ", charucoCorners)
            if charucoCorners is not None and charucoIds is not None:
                if len(charucoCorners) > 6:

                    # SUB PIXEL DETECTION
                    for corner in charucoCorners:
                        cv2.cornerSubPix(gray, corner, winSize=(3, 3), zeroZone=(-1, -1), criteria=criteria)
                    
                    allCorners.append(charucoCorners)
                    allIds.append(charucoIds)
                    
                    frame_temp = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds, (0, 255, 0))
                    frame_temp = cv2.aruco.drawDetectedCornersCharuco(frame, charucoCorners, charucoIds, (0, 0, 255))

                    image_resized = cv2.resize(frame_temp, (800, 600))
                    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
                    cv2.imshow('output', image_resized)
                    cv2.waitKey(50)

            imSize = gray.shape
        
        cv2.destroyAllWindows()

        return allCorners, allIds, imSize
    
    def calibrateCameraCharuco(self, allCorners, allIds, imSize):
        """ Calibrate camera using charuco board

        Args:
            allCorners (list): list of corners of charuco board
            allIds (list): list of ids of charuco board
            imSize (tuple): image size (height, width)

        Returns:
            cameraMatrix (numpy array): camera matrix
            distortionCoefficients (numpy array): distortion coefficients
            rotationVectors (list): list of rotation vectors
            translationVectors (list): list of translation vectors

        """
        print("CALIBRATION OF CAMERA:")
        imSize = (imSize[1], imSize[0])
        numImages = len(allCorners)
        numImages_objective = 60

        if numImages > numImages_objective:
            steps = int(numImages/numImages_objective)
            allCorners = allCorners[::steps]
            allIds = allIds[::steps]
            
            
        retval, cameraMatrix, distortionCoefficients, rotationVectors, translationVectors = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=allCorners,
                charucoIds=allIds,
                board=self.arucoBoard,
                imageSize=imSize, cameraMatrix=None, distCoeffs=None)
        
        print("Camera calibration finised")

        return cameraMatrix, distortionCoefficients, rotationVectors, translationVectors
    

    def runCalibrationCharuco(self, imagesNames, resultFolderPath, params):        
        """ Run calibration using charuco board and save calibration results

        Args:
            imagesNames (list): list of image names or image arrays
            resultFolderPath (str): path of the result folder
            params (dict): dictionary of parameters of charuco board (numberSquaresX, numberSquaresY, sizeSquare, sizeMarker, markerType, legacyPattern)

        Returns:
            undistortedImage (numpy array): undistorted image
        """
        allCorners, allIds, imSize = self.readCharucoBoard(imagesNames, params)
        results = self.calibrateCameraCharuco(allCorners, allIds, imSize)
        cameraMatrix, distortionCoefficients = results[0], results[1]
        img = cv2.imread(imagesNames[0])
        undistortedImage = self.undistortImage(img, cameraMatrix, distortionCoefficients)

        # save calibration results
        self.saveCalibrationResults(cameraMatrix, distortionCoefficients, resultFolderPath)

        return undistortedImage

############################################################################################################
## CHESSBOARD CALIBRATION
############################################################################################################

    def readChessboard(self, imageNames, params):
        """ Read chessboard from images 

        Args:
            imageNames (list): list of image names or image arrays
            params (dict): dictionary of parameters of chessboard (numberCornersX, numberCornersY, sizeSquare)

        Returns:
            threedpoints (list): list of 3D points
            twodpoints (list): list of 2D points
            imsize (tuple): image size (height, width)
        """
        
        
        numberCornersX = params['numberCornersX']
        numberCornersY = params['numberCornersY']
        sizeSquare = params['sizeSquare']

        ## 3D points real world coordinates
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # and convert in physical units

        grid = np.zeros((numberCornersX * numberCornersY, 3), np.float32)
        grid[:, :2] = np.mgrid[0:numberCornersX, 0:numberCornersY].T.reshape(-1, 2) * sizeSquare


        # Arrays to store object points and image points from all the images.
        threedpoints = [] # 3d point in real world space (object points)
        twodpoints = [] # 2d points in image plane (image points)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        for imageName in imageNames:       
            # read image and convert into gray scale
            if isinstance(imageName, str):
                img = cv2.imread(imageName)
            elif isinstance(imageName, np.ndarray):
                img = imageName        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
            
            
            ## Find the chess board corners   
            # cv2.findChessboardCorners(image, patternSize, flags)
            ret, corners = cv2.findChessboardCorners(gray, (numberCornersX, numberCornersY), flags=cv2.CALIB_CB_FAST_CHECK)
            
            # If found, add object points, image points (after refining them)
            if ret == False:
                # increase sharpness of the image
                sharpeningKernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                gray = cv2.filter2D(gray, -1, sharpeningKernel)
                ret, corners = cv2.findChessboardCorners(gray, (numberCornersX, numberCornersY), flags=cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_ADAPTIVE_THRESH)
                if ret == False:
                    print("Chessboard corners not found in image: ", imageName)
                    continue
            if ret == True:
                
                threedpoints.append(grid)
        
                # Refines the corner locations
                # for given 2d points
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
                twodpoints.append(corners2)
        
                # Draw and display the corners
                image = cv2.drawChessboardCorners(img, (numberCornersX, numberCornersY), corners2, ret)
               
                image_resized = cv2.resize(image, (800, 600))
                cv2.namedWindow('output', cv2.WINDOW_NORMAL)

                cv2.imshow('output', image_resized)
                # cv2.imwrite('cornerdetection.png', image)
                cv2.waitKey(50)
        
        cv2.destroyAllWindows()

        imsize = gray.shape
        return threedpoints, twodpoints, imsize
    

    def calibrateCameraChessboard(self, threedpoints, twodpoints, imsize):
        """ Calibrate camera using chessboard

        Args:
            threedpoints (list): list of 3D points
            twodpoints (list): list of 2D points
            imsize (tuple): image size (height, width)

        Returns:
            cameraMatrix (numpy array): camera matrix
            distortionCoefficients (numpy array): distortion coefficients
            rotationVectors (list): list of rotation vectors
            translationVectors (list): list of translation vectors
        """
        print("CALIBRATION OF CAMERA:")
        imsize = (imsize[1], imsize[0])

        ret, cameraMatrix, distortionCoefficients, rotationVectors, translationVectors = cv2.calibrateCamera(threedpoints, twodpoints, imsize, None, None)
        print("Camera calibration finised")
        return cameraMatrix, distortionCoefficients, rotationVectors, translationVectors
    
    def runCalibrationChessboard(self, imagesNames, resultFolderPath, params):
        """ Run calibration using chessboard and save calibration results
        
        Args:
            imagesNames (list): list of image names or image arrays
            resultFolderPath (str): path of the result folder
            params (dict): dictionary of parameters of chessboard (numberCornersX, numberCornersY, sizeSquare)

        Returns:
            undistortedImage (numpy array): undistorted image
        """

        threedpoints, twodpoints, imsize = self.readChessboard(imagesNames, params)
        results = self.calibrateCameraChessboard(threedpoints, twodpoints, imsize)
        cameraMatrix, distortionCoefficients = results[0], results[1]
        img = cv2.imread(imagesNames[0])
        undistortedImage = self.undistortImage(img, cameraMatrix, distortionCoefficients)
        # save calibration results
        self.saveCalibrationResults(cameraMatrix, distortionCoefficients, resultFolderPath)
        return undistortedImage

############################################################################################################
## CIRCLES GRID CALIBRATION
############################################################################################################

    def readCirclesGrid(self, imageNames, params):
        """ Read circles grid from images

        Args:
            imageNames (list): list of image names or image arrays
            params (dict): dictionary of parameters of circles grid (numberCirclesX, numberCirclesY, distanceBetweenCircles)

        Returns:
            threedpoints (list): list of 3D points
            twodpoints (list): list of 2D points
            imsize (tuple): image size (height, width)
        """

        numberCirclesX = params['numberCirclesX']
        numberCirclesY = params['numberCirclesY']
        distanceBetweenCircles = params['distanceBetweenCircles']

        grid = np.zeros((numberCirclesX * numberCirclesY, 3), np.float32)
        grid[:, :2] = np.mgrid[0:numberCirclesX, 0:numberCirclesY].T.reshape(-1, 2) * distanceBetweenCircles


        ## Arrays to store object points and image points from all the images.
        threedpoints = [] # 3d point in real world space (object points)
        twodpoints = [] # 2d points in image plane (image points)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # loop on images
        for imageName in imageNames:       
            # read image and convert into gray scale
            if isinstance(imageName, str):
                img = cv2.imread(imageName)
            elif isinstance(imageName, np.ndarray):
                img = imageName        
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to gray
            # histogram equalization
            # gray = cv2.equalizeHist(gray)
            # Find the circles and locate the center
            ret, corners = cv2.findCirclesGrid(gray, (numberCirclesX, numberCirclesY), None)
                        # If found, add object points, image points (after refining them)
            if ret == False:
                gray = cv2.equalizeHist(gray)
                ret, corners = cv2.findCirclesGrid(gray, (numberCirclesX, numberCirclesY), None)
            

            if ret == True:
                
                threedpoints.append(grid)
                twodpoints.append(corners)
        
                # Draw and display the corners
                image = cv2.drawChessboardCorners(img, (numberCirclesX, numberCirclesY), corners, ret)
               
                image_resized = cv2.resize(image, (800, 600))
                cv2.namedWindow('output', cv2.WINDOW_NORMAL)

                cv2.imshow('output', image_resized)
                cv2.waitKey(50) 
        

        cv2.destroyAllWindows()

        imsize = gray.shape

        return threedpoints, twodpoints, imsize
    
    def calibrateCameraCirclesGrid(self, threedpoints, twodpoints, imsize):
        """ Calibrate camera using circles grid

        Args:
            threedpoints (list): list of 3D points
            twodpoints (list): list of 2D points
            imsize (tuple): image size (height, width)

        Returns:
            cameraMatrix (numpy array): camera matrix
            distortionCoefficients (numpy array): distortion coefficients
            rotationVectors (list): list of rotation vectors
            translationVectors (list): list of translation vectors
        """
        
        print("CALIBRATION OF CAMERA:")
        cameraMatrix = np.array([[1, 0, imsize[1]/2], [0, 1, imsize[0]/2], [0, 0, 1]])
        imsize = (imsize[1], imsize[0])
        
        distortionCoefficients = np.zeros((5, 1))
        ret, cameraMatrix, distortionCoefficients, rotationVectors, translationVectors = cv2.calibrateCamera(threedpoints, twodpoints, imsize, None, None)
        
        print("Camera calibration finised")
        
        return cameraMatrix, distortionCoefficients, rotationVectors, translationVectors
    
    def runCalibrationCirclesGrid(self, imagesNames, resultFolderPath, params):
        """ Run calibration using circles grid and save calibration results

        Args:
            imagesNames (list) : list of image names or image arrays
            resultFolderPath (str): path of the result folder

        Returns:
            undistortedImage (numpy array): undistorted image
        """
        threedpoints, twodpoints, imsize = self.readCirclesGrid(imagesNames, params)
        results = self.calibrateCameraCirclesGrid(threedpoints, twodpoints, imsize)
        cameraMatrix, distortionCoefficients = results[0], results[1]
        img = cv2.imread(imagesNames[0])
        undistortedImage = self.undistortImage(img, cameraMatrix, distortionCoefficients)

        self.saveCalibrationResults(cameraMatrix, distortionCoefficients, resultFolderPath)

        return undistortedImage
        
############################################################################################################
## COMMON FUNCTIONS
############################################################################################################
    def undistortImage(self, image, cameraMatrix, distortionCoefficients, flag_newCameraMatrix = False):
        """ Undistort image using camera matrix and distortion coefficients

        Args:
            image (str or numpy array): image name or image array
            cameraMatrix (list or numpy array): camera matrix
            distortionCoefficients (list or numpy array): distortion coefficients
            flag_newCameraMatrix (bool, optional): flag to return new camera matrix. Defaults to False.

        Returns:
            undistortedImage (numpy array): undistorted image
        """
        # if imagetype is string
        if isinstance(image, str):
            image = cv2.imread(image)
        h,  w = image.shape[:2]
        # if cameraMatrix is not numpy array
        if not isinstance(cameraMatrix, np.ndarray):
            cameraMatrix = np.array(cameraMatrix)
        # if distortionCoefficients is not numpy array
        if not isinstance(distortionCoefficients, np.ndarray):
            distortionCoefficients = np.array(distortionCoefficients)
            
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, (w,h), 1, (w,h))
        undistortedImage = cv2.undistort(image, cameraMatrix, distortionCoefficients, None, newCameraMatrix)
        if flag_newCameraMatrix:
            return undistortedImage, newCameraMatrix
        else:
            return undistortedImage
    
    def saveCalibrationResults(self, cameraMatrix, distortionCoefficients, resultFolderPath):
        """ Save calibration results in json file

        Args:
            cameraMatrix (numpy array): array of camera matrix
            distortionCoefficients (_type_): _description_
            resultFolderPath (_type_): _description_
        """
        self.outputName = os.path.join(resultFolderPath, 'calibrationResults.json')
        calibrationResults = {"cameraMatrix": cameraMatrix.tolist(), "distortionCoefficients": distortionCoefficients.tolist()}
        with open(self.outputName, 'w') as f:
            json.dump(calibrationResults, f)

        print("Calibration results saved in: ", self.outputName)
    
    def createResultFolder(self, index = 0):
        """ Create result folder to save calibration results

        Args:
            index (int, optional): index of the result folder to be created folder name will be "CalibrateCamera_index". Defaults to 0.

        Returns:
            resultFolderPath (str): path of the result folder
        """
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "CalibrateCamera_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath
    
    def readCalibrationResults(self, jsonPath):
        """ Read calibration results from json file

        Args:
            jsonPath (str): path of the json file

        Returns:
            calibrationResults (dict): dictionary of calibration results
        """
        with open(jsonPath) as f:
            calibrationResults = json.load(f)
        return calibrationResults
    


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    CalibrateCamera = CalibrateCamera()
    CalibrateCamera.openFileDialog()
    imagesNames = CalibrateCamera.readImageNamesFromJson(CalibrateCamera.filePath)
    
    resultFolderPath = CalibrateCamera.createResultFolder()
    CalibrateCamera.runCalibrationCharuco(imagesNames, resultFolderPath)
    sys.exit(app.exec_())
