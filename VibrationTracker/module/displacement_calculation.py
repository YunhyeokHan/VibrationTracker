import json
import cv2
import os, sys
import PyQt5.QtWidgets as QtGui
from PyQt5.QtWidgets import QApplication
import numpy as np
import matplotlib.pyplot as plt

class CalculateDisplacement:

    def openFileDialog(self):
        startingDir = './'
        dialog = QtGui.QFileDialog()
        dialog.setFileMode( QtGui.QFileDialog.FileMode() )
        
        filePath = dialog.getOpenFileName( None, 'Open working directory', startingDir )[0]
        return filePath
    
    def readCalibNameFromJson(self, jsonPath):
        with open(jsonPath) as f:
            calibrationResults = json.load(f)        
        cameraMatrix = np.array(calibrationResults["cameraMatrix"])
        distortionCoefficients = np.array(calibrationResults["distortionCoefficients"])
        return cameraMatrix, distortionCoefficients
    
    def readHomographyMatrixFromJson(self, jsonPath):
        with open(jsonPath) as f:
            homographyResults = json.load(f)
        homographyMatrix = np.array(homographyResults["homographyMatrix"])
        return homographyMatrix
    
    def readTrackingResultsFromJson(self, jsonPath):
        with open(jsonPath) as f:
            trackingResults = json.load(f)
        trackingResults = np.array(trackingResults['TrackResults'])
        return trackingResults
    
    def readPoseEstimationResultsFromJson(self, jsonPath):
        with open(jsonPath) as f:
            poseEstimationResults = json.load(f)
        rvec = np.array(poseEstimationResults['rvec'])
        tvec = np.array(poseEstimationResults['tvec'])
        newCameraMatrix = np.array(poseEstimationResults['newCameraMatrix'])
        return rvec, tvec, newCameraMatrix

    def calculate3DPosition(self, list_TrackResultsPath, list_poseEstimationResultsPath, resultFolderPath):

        numCamera = len(list_TrackResultsPath)

        tracking_result = []
        pose_estimation_result = []
        projection_matrix = []
        for i in range(numCamera):
            tracking_result.append(self.readTrackingResultsFromJson(list_TrackResultsPath[i]))
            pose_estimation_result.append(self.readPoseEstimationResultsFromJson(list_poseEstimationResultsPath[i]))
            projection_matrix.append(self.constructProjectionMatrix(pose_estimation_result[i][0], pose_estimation_result[i][1], pose_estimation_result[i][2]))

        numImage1, numPoint1, _ = np.array(tracking_result[0]).shape
        numImage2, numPoint2, _ = np.array(tracking_result[1]).shape

        numImage = min(numImage1, numImage2)
        numPoint = min(numPoint1, numPoint2)
        print("numImage: ", numImage)
        print("numPoint: ", numPoint)


        result3DPoints = np.zeros((numImage, numPoint, 3))
    
        for ind_image in range(numImage):

            # triangulate 3D points
            point1_2D = np.array(tracking_result[0])[ind_image, :, :].T
            point2_2D = np.array(tracking_result[1])[ind_image, :, :].T
            # print("point1_2D: ", point1_2D.shape)
            # print("point2_2D: ", point2_2D.shape)
            point3D = cv2.triangulatePoints(projection_matrix[0], projection_matrix[1], point1_2D, point2_2D)
            point3D = point3D / point3D[3]
            point3D = point3D[:3].T

            if numCamera > 2:
                # TODO: implement for more than 2 cameras
                pass

            result3DPoints[ind_image, :, :] = point3D

        resultDisplacement = self.calculateDisplacementFromPosition(result3DPoints)
        Point3D = result3DPoints[0, :, :]

        self.saveDisplacementResults3D(resultFolderPath, resultDisplacement, Point3D)
        
        return resultDisplacement
    
    def calculate2DDisplacement(self, TrackResultsPath, scale_factor, resultFolderPath):

        tracking_result = self.readTrackingResultsFromJson(TrackResultsPath)
        numImage, numPoint, _ = np.array(tracking_result).shape
        result2DDisplacement = np.zeros((numImage, numPoint, 2))
        result2DDisplacement = tracking_result - tracking_result[0, :, :]
        result2DDisplacement = result2DDisplacement * scale_factor
        
        self.saveDisplacementResults2D(resultFolderPath, result2DDisplacement, tracking_result[0, :, :])

        return result2DDisplacement
    
    def calculate2DDisplacementWithHomography(self, TrackResultsPath, homographyMatrixPath, resultFolderPath):

        tracking_result = self.readTrackingResultsFromJson(TrackResultsPath)
        homographyMatrix = self.readHomographyMatrixFromJson(homographyMatrixPath)
        numImage, numPoint, _ = np.array(tracking_result).shape

        result2DPosition = np.zeros((numImage, numPoint, 2))
        
        for ind_image in range(numImage):
            point2D = np.array(tracking_result)[ind_image, :, :].T
            point2D = np.vstack((point2D, np.ones(point2D.shape[1])))
            point2D = np.dot(homographyMatrix, point2D)
            point2D = point2D / point2D[2, :]
            result2DPosition[ind_image, :, :] = point2D[:2, :].T

        result2DDisplacement = np.zeros((numImage, numPoint, 2))
        result2DDisplacement = result2DPosition - result2DPosition[0, :, :]

        self.saveDisplacementResults2D(resultFolderPath, result2DDisplacement, result2DPosition[0, :, :])

        return result2DDisplacement
    
    
    def calculateDisplacementFromPosition(self, result3DPoints):
        numImage, numPoint, _ = result3DPoints.shape
        resultDisplacement = np.zeros_like(result3DPoints)
        resultDisplacement = result3DPoints - result3DPoints[0, :, :]
        return resultDisplacement
    

    def constructProjectionMatrix(self, rvec, tvec, newCameraMatrix):
        R = cv2.Rodrigues(rvec)[0]
        P = np.dot(newCameraMatrix, np.hstack((R, tvec)))
        print("P: ", P)
        return P

    def draw3DPoints(self, result3DPoints, indexImage = 0, indexPoint = None):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([np.ptp(result3DPoints[:, :, 0]), np.ptp(result3DPoints[:, :, 1]), np.ptp(result3DPoints[:, :, 2])])
        if indexPoint is not None:
            ax.scatter(result3DPoints[indexImage, indexPoint, 0], result3DPoints[indexImage, indexPoint, 1], result3DPoints[indexImage, indexPoint, 2])
        else:
            ax.scatter(result3DPoints[indexImage, :, 0], result3DPoints[indexImage, :, 1], result3DPoints[indexImage, :, 2], c='r', marker='o', s=20)
            for indPoints in range(result3DPoints.shape[1]):
                ax.text(result3DPoints[indexImage, indPoints, 0], result3DPoints[indexImage, indPoints, 1], result3DPoints[indexImage, indPoints, 2], str(indPoints), color='black')
                
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')

        return fig, ax

    def draw3Ddisplacement(self, result3DPoints, indexPoint = 0):
        
        fig, axes = plt.subplots(3, 1, figsize=(20, 5))
        axes[0].plot(result3DPoints[:, indexPoint, 0] - result3DPoints[0, indexPoint, 0])
        axes[0].set_xlabel('Frame')
        axes[0].set_ylabel('X mm)')

        axes[1].plot(result3DPoints[:, indexPoint, 1] - result3DPoints[0, indexPoint, 1])
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Y (mm)')

        axes[2].plot(result3DPoints[:, indexPoint, 2] - result3DPoints[0, indexPoint, 2])
        axes[2].set_xlabel('Frame')
        axes[2].set_ylabel('Z (mm)')

        # plt.show()
        return fig, axes

    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "Displacement_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath
    
    def saveDisplacementResults3D(self, resultFolderPath, resultDisplacement, Point3D):
        result = {"resultDisplacement": resultDisplacement.tolist(), "Point3D": Point3D.tolist()}
        self.outputName = os.path.join(resultFolderPath, 'displacementResults.json')
        with open(self.outputName, 'w') as f:
            json.dump(result, f)
        return  
    
    def saveDisplacementResults2D(self, resultFolderPath, resultDisplacement, Point2D=np.array([])):

        result = {"resultDisplacement": resultDisplacement.tolist(), "Point2D": Point2D.tolist()}
        
        self.outputName = os.path.join(resultFolderPath, 'displacementResults.json')
        
        with open(self.outputName, 'w') as f:
            json.dump(result, f)
        
        return
    

    
    def readDisplacementResults(self, jsonPath):
        with open(jsonPath) as f:
            result = json.load(f)
        keyList = list(result.keys())

        resultDisplacement = np.array(result["resultDisplacement"])
        if "Point2D" in keyList:
            Point = np.array(result["Point2D"])
            return resultDisplacement, Point
        elif "Point3D" in keyList:
            Point = np.array(result["Point3D"])

            return resultDisplacement, Point
        else:
            return resultDisplacement, None


    


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    CD = CalculateDisplacement()
    path_trackingResult1 = "example_assemblage\TrackTarget_1940779266976\TrackResults.json"
    path_poseEstimationResult1 = "example_assemblage\EstimatePose_2483311058416\poseEstimationResults.json"
    path_trackingResult2 = "example_assemblage\TrackTarget_2821945022688\TrackResults.json"
    path_poseEstimationResult2 = "example_assemblage\EstimatePose_2821945943904\poseEstimationResults.json"

    list_TrackResultsPath = [path_trackingResult1, path_trackingResult2]
    list_poseEstimationResultsPath = [path_poseEstimationResult1, path_poseEstimationResult2]
    CD.filePath = path_trackingResult1
    resultFolderPath = CD.createResultFolder(index=0)

    result3DPoints = CD.calculate3DPosition(list_TrackResultsPath, list_poseEstimationResultsPath, resultFolderPath)
    fig, ax = CD.draw3DPoints(result3DPoints)
    fig, ax = CD.draw3Ddisplacement(result3DPoints, indexPoint=0)