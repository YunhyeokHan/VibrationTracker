import ujson
import cv2
import os, sys
import numpy as np
import glob
from tqdm import tqdm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing


def function_star(args):
    return postprocess_current(*args)

def function_star_3D(args):
    return postprocess_current_3D(*args)


def postprocess_current(ind_img, jsonPath_all, reference_point, indices_within_windows, resultFolderPath, scale, homography, postprocessDIC):
    
    _ = postprocessDIC.runPostprocessing(jsonPath_all, reference_point, indices_within_windows, resultFolderPath, ind_img, scale, homography)

    return ind_img,2
def postprocess_current_3D(ind_img, jsonPath_all, indices_within_windows, resultFolderPath, projectionMatrix1, projectionMatrix2, postprocessDIC):
    
    _ = postprocessDIC.runPostprocessing_3D(jsonPath_all, indices_within_windows, resultFolderPath, ind_img, projectionMatrix1, projectionMatrix2)

    return ind_img, 2


class PostprocessDIC:

    
    def readImageNamesFromJson(self, jsonPath):
        with open(jsonPath) as f:
            imagesNames = ujson.load(f)
        return imagesNames
    
    def readCalibNameFromJson(self, jsonPath):

        with open(jsonPath) as f:
            calibrationResults = ujson.load(f)
            
        cameraMatrix = np.array(calibrationResults["cameraMatrix"])
        distortionCoefficients = np.array(calibrationResults["distortionCoefficients"])
        return cameraMatrix, distortionCoefficients

    def undistortImage(self, image, cameraMatrix, distortionCoefficients):
        h,  w = image.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, (w,h), 1, (w,h))
        undistortedImage = cv2.undistort(image, cameraMatrix, distortionCoefficients, None, newCameraMatrix)
        return undistortedImage
    
    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "DIC_postprocessing" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath
    
    def savePostProcessingResult(self, DIC_PostProcessing, resultFolderPath, ind_images= None):
        if ind_images is not None:
            self.outputName = os.path.join(resultFolderPath, "DIC_postprocessing_%04d.json" % ind_images)
        else:
            self.outputName = os.path.join(resultFolderPath, 'DIC_processing.json')
        with open(self.outputName, 'w') as f:
            ujson.dump(DIC_PostProcessing, f)
        # print("Tracking results saved in: ", self.outputName)
    
    def readTrackingResult(self, jsonPath):
        with open(jsonPath) as f:
            DIC_Results = ujson.load(f)
        return np.array(DIC_Results["DIC_Results"])
    
    def readTrackingResult3D(self, jsonPath):
        with open(jsonPath) as f:
            DIC_Results = ujson.load(f)
        DIC_Results1 = np.array(DIC_Results["DIC_Results1"])
        DIC_Results2 = np.array(DIC_Results["DIC_Results2"])
        return DIC_Results1, DIC_Results2
    
    
    def readResultsNames(self, DIC_path):
        jsonPath_all = glob.glob(DIC_path + "/*.json")
        jsonPath_all = sorted(jsonPath_all)
        return jsonPath_all

    def compute_window_points(self, reference_point, windowsize_pixel):
        num_points = reference_point.shape[0]
        
        tree = KDTree(reference_point)
        self.indices_within_windows = []
        for i in tqdm(range(num_points)):
            reference_point_i = reference_point[i, :]
            indices_within_window = tree.query_ball_point(reference_point_i, windowsize_pixel)
            self.indices_within_windows.append(indices_within_window)
        return self.indices_within_windows

    def compute_strain(self, reference_point, disp, indices_within_windows, scale=None, homography = None):
        num_points = reference_point.shape[0]
        strainField = np.zeros((num_points, 3))

        if homography is not None:
            reference_point = cv2.perspectiveTransform(np.array([reference_point]), homography)[0]

        # Iterate through each reference point
        for i in range(num_points):
            indices_within_window = indices_within_windows[i]
            window_point = reference_point[indices_within_window, :]

            displacement = disp[indices_within_windows[i], :]

            if window_point.shape[0] < 3:  # Skip if not enough points for plane fitting
                continue

            # Get the relative x and y coordinates
            x_ref_c = window_point[:, 0] - reference_point[i, 0]
            y_ref_c = window_point[:, 1] - reference_point[i, 1]
            
            if scale is not None:
                x_ref_c = x_ref_c * scale
                y_ref_c = y_ref_c * scale
            # Construct matrix A
            matrixA = np.hstack((np.ones((x_ref_c.shape[0], 1)), x_ref_c.reshape(-1, 1), y_ref_c.reshape(-1, 1)))

            # Solve the least squares problem for both displacement components (u and v)
            coefficients, _, _, _ = np.linalg.lstsq(matrixA, displacement, rcond=None)

            # Extract strain tensor components
            dudx, dudy = coefficients[1, 0], coefficients[2, 0]
            dvdx, dvdy = coefficients[1, 1], coefficients[2, 1]

            # Compute strain field using Green-Lagrange strain tensor
            strain_xx = 0.5 * (2 * dudx + dudx**2 + dvdx**2)
            strain_yy = 0.5 * (2 * dvdy + dudy**2 + dvdy**2)
            strain_xy = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)

            # Store the computed strain in the strain field array
            strainField[i, :] = np.array([strain_xx, strain_yy, strain_xy])

        return strainField
    
    def find_displacement(self, current_point, reference_point, scale = None, homography = None):
        if homography is not None:
            current_point = cv2.perspectiveTransform(np.array([current_point]), homography)[0]
            reference_point = cv2.perspectiveTransform(np.array([reference_point]), homography)[0]

        if scale is not None:
            current_point = current_point * scale
            reference_point = reference_point * scale

        return current_point - reference_point


    def initPostprocessing(self, windowsize_pixel, jsonPath_all, index_reference = 0):
        
        self.reference_point = self.readTrackingResult(jsonPath_all[index_reference])
        self.indices_within_windows = self.compute_window_points(self.reference_point, windowsize_pixel)
        return self.indices_within_windows
    def initPostprocessing_3D(self, windowsize_pixel, jsonPath_all, index_reference = 0):
        
        self.reference_point, reference_point2 = self.readTrackingResult3D(jsonPath_all[index_reference])
        self.indices_within_windows = self.compute_window_points(self.reference_point, windowsize_pixel)
        return self.indices_within_windows
    
    def runPostprocessing(self, jsonPath_all, reference_point, indices_within_windows, resultFolderPath, ind_img=0, scale=None, homography = None):
        
        current_point = self.readTrackingResult(jsonPath_all[ind_img])
        # Compute displacement field
        displacementField = self.find_displacement(current_point, reference_point, scale, homography)    
        # Compute strain field
        strainField = self.compute_strain(reference_point, displacementField, indices_within_windows, scale, homography)

        postprocessingResults = {"currentPoint": current_point.tolist(), "displacementField": displacementField.tolist(), "strainField": strainField.tolist()}
    
        self.savePostProcessingResult(postprocessingResults, resultFolderPath, ind_img)
        return postprocessingResults
    
    def triangulatePoint(self, projectionMatrix1, projectionMatrix2, point1, point2):
        point4D = cv2.triangulatePoints(projectionMatrix1, projectionMatrix2, point1.T, point2.T)
        point3D = point4D[:3] / point4D[3]
        point3D = point3D.T
        return point3D
    
    def compute_strain_3D(self, reference_point, reference_position, disp, indices_within_windows):
        num_points = reference_point.shape[0]
        strainField = np.zeros((num_points, 6))

        # Iterate through each reference point
        for i in range(num_points):
            indices_within_window = indices_within_windows[i]
            window_point = reference_point[indices_within_window, :]
            window_position = reference_position[indices_within_window, :]

            displacement = disp[indices_within_windows[i], :]

            if window_point.shape[0] < 3:  # Skip if not enough points for plane fitting
                continue

            # Get the relative x and y coordinates
            x_ref_c = window_position[:, 0] - reference_position[i, 0]
            y_ref_c = window_position[:, 1] - reference_position[i, 1]
            z_ref_c = window_position[:, 2] - reference_position[i, 2] 
            

            # Construct matrix A
            matrixA = np.hstack((
                np.ones((x_ref_c.shape[0], 1)),  # Constant term
                x_ref_c.reshape(-1, 1),          # x term
                y_ref_c.reshape(-1, 1),          # y term
                z_ref_c.reshape(-1, 1)           # z term
            ))
            # Solve the least squares problem for both displacement components (u and v)
            coefficients, _, _, _ = np.linalg.lstsq(matrixA, displacement, rcond=None)

            # Extract strain tensor components
            dudx, dudy, dudz = coefficients[1, 0], coefficients[2, 0], coefficients[3, 0]
            dvdx, dvdy, dvdz = coefficients[1, 1], coefficients[2, 1], coefficients[3, 1]
            dwdx, dwdy, dwdz = coefficients[1, 2], coefficients[2, 2], coefficients[3, 2]

            # Compute strain field using Green-Lagrange strain tensor
            strain_xx = 0.5 * (2 * dudx + dudx**2 + dvdx**2 + dwdx**2)
            strain_yy = 0.5 * (2 * dvdy + dudy**2 + dvdy**2 + dwdy**2)
            strain_zz = 0.5 * (2 * dwdz + dudz**2 + dvdz**2 + dwdz**2)
            strain_xy = 0.5 * (dudy + dvdx + dudx * dudy + dvdx * dvdy)
            strain_xz = 0.5 * (dudz + dwdx + dudx * dudz + dwdx * dwdz)
            strain_yz = 0.5 * (dvdz + dwdy + dudy * dvdz + dwdy * dwdz)

            # Store the computed strain in the strain field array
            strainField[i, :] = np.array([strain_xx, strain_yy, strain_xy, strain_zz, strain_xz, strain_yz])

        return strainField


    def runPostprocessing_3D(self, jsonPath_all, indices_within_windows, resultFolderPath, ind_img, projectionMatrix1, projectionMatrix2):

        current_point1, current_point2 = self.readTrackingResult3D(jsonPath_all[ind_img])
        reference_point1, reference_point2 = self.readTrackingResult3D(jsonPath_all[0])

        reference_position = self.triangulatePoint(projectionMatrix1, projectionMatrix2, reference_point1, reference_point2)
        current_position = self.triangulatePoint(projectionMatrix1, projectionMatrix2, current_point1, current_point2)
        
        # Compute displacement field
        displacementField = current_position - reference_position
        # Compute strain field
        strainField = self.compute_strain_3D(reference_point1, reference_position, displacementField, indices_within_windows)

        postprocessingResults = {"currentPoint": current_point1.tolist(), "displacementField": displacementField.tolist(), "strainField": strainField.tolist()}

        self.savePostProcessingResult(postprocessingResults, resultFolderPath, ind_img)
        return postprocessingResults
    

    def visualizationDICResult(self, curImg, postprocessingResults, type = "DisplacementX"):
        current_point = np.array(postprocessingResults["currentPoint"])
        strainField = np.array(postprocessingResults["strainField"])
        displacementField = np.array(postprocessingResults["displacementField"])

        if type == "DisplacementX":
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.imshow(curImg)
            disx = ax.scatter(current_point[:, 0], current_point[:, 1], c = displacementField[:, 0], cmap = cm.jet)
            ax.set_title("Displacement X")
            plt.colorbar(disx, ax = ax)
        elif type == "DisplacementY":
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.imshow(curImg)
            disy = ax.scatter(current_point[:, 0], current_point[:, 1], c = displacementField[:, 1], cmap = cm.jet)
            ax.set_title("Displacement Y")
            plt.colorbar(disy, ax = ax)
        elif type == "StrainXX":
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.imshow(curImg)
            strainxx = ax.scatter(current_point[:, 0], current_point[:, 1], c = strainField[:, 0], cmap = cm.jet)
            ax.set_title("Strain XX")
            plt.colorbar(strainxx, ax = ax)
        elif type == "StrainYY":
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.imshow(curImg)
            strainyy = ax.scatter(current_point[:, 0], current_point[:, 1], c = strainField[:, 1], cmap = cm.jet)
            ax.set_title("Strain YY")
            plt.colorbar(strainyy, ax = ax)
        elif type == "StrainXY":
            fig, ax = plt.subplots(1,1,figsize=(8,5))
            ax.imshow(curImg)
            strainxy = ax.scatter(current_point[:, 0], current_point[:, 1], c = strainField[:, 2], cmap = cm.jet)
            ax.set_title("Strain XY")
            plt.colorbar(strainxy, ax = ax)
        else:
            print("Invalid type")
            return
        
    def readHomography(self, jsonPath):
        with open(jsonPath) as f:
            homography = ujson.load(f)
        return np.array(homography["homographyMatrix"])
    
    def readProjectionMatrix(self, jsonPath):
        with open(jsonPath) as f:
            data = ujson.load(f)
        newcameraMatrix = np.array(data["newCameraMatrix"])
        rvec = np.array(data["rvec"])
        tvec = np.array(data["tvec"])

        temp = np.zeros((3, 4))
        temp[:, :3] = cv2.Rodrigues(rvec)[0]
        temp[:, 3] = tvec.reshape(3)
        projectionMatrix = np.matmul(newcameraMatrix, temp)
        return projectionMatrix

    
    def runPostProcessingAll(self, jsonPath_all, reference_point, indices_within_windows, resultFolderPath, scale = None, homography=None,numProcess=1):

        input_list = []
        for ind_img in range(len(jsonPath_all)):
            input_list.append((ind_img, jsonPath_all, reference_point, indices_within_windows, resultFolderPath, scale, homography, self))

        pool = multiprocessing.Pool(processes=numProcess)
        # chunksize = max(1, len(input_list)//numProcess)

        results = list(tqdm(pool.imap(function_star, input_list), total=len(input_list)))
        pool.close()
        pool.join()

        return
    
    def runPostProcessingAll_3D(self, jsonPath_all, reference_point, indices_within_windows, resultFolderPath, projectionMatrix1, projectionMatrix2, numProcess=1):

        input_list = []

        for ind_img in range(len(jsonPath_all)):
            input_list.append((ind_img, jsonPath_all, indices_within_windows, resultFolderPath, projectionMatrix1, projectionMatrix2, self))

        pool = multiprocessing.Pool(processes=numProcess)
        # chunksize = max(1, len(input_list)//numProcess)

        results = list(tqdm(pool.imap(function_star_3D, input_list), total=len(input_list)))
        pool.close()
        pool.join()
        return
    
    
    def readPostProcessingResult(self, jsonPath):
        with open(jsonPath) as f:
            postProcessingResults = ujson.load(f)
        currentPoint = np.array(postProcessingResults["currentPoint"])
        displacementField = np.array(postProcessingResults["displacementField"])
        strainField = np.array(postProcessingResults["strainField"])

        return currentPoint, displacementField, strainField
    
    def findConfidence(self, array_2D):
        mean = np.mean(array_2D, axis=0)
        std = np.std(array_2D, axis=0)
        min_confidence = mean - 1.96 * std
        max_confidence = mean + 1.96 * std
        return min_confidence, max_confidence

    def readTimeseries(self, point, ind_image, resultFolderPath):

        # jsonpath_postprocessing

        jsonPath_all = self.readResultsNames(resultFolderPath)
        jsonPath_all = sorted(jsonPath_all)
        jsonPath = jsonPath_all[ind_image]
        currentPoint, displacementField, strainField = self.readPostProcessingResult(jsonPath)

        # Find the index of the point closest to the input point
        index = np.argmin(np.linalg.norm(currentPoint - point, axis=1))

        # Extract the timeseries of the selected point
        displacementTimeseries = np.zeros((len(jsonPath_all), 2))
        strainTimeseries = np.zeros((len(jsonPath_all), 3))
        print("Extracting timeseries of point %d" % index)
        for i in tqdm(range(len(jsonPath_all))):
            currentPoint, displacementField, strainField = self.readPostProcessingResult(jsonPath_all[i])
            displacementTimeseries[i, :] = displacementField[index, :]
            strainTimeseries[i, :] = strainField[index, :]
        
        return displacementTimeseries, strainTimeseries

if __name__ == '__main__':


    # type 1 example when scale factor is provided
    ### ================== Example ================== ###
    # postprocessDIC = PostprocessDIC()

    # postprocessDIC.filePath = "example_dic\ImportImages_2602045476576\imagesNames.json"
    # imagesNames = postprocessDIC.readImageNamesFromJson(postprocessDIC.filePath)
    # # calibPath = "example_dic\Calibration_2602045476576\calibrationResults
    # calibPath = ''
    # if calibPath == '':
    #     print("Calibration file not selected")
    #     calibResult = None
    # else: 
    #     calibResult = postprocessDIC.readCalibNameFromJson(calibPath)
    
    # DICResultsPath = "example_dic\DIC_processing1852325745760"
    
    # jsonPath_all = postprocessDIC.readResultsNames(DICResultsPath)


    # # Example data: current_point and reference_point should be arrays of shape (N, 2)
    # reference_point = postprocessDIC.readTrackingResult(jsonPath_all[0])

    # # Initialize postprocessing
    # windowsize_pixel = 50
    # postprocessDIC.initPostprocessing(reference_point, windowsize_pixel)
    # # Compute strain field

    # ind_img = 3

    # resultFolderPath = postprocessDIC.createResultFolder()
    # postprocessingResults = postprocessDIC.runPostprocessing(jsonPath_all, reference_point, postprocessDIC.indices_within_windows, resultFolderPath, ind_img, scale = 5, homography=None)
    

    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "DisplacementX")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "DisplacementY")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "StrainXX")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "StrainYY")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "StrainXY")
    # plt.show()


    ### ================== Example ================== ###
    # type 2 example when homography is provided

    postprocessDIC = PostprocessDIC()

    postprocessDIC.filePath = "example_dicAll4wall\ImportImages_2171294723232\imagesNames.json"
    
    imagesNames = postprocessDIC.readImageNamesFromJson(postprocessDIC.filePath)
    calibPath = "example_dicAll4wall\CalibrateCamera_1323727250048\calibrationResults.json"
    calibPath = ''
    if calibPath == '':
        print("Calibration file not selected")
        calibResult = None
    else: 
        calibResult = postprocessDIC.readCalibNameFromJson(calibPath)
    
    DICResultsPath = "example_dicAll4wall\DIC_processing2146526703280"
    
    jsonPath_all = postprocessDIC.readResultsNames(DICResultsPath)

    # find homography
    poseEstimationPath = "example_dicAll4wall\EstimatePose_2064037668848\poseEstimationResults.json"

    homography = postprocessDIC.readHomography(poseEstimationPath)

    print("%d images found" % len(jsonPath_all))
    # Example data: current_point and reference_point should be arrays of shape (N, 2)
    # initialize postprocessing
    windowsize_pixel = 50
    # postprocessDIC.initPostprocessing(windowsize_pixel, jsonPath_all, index_reference = 0)

    resultFolderPath = postprocessDIC.createResultFolder()
    ind_img = 1350

    # postprocessingResults = postprocessDIC.runPostprocessing(jsonPath_all, postprocessDIC.reference_point, postprocessDIC.indices_within_windows, resultFolderPath, ind_img, scale = None, homography=homography)
    


    # postprocessDIC.runPostProcessingAll(jsonPath_all, postprocessDIC.reference_point, postprocessDIC.indices_within_windows, resultFolderPath, scale = None, homography=homography)

    # resultFile = resultFolderPath + "\DIC_postprocessing_%04d.json" % ind_img
    # postprocessingResults = postprocessDIC.readPostProcessingResult(resultFile)

    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "DisplacementX")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "DisplacementY")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "StrainXX")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "StrainYY")
    # postprocessDIC.visualizationDICResult(cv2.imread(imagesNames[ind_img]), postprocessingResults, type = "StrainXY")
    plt.show()