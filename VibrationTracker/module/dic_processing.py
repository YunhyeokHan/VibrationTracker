import json
import cv2
import os, sys
import PyQt5.QtWidgets as QtGui
from PyQt5.QtWidgets import QApplication
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.interpolate as interp
from VibrationTracker.module.target_tracking import TrackTarget
import glob
import multiprocessing



def function_star(args):
        return process_imageCurrent_star(*args)

def process_imageCurrent_star(ind_subset, pt_init, curImg, subset, dfdp, Hessian_inv, meshsize, searchSize, dicProcessing):

    curImgSearch, startPoint = dicProcessing.findSearchImg(curImg, pt_init, meshsize, searchSize)
    interpG = dicProcessing.findInterpolation(curImgSearch, "Bicubic", grad=False)
    # Current position of the target
    point_search = pt_init - startPoint

    # Initial guess of p0
    
    p_old = dicProcessing.guessInitialP0(subset, curImgSearch, point_search, meshsize)
    # Iterate to refine p
    p_result = dicProcessing.iterate_subset_tracking(p_old, subset, dfdp, Hessian_inv, interpG, point_search, meshsize)

    return p_result, ind_subset

class ProcessDIC(TrackTarget):

    def findInterpolation(self, image, interpType, grad = True):
        def findGradient(interpF, X_coord, Y_coord):
            # find the gradient of the image
            dIdx = interpF(Y_coord, X_coord, dx=0, dy=1)
            dIdy = interpF(Y_coord, X_coord, dx=1, dy=0)
            gradient = np.stack((dIdx, dIdy), axis=-1)
            return gradient
                
        X_coord = np.arange(0, image.shape[1], 1)
        Y_coord = np.arange(0, image.shape[0], 1)
        if interpType == "Bicubic":
            interpF = interp.RectBivariateSpline(Y_coord, X_coord, image, kx=3, ky=3)
        elif interpType == "Bilinear":
            interpF = interp.RectBivariateSpline(Y_coord, X_coord, image, kx=1, ky=1)
        # find the gradient of the image
        if grad == True:
            gradient = findGradient(interpF,X_coord,Y_coord)
            return interpF, gradient
        else:
            return interpF
        
        
    def findSteepestDescent(self, point, gradient, meshsize, interpF):
        # find the steepest descent images
        # gradient [dIdx, dIdy]
        # point [x, y]
        # meshsize 51
        # if all points are in the integer
        # if abs(point[0]%1)  < 1e-6 and abs(point[1]%1)  < 1e-6:
        #     X_coord = np.linspace(point[0] - (meshsize-1)//2, point[0] + (meshsize-1)//2, meshsize).astype(int)
        #     Y_coord = np.linspace(point[1] - (meshsize-1)//2, point[1] + (meshsize-1)//2, meshsize).astype(int)
        #     coord = np.meshgrid(X_coord, Y_coord)

        #     coord_x = coord[0].flatten()
        #     coord_y = coord[1].flatten()

        #     # find the steepest descent images
        #     dfdx = gradient[coord_y, coord_x, 0].reshape(meshsize, meshsize)
            # dfdy = gradient[coord_y, coord_x, 1].reshape(meshsize, meshsize)


        X_coord = np.linspace(point[0] - (meshsize-1)//2, point[0] + (meshsize-1)//2, meshsize)
        Y_coord = np.linspace(point[1] - (meshsize-1)//2, point[1] + (meshsize-1)//2, meshsize)
        coord = np.meshgrid(X_coord, Y_coord)

        coord_x = coord[0].flatten()
        coord_y = coord[1].flatten()

        # find the steepest descent images
        dfdx = interpF(coord_y, coord_x, dx=0, dy=1, grid = False).reshape(meshsize, meshsize)
        dfdy = interpF(coord_y, coord_x, dx=1, dy=0, grid = False).reshape(meshsize, meshsize)

        
        coord_centerd = np.meshgrid(X_coord - point[0], Y_coord - point[1])
        coord_centerd_x = coord_centerd[0]
        coord_centerd_y = coord_centerd[1]

        dfdu = dfdx
        dfdv = dfdy
        dfdudx = dfdx*coord_centerd_x
        dfdudy = dfdx*coord_centerd_y
        dfdvdx = dfdy*coord_centerd_x
        dfdvdy = dfdy*coord_centerd_y
        # return as a meshsize x meshsize x 6 array
        return np.stack((dfdu, dfdv, dfdudx, dfdudy, dfdvdx, dfdvdy), axis=-1)

    def findSubset(self, point, interpF, meshsize):
        # find the subset image
        halfsize = (meshsize-1)//2
        X_coord = np.arange(point[0]-halfsize, point[0]+halfsize+1, 1)
        Y_coord = np.arange(point[1]-halfsize, point[1]+halfsize+1, 1)
        coord = np.meshgrid(X_coord, Y_coord)
        coord_x = coord[0].flatten()
        coord_y = coord[1].flatten()
        
        subset = interpF(coord_y, coord_x, grid=False).reshape(meshsize, meshsize)
        return subset

    def findSubsetNorm(self, subset):
        subset_mean = np.average(subset)
        subset_norm = np.sqrt(np.sum(np.square(subset - subset_mean)))
        # find the standard deviation of the subset
        return subset_norm

    def findHessian(self, subset, dfdp):
        # find the variance of the subset
        subsetVar = self.findSubsetNorm(subset)
        # find the steepest descent images
        # dfdp [dfdu, dfdv, dfdudx, dfdudy, dfdvdx, dfdvdy]
        # H = 2/subsetVar * np.sum(dfdp * dfdp.T)
        Hessian = np.zeros((6, 6))
        for i in range(0, 6):
            for j in range(0, 6):
                Hessian[i, j] = (2/subsetVar**2)*np.sum(dfdp[:,:,i]*dfdp[:,:,j])

        return Hessian


    def guessInitialP0(self, subset, Img, point, meshsize):
        # find the NCC between the subset and the reference image
        res = cv2.matchTemplate(Img.astype(np.uint8), subset.astype(np.uint8), eval('cv2.TM_CCOEFF_NORMED'))
        # find the maximum value of the NCC
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # find Left-Top corner of the subset in the image
        x0 = point[0] - (meshsize-1)//2
        y0 = point[1] - (meshsize-1)//2
        # find the position of the maximum value
        x = max_loc[0]
        y = max_loc[1]
        # find the initial guess of the p0
        p0 = np.array([x-x0, y-y0, 0, 0, 0, 0])
        return p0

    def ptoAffine(self, p):
        affineMatrix = np.zeros((3, 3))
        affineMatrix[0, 0] = 1 + p[2].item()
        affineMatrix[0, 1] = p[4].item()
        affineMatrix[0, 2] = p[0].item()

        affineMatrix[1, 0] = p[3].item()
        affineMatrix[1, 1] = 1 + p[5].item()
        affineMatrix[1, 2] = p[1].item()

        affineMatrix[2, 2] = 1

        return affineMatrix

    def transformPoints(self, point, p, meshsize):
        # transform the points using the affine transformation  
        coord_centerd_homogeneous = self.findCoordcenterd(meshsize)
        affineMat = self.ptoAffine(p)
        coord_transformed = np.dot(affineMat, coord_centerd_homogeneous.T).T

        Xcoord_transformed = coord_transformed[:, 0] + point[0]
        Ycoord_transformed = coord_transformed[:, 1] + point[1]

        return Xcoord_transformed, Ycoord_transformed

    def findCoordcenterd(self, meshsize):
        halfsize = (meshsize-1)//2
        coord_centerd = np.meshgrid(np.arange(-halfsize, halfsize+1, 1), np.arange(-halfsize, halfsize+1, 1))
        coord_centerd = np.stack((coord_centerd[0].flatten(), coord_centerd[1].flatten()), axis=-1)
        coord_centerd_homogeneous = np.stack((coord_centerd[:, 0], coord_centerd[:, 1], np.ones(coord_centerd.shape[0])), axis=-1)
        return coord_centerd_homogeneous


    def findCurrentSubset(self, interpG, p, point, meshsize):
        
        x_coord, y_coord = self.transformPoints(point, p, meshsize)
        
        subset = interpG(y_coord, x_coord, grid=False).reshape(meshsize, meshsize)

        return subset

    def findCurrentGradient(self, subsetRef, subsetCur, dfdp):

        subsetRef_norm = self.findSubsetNorm(subsetRef)
        subsetRef_mean = np.mean(subsetRef)
        subsetCur_norm = self.findSubsetNorm(subsetCur)
        subsetCur_mean = np.mean(subsetCur)
        znssd = ((subsetRef - subsetRef_mean)/subsetRef_norm - (subsetCur - subsetCur_mean)/subsetCur_norm)
        
        # find the gradient of the error image
        gradient = np.zeros((6, 1))
        for i in range(6):
            gradient[i] = (2/subsetRef_norm)*np.sum(znssd*dfdp[:,:,i])
        return gradient, znssd

    def affineToP(self,affineMatrix):
        p = np.zeros(6)
        p[0] = affineMatrix[0, 2]
        p[1] = affineMatrix[1, 2]
        p[2] = affineMatrix[0, 0] - 1
        p[3] = affineMatrix[1, 0]
        p[4] = affineMatrix[0, 1]
        p[5] = affineMatrix[1, 1] - 1
        return p

    def findDeltaP(self,gradient, inv_hessian):

        deltaP = -np.dot(inv_hessian, gradient)
        
        return deltaP

    def updateTransformation(self,p_old, deltaP):
        
        affine_Pold = self.ptoAffine(p_old)
        affine_deltaP = self.ptoAffine(deltaP)

        # affine_Pnew = np.dot(affine_Pold, np.linalg.inv(affine_deltaP))
        affine_Pnew = affine_Pold@np.linalg.inv(affine_deltaP)

        return self.affineToP(affine_Pnew)
        # p to matrix 
        
    def cropImage(self, image, point, meshsize):
        halfsize = (meshsize-1)//2
        points_int = np.floor(point).astype(int)
        crooppedImg = image[points_int[1]-halfsize-1:points_int[1]+halfsize+2, points_int[0]-halfsize-1:points_int[0]+halfsize+2]
        # point_coord in crooppedImg
        point_coord_int = [halfsize+1, halfsize+1]
        point_coord = point - points_int + point_coord_int
        return crooppedImg, point_coord
    


    def prepareRefenceData(self, refImg, posTrack, meshsize):
        # print("interpolating the reference image")
        # interpF, gradient = self.findInterpolation(refImg, "Bicubic", grad=True)
        print("read reference data and caculate gradient / hessian")
        list_subset, list_dfdp, list_Hessian_inv = [], [], []
        for point in tqdm(posTrack):
            # crop the reference
            croppedImg, point_coord = self.cropImage(refImg, point, meshsize)
            # interpolate the cropped image
            interpF, gradient = self.findInterpolation(croppedImg, "Bicubic", grad=True)
            # find the subset, dfdp, Hessian_inv            
            subset = self.findSubset(point_coord, interpF, meshsize)

            dfdp = self.findSteepestDescent(point_coord, gradient, meshsize, interpF)
            Hessian = self.findHessian(subset, dfdp)
            list_subset.append(subset)
            list_dfdp.append(dfdp)
            list_Hessian_inv.append(np.linalg.inv(Hessian))

        return list_subset, list_dfdp, list_Hessian_inv

    def findSearchImg(self, curImg, point, meshsize, searchSize):
        
        startpoint = [point[0]-meshsize//2-searchSize, point[1]-meshsize//2-searchSize]
        endpoint = [point[0]+meshsize//2+searchSize, point[1]+meshsize//2+searchSize]

        # startpoint and end point should be inside the image 
        imagesize = curImg.shape

        if startpoint[0] < 0:
            startpoint[0] = 0
        if startpoint[1] < 0:
            startpoint[1] = 0
        if endpoint[0] > imagesize[1]:
            endpoint[0] = imagesize[1]
        if endpoint[1] > imagesize[0]:
            endpoint[1] = imagesize[0]    
        
        searchImg = curImg[startpoint[1]:startpoint[1]+2*searchSize+meshsize, startpoint[0]:startpoint[0]+2*searchSize+meshsize]
        
        return searchImg, startpoint



    def process_imageCurrent(self, curImg, posTrack, pts_init, list_subset, list_dfdp, list_Hessian_inv, meshsize, searchSize, numProcess = 4):
        """
        Tracks subsets for a single image.

        Args:
            curImg: Current image in grayscale.
            posTrack: List of positions to track.
            pts_init: Initial points of tracking.
            list_subset: List of subsets for each position.
            list_dfdp: List of derivatives for each subset.
            list_Hessian_inv: List of Hessians for each subset.
            meshsize: Size of the mesh grid.
            searchSize: Size of the search window.

        Returns:
            pts_init: Updated initial points after processing.
            TrackResults_image: Tracking results for the current image.
        """
        TrackResults_image = np.zeros((len(posTrack), 2))  # Assuming 2 coordinates to track

        input_list = []
        # create input list for multiprocessing    
        for ind_subset in range(len(posTrack)):
            pt_init = pts_init[ind_subset, :]
            # Read the subset, dfdp, Hessian, Hessian_inv
            subset = list_subset[ind_subset]
            dfdp = list_dfdp[ind_subset]
            Hessian_inv = list_Hessian_inv[ind_subset]

            input_list.append((ind_subset, pt_init, curImg, subset, dfdp, Hessian_inv,meshsize, searchSize,self))

        TrackResults_image = np.zeros((len(posTrack), 2))  # Assuming 2 coordinates to track

        # generate process poll

        pool = multiprocessing.Pool(processes=numProcess)
        chunksize = max(1, len(input_list) // (4 * numProcess))  

        results = list(tqdm(pool.imap(function_star, input_list, chunksize=chunksize), total=len(input_list)))

        pool.close()
        pool.join()
        
        index_list = []
        for i in range(len(results)):
            TrackResults_image[results[i][1],:] = results[i][0][0:2] + pts_init[i, :]
            pts_init[results[i][1], :] = np.round([TrackResults_image[i,0], TrackResults_image[i,1]]).astype(int)
            index_list.append(results[i][1])

        return pts_init, TrackResults_image

    

    def iterate_subset_tracking(self, p_old, subset, dfdp, Hessian_inv, interpG, point_search, meshsize, max_iter=100, tol=1e-9):
        """
        Iterates to update the transformation parameters until convergence for one subset.

        Args:
            p_old: Initial guess for transformation parameters.
            subset: Original subset.
            dfdp: Derivatives for subset.
            Hessian_inv: Inverse Hessian matrix.
            interpG: Interpolated gradient image.
            point_search: Current search point.
            meshsize: Size of the mesh grid.
            max_iter: Maximum number of iterations (default=100).
            tol: Tolerance for convergence (default=1e-9).

        Returns:
            Final transformation parameters after convergence (p_result).
        """
        p_result, min_norm = np.zeros(6), 1e12

        for ind_iter in range(max_iter):
            subsetCurrent = self.findCurrentSubset(interpG, p_old, point_search, meshsize)
            gradientCurrent, znssd = self.findCurrentGradient(subset, subsetCurrent, dfdp)
            p_delta = self.findDeltaP(gradientCurrent, Hessian_inv)
            p_new = self.updateTransformation(p_old, p_delta)

            p_old = p_new
            p_delta_norm = np.linalg.norm(p_delta)

            if p_delta_norm < min_norm:
                p_result = p_new
                min_norm = p_delta_norm

            if abs(p_delta_norm - min_norm) < tol:
                break

        return p_result

    def trackTarget_DICMP3D(self, imageNames1, imageNames2, posTrack, resultFolderPath, meshsize, searchSize, calibResult1, calibResult2, searchSize_twoimage, update = True, show = True, numProcess = 8):

        num_points = posTrack.shape[0]
        num_images = len(imageNames1)

        # Create some random colors
        color = np.random.randint(0, 255, (num_points, 3))
        # Take the first frame and find corners in it
        old_frame1 = cv2.imread(imageNames1[0])
        old_frame2 = cv2.imread(imageNames2[0])

        cameraMatrix1, distortionCoefficients1 = calibResult1
        cameraMatrix2, distortionCoefficients2 = calibResult2

        old_frame1 = self.undistortImage(old_frame1, cameraMatrix1, distortionCoefficients1)
        old_frame2 = self.undistortImage(old_frame2, cameraMatrix2, distortionCoefficients2)

        old_gray1 = cv2.cvtColor(old_frame1, cv2.COLOR_BGR2GRAY)

        imsize = old_gray1.shape
        max_size = max(imsize)
        mask = np.zeros_like(old_frame1)
        
        print("preparing reference data")
        list_subset1, list_dfdp1, list_Hessian_inv1 = self.prepareRefenceData(old_gray1, posTrack, meshsize)

        TrackResults1 = np.zeros((len(imageNames1), len(posTrack), 2))
        TrackResults2 = np.zeros((len(imageNames2), len(posTrack), 2))
        pts_init1 = np.round(posTrack.copy()).astype(int)
        pts_init2 = np.round(posTrack.copy()).astype(int)

        for ind_image in tqdm(range(0,num_images,1)):
            curImg1 = cv2.imread(imageNames1[ind_image], cv2.IMREAD_GRAYSCALE)
            curImg2 = cv2.imread(imageNames2[ind_image], cv2.IMREAD_GRAYSCALE)

            curImg1 = self.undistortImage(curImg1, cameraMatrix1, distortionCoefficients1)
            curImg2 = self.undistortImage(curImg2, cameraMatrix2, distortionCoefficients2)

            pts_init1, TrackResults1[ind_image] = self.process_imageCurrent(curImg1, posTrack, pts_init1, list_subset1, list_dfdp1, list_Hessian_inv1, meshsize, searchSize, numProcess)


            if ind_image == 0:
                pts_init2, TrackResults2[ind_image] = self.process_imageCurrent(curImg2, posTrack, pts_init2, list_subset1, list_dfdp1, list_Hessian_inv1, meshsize, searchSize=searchSize_twoimage, numProcess=numProcess)
                posTrack2 = TrackResults2[ind_image].reshape(posTrack.shape)
                
                list_subset2, list_dfdp2, list_Hessian_inv2 = self.prepareRefenceData(curImg2, posTrack2, meshsize)
            else:
                pts_init2, TrackResults2[ind_image] = self.process_imageCurrent(curImg2, posTrack, pts_init2, list_subset2, list_dfdp2, list_Hessian_inv2, meshsize, searchSize=searchSize, numProcess=numProcess)
            
            print("image: ", ind_image, "finised")
            if show == True:
                frame1 = cv2.cvtColor(curImg1, cv2.COLOR_GRAY2BGR)
                frame2 = cv2.cvtColor(curImg2, cv2.COLOR_GRAY2BGR)

                good_old1 = TrackResults1[ind_image]
                good_new1 = pts_init1
                print("good_old: ", good_old1.shape)
                print("good_new: ", good_new1.shape)
                good_old2 = TrackResults2[ind_image]
                good_new2 = pts_init2


                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new1, good_old1)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame1 = cv2.circle(frame1, (a, b), 5, color[i].tolist(), -1)
                for i, (new, old) in enumerate(zip(good_new2, good_old2)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    # mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame2 = cv2.circle(frame2, (a, b), 5, color[i].tolist(), -1)

                img1 = cv2.add(frame1, mask)
                img2 = cv2.add(frame2, mask)
                img = np.hstack((img1, img2))

                # cv2.imwrite("frame_%04d.png" % ind_image, img)
                img = cv2.resize(img, (1800, 800))
                cv2.imshow('frame', img)    
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break

            self.saveTrackingResults3D(TrackResults1[ind_image], TrackResults2[ind_image], resultFolderPath, ind_image)
        cv2.destroyAllWindows()






    def trackTarget_DICMP(self, imageNames, posTrack, resultFolderPath, meshsize, searchSize, calibResult = None, update = True, show = True, numProcess = 8):
            
        num_points = posTrack.shape[0]
        num_images = len(imageNames)

        # Create some random colors
        color = np.random.randint(0, 255, (num_points, 3))
        # Take the first frame and find corners in it
        old_frame = cv2.imread(imageNames[0])

        if calibResult is not None:
            cameraMatrix, distortionCoefficients = calibResult
            old_frame = self.undistortImage(old_frame, cameraMatrix, distortionCoefficients)
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        mask = np.zeros_like(old_frame)
        print("preparing reference data")
        list_subset, list_dfdp, list_Hessian_inv = self.prepareRefenceData(old_gray, posTrack, meshsize)

        # initialize the initial points
        pts_init = np.round(posTrack.copy()).astype(int)

        TrackResults = np.zeros((len(imageNames), len(pts_init), 2))
        
        for ind_image in tqdm(range(0,num_images,1)):
            # read the current image
            curImg = cv2.imread(imageNames[ind_image], cv2.IMREAD_GRAYSCALE)
            if calibResult is not None:
                curImg = self.undistortImage(curImg, cameraMatrix, distortionCoefficients)
            # iteration with the different subset
            pts_init, TrackResults[ind_image] = self.process_imageCurrent(curImg, posTrack, pts_init, list_subset, list_dfdp, list_Hessian_inv, meshsize, searchSize, numProcess)
            if show == True:
                frame = cv2.cvtColor(curImg, cv2.COLOR_GRAY2BGR)

                good_old = TrackResults[ind_image]
                good_new = pts_init
                print("good_old: ", good_old.shape)
                print("good_new: ", good_new.shape)

                # Draw the tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                    frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
                img = cv2.add(frame, mask)
                img = cv2.resize(img, (1600, 1200))
                cv2.imshow('frame', img)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            # cv2.imwrite("frame_%04d.png" % ind_image, img)
            self.saveTrackingResult(TrackResults[ind_image], resultFolderPath, ind_image)
        cv2.destroyAllWindows()
        # return TrackResults


    def openFileDialog(self):
        startingDir = './'
        dialog = QtGui.QFileDialog()
        dialog.setFileMode( QtGui.QFileDialog.FileMode() )
        
        filePath = dialog.getOpenFileName( None, 'Open working directory', startingDir )[0]
        return filePath
    
    def readImageNamesFromJson(self, jsonPath):
        with open(jsonPath) as f:
            imagesNames = json.load(f)
        return imagesNames
    
    def readCalibNameFromJson(self, jsonPath):
        
        with open(jsonPath) as f:
            calibrationResults = json.load(f)
            
        cameraMatrix = np.array(calibrationResults["cameraMatrix"])
        distortionCoefficients = np.array(calibrationResults["distortionCoefficients"])
        return cameraMatrix, distortionCoefficients

    def undistortImage(self, image, cameraMatrix, distortionCoefficients):
        h,  w = image.shape[:2]
        newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distortionCoefficients, (w,h), 1, (w,h))
        undistortedImage = cv2.undistort(image, cameraMatrix, distortionCoefficients, None, newCameraMatrix)
        return undistortedImage
    
    def readDICPreprocessResults(self, jsonPath):
        with open(jsonPath) as f:
            dicpreprocessingResults = json.load(f)
        posTrack = np.array(dicpreprocessingResults["posTrack"])
        meshSize = dicpreprocessingResults["meshSize"]
        return posTrack, meshSize
    
    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "DIC_processing" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath
    
    def saveTrackingResult(self, DIC_Results, resultFolderPath, ind_images= None):
        if ind_images is not None:
            self.outputName = os.path.join(resultFolderPath, "DIC_processing_%04d.json" % ind_images)
        else:
            self.outputName = os.path.join(resultFolderPath, 'DIC_processing.json')
        DIC_Results = {"DIC_Results": DIC_Results.tolist()}
        with open(self.outputName, 'w') as f:
            json.dump(DIC_Results, f)
        # print("Tracking results saved in: ", self.outputName)
        
    def saveTrackingResults3D(self, DIC_Results1, DIC_Results2, resultFolderPath, ind_images= None):
        if ind_images is not None:
            self.outputName = os.path.join(resultFolderPath, "DIC_processing_%04d.json" % ind_images)
        else:
            self.outputName = os.path.join(resultFolderPath, 'DIC_processing.json')
        DIC_Results = {"DIC_Results1": DIC_Results1.tolist(), "DIC_Results2": DIC_Results2.tolist()}
        with open(self.outputName, 'w') as f:
            json.dump(DIC_Results, f)
        # print("Tracking results saved in: ", self.outputName)

    def plotTrackingResult(self, TrackResults):
        ind_points = 0
        # Plot the tracking result of the first point
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(TrackResults[:, ind_points, 0],'ko')
        axes[1].plot(TrackResults[:, ind_points, 1],'ko')
        axes[0].set_xlabel('Frame')
        axes[1].set_xlabel('Frame')
        axes[0].set_ylabel('u (pixels)')
        axes[1].set_ylabel('v (pixels)')

        plt.show()

    def readTrackingResult(self, jsonPath, ind_point = None):
        print(jsonPath)
        with open(jsonPath) as f:
            DIC_Results = json.load(f)
            if ind_point is None:
                return np.array(DIC_Results["DIC_Results"])
            else:
                return np.array(DIC_Results["DIC_Results"])[ind_point,:].reshape(1,2)

    def readResultsNames(self, DIC_path):
        jsonPath_all = glob.glob(DIC_path + "/*.json")
        jsonPath_all = sorted(jsonPath_all)
        return jsonPath_all
    
    def trackTarget_DIC(self, imageNames, posTrack, resultFolderPath, winsize = 5, search = 5, calibResult = None, update = True, show = True, reinitialize = False):

        print("Tracking target using Digital Image Correlation")
        print("Parameters: ")
        print("winsize: ", winsize)
        print("update: ", update)
        print("show: ", show)
        print("Calibration: ", calibResult)
        print("Result folder: ", resultFolderPath)

        p0 = np.array(posTrack).reshape(-1,2)
        num_points = p0.shape[0]
        num_images = len(imageNames)

        # Create some random colors
        color = np.random.randint(0, 255, (num_points, 3))

        # Take the first frame and find corners in it
        old_frame = cv2.imread(imageNames[0])

        if calibResult is not None:
            cameraMatrix, distortionCoefficients = calibResult
            old_frame = self.undistortImage(old_frame, cameraMatrix, distortionCoefficients)
        
        
        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)


        height, width = old_gray.shape

        X_all = np.arange(0, width,1)
        Y_all = np.arange(0, height,1)

        f_all = interp.RectBivariateSpline(Y_all, X_all, old_gray, kx=3, ky=3)

        # set up the first frame and the first points with reference subsets

        # p_array = np.zeros((6, num_points, num_images))
        TrackResults = np.zeros((num_images, num_points, 2))
        p0_init = p0
        for ind_image in tqdm(range(0, num_images)):
            
            if calibResult is not None:
                frame = self.undistortImage(cv2.imread(imageNames[ind_image]), cameraMatrix, distortionCoefficients)
            else:
                frame = cv2.imread(imageNames[ind_image])
            
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if reinitialize == True:
                p1, points_init = self.calDIC(f_all, gray_current, p0, mesh_size=(winsize-1)//2, pts_init = p0_init, search = search)
            else:
                p1, points_init = self.calDIC(f_all, gray_current, p0, mesh_size=(winsize-1)//2, search = search)
            
            p0_init = points_init

            good_new = p1
            good_old = p0


            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)

            if show == True:
                img = cv2.resize(img, (800, 600))
                cv2.imshow('frame', img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            if update == True:
                p0 = good_new
                f_all = interp.RectBivariateSpline(Y_all, X_all, gray_current, kx=3, ky=3)
            TrackResults[ind_image] = p1.reshape(-1, 2)
            self.saveTrackingResult(TrackResults[ind_image], resultFolderPath, ind_image)

        cv2.destroyAllWindows()
    


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    processDIC = ProcessDIC()
    # TrackTarget.filePath = TrackTarget.openFileDialog()
    processDIC.filePath = "example_dic\ImportImages_2602045476576\imagesNames.json"
    imagesNames = processDIC.readImageNamesFromJson(processDIC.filePath)
    # TrackTarget.calibPath = TrackTarget.openFileDialog()
    processDIC.calibPath = ''
    if processDIC.calibPath == '':
        print("Calibration file not selected")
        calibResult = None
    else: 
        calibResult = processDIC.readCalibNameFromJson(processDIC.calibPath)
    # TrackTarget.posTrackPath = TrackTarget.openFileDialog()
    processDIC.preprocessingResultsPath = "example_dic\DIC_preprocess2956443480896\DICpreprocessResults.json"
    posTrack, meshsize = processDIC.readDICPreprocessResults(processDIC.preprocessingResultsPath)

    resultFolderPath = processDIC.createResultFolder()
    # DIC_Results = ProcessDIC.trackTarget_DIC(imagesNames, posTrack, resultFolderPath, winsize = meshsize, search = 5, calibResult = calibResult, update = False, show = True)
    # DIC_Results,ind_images = DIC_Multiprocessing(imagesNames, posTrack, resultFolderPath, meshsize = meshsize, search = 5, calibResult = calibResult, numProcess = 8, dicProcessing = processDIC)
    # print("results: ", DIC_Results)
    
    # processDIC.plotTrackingResult(DIC_Results)
    # sys.exit(app.exec_())
    
