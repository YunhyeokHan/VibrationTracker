import json
import cv2
import os, sys
import PyQt5.QtWidgets as QtGui
from PyQt5.QtWidgets import QApplication
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.interpolate

class TrackTarget:

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
    
    def readInitializeTarget(self, jsonPath):
        with open(jsonPath) as f:
            initializationResults = json.load(f)
        return initializationResults["posTrack"]

    def trackTarget_LKOF(self, imageNames, posTrack, resultFolderPath, winsize, maxLevel, calibResult = None, update = True, show = True):

        # Parameters for Lucas Kanade optical flow
        lk_params = dict(
            winSize=(winsize, winsize),
            maxLevel=maxLevel,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.001),
        )

        print("Tracking target using Lucas-Kanade Optical Flow")
        print("Parameters: ") 
        print("winsize: ", winsize)
        print("maxLevel: ", maxLevel)
        print("update: ", update)
        print("show: ", show)

        # Create some random colors
        color = np.random.randint(0, 255, (200, 3))

        # Take the first frame and find corners in it
        old_frame = cv2.imread(imageNames[0])

        if calibResult is not None:
            cameraMatrix, distortionCoefficients = calibResult
            old_frame = self.undistortImage(old_frame, cameraMatrix, distortionCoefficients)
        
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        
        p0 = np.array(posTrack, dtype=np.float32).reshape(-1, 1, 2)

        # Create a mask image for drawing purposes
        mask = np.zeros_like(old_frame)

        TrackResults = np.zeros((len(imageNames), len(p0), 2))

        for ind_image in tqdm(range(0, len(imageNames))):
            frame = cv2.imread(imageNames[ind_image])
            if calibResult is not None:
                frame = self.undistortImage(frame, cameraMatrix, distortionCoefficients)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate optical flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv2.add(frame, mask)


            # Now update the previous frame and previous points
            if update == True:
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
            if show == True:
                img = cv2.resize(img, (1600, 1200))
                cv2.imshow('frame', img)
                k = cv2.waitKey(500) & 0xff
                if k == 27:
                    break
            TrackResults[ind_image] = p1.reshape(-1, 2)

        self.saveTrackingResult(TrackResults, resultFolderPath)
        cv2.destroyAllWindows()
        return TrackResults
    

    

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

        f_all = scipy.interpolate.RectBivariateSpline(Y_all, X_all, old_gray, kx=3, ky=3)

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
                img = cv2.resize(img, (1600, 1200))
                cv2.imshow('frame', img)
                # cv2.imwrite(os.path.join(resultFolderPath, 'frame_' + str(ind_image) + '.png'), img)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break

            if update == True:
                p0 = good_new
                f_all = scipy.interpolate.RectBivariateSpline(Y_all, X_all, gray_current, kx=3, ky=3)
            TrackResults[ind_image] = p1.reshape(-1, 2)

        self.saveTrackingResult(TrackResults, resultFolderPath)
        cv2.destroyAllWindows()
        return TrackResults
    
    def calDIC(self, gray_reference, gray_current, centers, mesh_size, pts_init = None, search = 5):

        centers = np.array(centers).reshape(-1,2)
        numPoints = centers.shape[0]

        if isinstance(gray_reference, np.ndarray):

            height, width = gray_reference.shape
            X_all = np.arange(0, width,1)
            Y_all = np.arange(0, height,1)
            f_all = scipy.interpolate.RectBivariateSpline(Y_all, X_all, gray_reference, kx=3, ky=3)

        else: 
            f_all = gray_reference

        points_new = np.zeros((numPoints, 2))
        points_init = np.zeros((numPoints, 2))
        for ind_points in range(numPoints):
            
            point = centers[ind_points,:].reshape(1,-1)
            if pts_init is not None:
                pt_init = pts_init[ind_points,:].reshape(1,-1)
            
            X_roi = np.linspace(point[0,0]-mesh_size, point[0,0]+mesh_size, 2*mesh_size+1)
            Y_roi = np.linspace(point[0,1]-mesh_size, point[0,1]+mesh_size, 2*mesh_size+1)
            ref_coord = np.zeros((2*mesh_size+1, 2*mesh_size+1, 2))
            ref_coord[:,:,0], ref_coord[:,:,1] = np.meshgrid(X_roi, Y_roi)

            ref_roi = f_all(Y_roi, X_roi)
            try:
                if pts_init is None:
                    p0_res = self.find_p_DIC(ref_roi, gray_current, M=int(mesh_size), point_mid = point, search = search)
                else:
                    p0_res = self.find_p_DIC(ref_roi, gray_current, M=int(mesh_size), point_mid = point, pt_init=pt_init, search = search)
                # print("p0_res: ", p0_res)

                xy_disp = np.array(p0_res[0:2]).reshape(1,-1)

                if pts_init is not None:
                    points_new[ind_points,:] = pt_init.astype(int) + xy_disp
                    # print(point)

                else:
                    points_new[ind_points,:] = point.astype(int) + xy_disp
                points_init[ind_points,:] = points_new[ind_points,:]
            except Exception as e:
                print("error occured: ", e)
                print("error at point: ", ind_points)
                points_init[ind_points,:] =  point.astype(int)

        return points_new, points_init
    
    def find_p_DIC_core(self, ref_roi, cur_temp, M, start):
        
        f = ref_roi
        fm = np.average(f)
        f_norm = (f-fm)/np.sqrt(np.sum(np.square(f-fm)))

        # start to save the top left point
        
        # print("cur_temp: ", cur_temp.shape)
        # print("ref_roi: ", ref_roi.shape)
        # Pixel wise initial guess
        res = cv2.matchTemplate((cur_temp).astype('uint8'), (ref_roi).astype(
            'uint8'), eval('cv2.TM_CCOEFF_NORMED'))

        _,_,_, max_loc = cv2.minMaxLoc(res)
        # First order transformation function p0 = [u,v,du/dx,du/dy,dv/dx,dv/dy]
        p0_init = np.append(np.array(max_loc) - np.array(start), np.array([0, 0, 0, 0]))
        p0_init = p0_init.reshape(-1, 1)
        # if p0_init[0] != 0:
            # print("p0_init: ", p0_init)

        M_search_coarse = 3

        X_roi_search = np.linspace(-M_search_coarse, 2*M+M_search_coarse, 2*M+2*M_search_coarse+1)
        Y_roi_search = np.linspace(-M_search_coarse, 2*M+M_search_coarse, 2*M+2*M_search_coarse+1)
        
        cur_search = cur_temp[int(max_loc[1]-M_search_coarse):int(max_loc[1]+2*M+M_search_coarse+1), int(max_loc[0]-M_search_coarse):int(max_loc[0]+2*M+M_search_coarse+1)]
        
        cur_coord = np.zeros((2*M+1, 2*M+1, 2))

        cur_coord[:,:,0], cur_coord[:,:,1] = np.meshgrid(X_roi_search[M_search_coarse:-M_search_coarse], Y_roi_search[M_search_coarse:-M_search_coarse])

        g_all = scipy.interpolate.RectBivariateSpline(Y_roi_search, X_roi_search, cur_search, kx=3, ky=3)


        min_ZNSSD = 1e12

        g = np.zeros((2*M+1, 2*M+1))
        dg_dx = np.zeros((2*M+1, 2*M+1))
        dg_dy = np.zeros((2*M+1, 2*M+1))

        new_coord = np.zeros_like(cur_coord)

        p0 = np.zeros((6, 1))

        for ind_iter in range(100):


            new_coord[:, :, 0] = cur_coord[:,:,0] + p0[0] + p0[2] * (cur_coord[:, :, 0] - M) + p0[3]*(cur_coord[:, :, 1] - M)
            new_coord[:, :, 1] = cur_coord[:,:,1] + p0[1] + p0[4] * (cur_coord[:, :, 0] - M) + p0[5]*(cur_coord[:, :, 1] - M)

            new_coord_2D = new_coord.reshape(-1,2)
            # print(new_coord_2D)
            g = g_all(new_coord_2D[:,1], new_coord_2D[:,0], grid=False).reshape(2*M+1,2*M+1)
            dg_dx = g_all(new_coord_2D[:,1], new_coord_2D[:,0], dy = 1, grid=False).reshape(2*M+1,2*M+1)
            dg_dy = g_all(new_coord_2D[:,1], new_coord_2D[:,0], dx = 1, grid=False).reshape(2*M+1,2*M+1)

            gm = np.average(g)
            g_denorm = np.sqrt(np.sum(np.square(g-gm)))
            g_norm = (g-gm)/g_denorm
            SD = f_norm - g_norm

            C_ZNSSD = np.sum(SD**2)

            # Calculation of gradient of g with respect to p
            temp = np.arange(-M, M+1, 1)
            xref_xrefc = np.ones((2*M+1, 1))@np.transpose(temp[..., None])
            yref_yrefc = np.transpose(xref_xrefc)

            dg_dp = np.zeros((2*M+1, 2*M+1, 6))
            dg_dp[:, :, 0] = dg_dx
            dg_dp[:, :, 1] = dg_dy
            dg_dp[:, :, 2] = dg_dx * (xref_xrefc)
            dg_dp[:, :, 3] = dg_dx * (yref_yrefc)
            dg_dp[:, :, 4] = dg_dy * (xref_xrefc)
            dg_dp[:, :, 5] = dg_dy * (yref_yrefc)

            # Calculation of gradient of C with respect to p
            dC_dp = np.zeros((6, 1))
            for i in range(6):
                dC_dp[i] = -(2/g_denorm)*np.sum((f_norm-g_norm)*dg_dp[:, :, i])
            # Calculation of hessian matrix with respect to p
            ddC_dpdp = np.zeros((6, 6))
            for i in range(6):
                for j in range(6):
                    ddC_dpdp[i, j] = (2/(g_denorm**2)) * \
                        np.sum((dg_dp[:, :, i])*(dg_dp[:, :, j]))

            # Update the first transformation (p0)
            dp = -np.linalg.inv(ddC_dpdp)@(dC_dp)
            p0 = p0 + dp


            if abs(C_ZNSSD-min_ZNSSD) < 1e-6:
                min_ZNSSD = C_ZNSSD
                p0_result = p0
                break

            if C_ZNSSD < min_ZNSSD:
                min_ZNSSD = C_ZNSSD
                p0_result = p0

            if ind_iter == 0:
                C_ZNSSD_init = C_ZNSSD
            else:
                if C_ZNSSD > C_ZNSSD_init*10:
                    break

        p0_result = p0_result + p0_init

        return p0_result

    def find_p_DIC(self, ref_roi, cur_gray, M, point_mid, pt_init = None, search = 5):

        if pt_init is not None:
            point_mid = pt_init.reshape(1,-1)
        
        M_search = search # B1 25 #B3 5
        start = np.array((M_search, M_search))

        
        cur_temp = np.zeros((2*M+1+2*M_search, 2*M+1+2*M_search))

        y_init = int(point_mid[0, 1]-M-M_search)
        y_end = int(point_mid[0, 1]+M+M_search+1)
        x_init = int(point_mid[0, 0]-M-M_search)
        x_end = int(point_mid[0, 0]+M+M_search+1)

        if y_init < 0:
            start[1] = start[1] + y_init
            y_init = 0
        if y_end > cur_gray.shape[0]:
            y_end = cur_gray.shape[0]
        if x_init < 0:
            start[0] = start[0] + x_init 
            x_init = 0
        if x_end > cur_gray.shape[1]:
            x_end = cur_gray.shape[1]
        
        cur_temp = cur_gray[y_init:y_end, x_init:x_end]
        
        p0_result = self.find_p_DIC_core(ref_roi, cur_temp, M, start)
        
        return p0_result
    


    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "TrackTarget_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath
    
    def saveTrackingResult(self, TrackResults, resultFolderPath):
        self.outputName = os.path.join(resultFolderPath, 'TrackResults.json')
        TrackResults = {"TrackResults": TrackResults.tolist()}
        with open(self.outputName, 'w') as f:
            json.dump(TrackResults, f)
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

    def readTrackingResult(self, jsonPath):
        with open(jsonPath) as f:
            TrackResults = json.load(f)
        return np.array(TrackResults["TrackResults"])


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    TrackTarget = TrackTarget()
    # TrackTarget.filePath = TrackTarget.openFileDialog()
    TrackTarget.filePath = "example_dic\ImportImages_2602045476576\imagesNames.json"
    imagesNames = TrackTarget.readImageNamesFromJson(TrackTarget.filePath)
    # TrackTarget.calibPath = TrackTarget.openFileDialog()
    TrackTarget.calibPath = ''
    if TrackTarget.calibPath == '':
        print("Calibration file not selected")
        calibResult = None
    else: 
        calibResult = TrackTarget.readCalibNameFromJson(TrackTarget.calibPath)
    # TrackTarget.posTrackPath = TrackTarget.openFileDialog()
    TrackTarget.posTrackPath = "example_dic\InitializeTarget_2602052669248\initializationResults.json"
    posTrack = TrackTarget.readInitializeTarget(TrackTarget.posTrackPath)

    resultFolderPath = TrackTarget.createResultFolder()
    TrackResults = TrackTarget.trackTarget_DIC(imagesNames, posTrack, resultFolderPath, winsize = 51, search = 5, calibResult = calibResult, update = False, show = True)
    print("results: ", TrackResults)
    TrackTarget.plotTrackingResult(TrackResults)
    # sys.exit(app.exec_())
    
