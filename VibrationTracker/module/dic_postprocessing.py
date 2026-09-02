import ujson
import cv2
import os
import numpy as np
import glob
from tqdm import tqdm
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing
from scipy.optimize import basinhopping

def function_star(args):
    return postprocess_current(*args)

def function_star_3D(args):
    return postprocess_current_3D(*args)

def function_star_2D_Contour(args):
    return postprocess_current_2D_Contour(*args)

def function_star_2D_NN(args):
    return postprocess_current_2D_NN(*args)

def function_star_3D_NN(args):
    return postprocess_current_3D_NN(*args)

def postprocess_current_2D_Contour(ind_img, jsonPath_all, resultFolderPath, postprocessDIC):
    _ = postprocessDIC.runPostprocessing_2D_Contour(jsonPath_all, resultFolderPath, ind_img)
    return ind_img, 2

def postprocess_current_2D_NN(ind_img,jsonPath_all,resultFolderPath,postprocessDIC):
    _ = postprocessDIC.runPostprocessing_2D_NN(jsonPath_all, resultFolderPath,ind_img)
    return ind_img, 2

def postprocess_current_3D_NN(ind_img,jsonPath_all,resultFolderPath,projectionMatrix1,projectionMatrix2,postprocessDIC):
    _ = postprocessDIC.runPostprocessing_3D_NN(jsonPath_all,resultFolderPath,ind_img,projectionMatrix1,projectionMatrix2)
    return ind_img, 2


#to calculate the rotation of the points in XY plan (3D)
def fit_plane_rotation(P_ref, dans_plan=(5, 3, 8), mark_in_plane=None):
    # P_ref: (N,3)
    if mark_in_plane is None:
        mark_in_plane = np.arange(P_ref.shape[0])

    def dist_plan(a, x, y, z):
        return np.abs(a[0]*x + a[1]*y + a[2]*z + 1) / np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)

    def sum_dist_plan(a, x, y, z):
        return np.sum(dist_plan(a, x, y, z))

    coord_plan = P_ref[np.array(dans_plan)]
    coeff_hyp = np.dot(np.linalg.inv(coord_plan), [-1, -1, -1])

    opti = basinhopping(sum_dist_plan,x0=coeff_hyp,niter=100,minimizer_kwargs={
            "args": (P_ref[mark_in_plane, 0],
                     P_ref[mark_in_plane, 1],
                     P_ref[mark_in_plane, 2]),
            "method": "powell",
            "tol": 1e-3})

    a_p, b_p, c_p = opti.x
    alpha_y = np.arccos(c_p / np.sqrt(a_p**2 + c_p**2)) - np.pi
    alpha_x = np.arccos(c_p / np.sqrt(b_p**2 + c_p**2)) - np.pi
    M_y = np.array([[ np.cos(alpha_y), 0, -np.sin(alpha_y)],
                    [ 0,               1,  0              ],
                    [ np.sin(alpha_y), 0,  np.cos(alpha_y)]])
    M_x = np.array([[1, 0,               0              ],
                    [0, np.cos(alpha_x), -np.sin(alpha_x)],
                    [0, np.sin(alpha_x),  np.cos(alpha_x)]])
    return M_x, M_y

# determination of the countour with a grid
def get_contour_Contour_center(i, j, nx, ny):
    contour = []
    left, right, top, bottom = i==0, i==nx-1, j==0, j==ny-1
    on_border = left or right or top or bottom
    is_corner = (left or right) and (top or bottom)

    if on_border:
        contour.append((i, j))

    if not on_border:
        offsets = [(0,-1),(1,-1),(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1)]
    else:
        if is_corner:
            if left and top: offsets=[(0,1),(1,1),(1,0),(0,-1)]
            elif right and top: offsets=[(0,1),(-1,1),(-1,0),(0,-1)]
            elif left and bottom: offsets=[(0,-1),(1,-1),(1,0),(0,1)]
            else: offsets=[(0,-1),(-1,-1),(-1,0),(0,1)]
        else:
            if left: offsets=[(0,-1),(1,-1),(1,0),(1,1),(0,1)]
            elif right: offsets=[(0,-1),(-1,-1),(-1,0),(-1,1),(0,1)]
            elif top: offsets=[(1,0),(1,1),(0,1),(-1,1),(-1,0)]
            else: offsets=[(1,0),(1,-1),(0,-1),(-1,-1),(-1,0)]

    for di, dj in offsets:
        ii, jj = i+di, j+dj
        if 0 <= ii < nx and 0 <= jj < ny:
            if (ii, jj) != (i, j) or not on_border:
                contour.append((ii, jj))
    return contour

def rotate_and_center(P, M_x, M_y, z0=None, mark_in_plane=None):
    Prot = (M_x @ (M_y @ P.T)).T
    if z0 is None:
        if mark_in_plane is None:
            mark_in_plane = np.arange(P.shape[0])
        z0 = np.mean(Prot[mark_in_plane, 2])
    Prot[:, 2] -= z0
    return Prot, z0

#To calculate the strain with the closer points : median pitch between points and find neighbour by angle around
def compute_reference_pitch(reference_point):
    tree = KDTree(reference_point)
    dist, idx = tree.query(reference_point,k=2)
    pitch_local = dist[:,1]
    pitch_median = np.median(pitch_local)
    return pitch_median,pitch_local

def get_contour_nearest_neighbour(point_id,reference_point,tree, pitch_median,distance_factor=1.8,min_neighbours=3,max_neighbours=8):
    center = reference_point[point_id]
    radius = pitch_median * distance_factor
    neighbours = tree.query_ball_point(center, radius)
    neighbours = [idx for idx in neighbours if idx != point_id]

    if len(neighbours) < min_neighbours:
        return []
    neighbours = np.asarray(neighbours, dtype=int)

    dx = reference_point[neighbours, 0] - center[0]
    dy = reference_point[neighbours, 1] - center[1]
    theta = np.mod(np.arctan2(dy, dx), 2*np.pi)

    # If less neighbour than max neighbour
    if len(neighbours) <= max_neighbours:
        order = np.argsort(theta)
        return neighbours[order].tolist()

    # angular repartition
    selected = []
    sector_edges = np.linspace(0,2*np.pi,max_neighbours + 1)

    for k in range(max_neighbours):
        a0 = sector_edges[k]
        a1 = sector_edges[k+1]
        mask = (theta >= a0) & (theta < a1)
        if not np.any(mask):
            continue
        candidates = neighbours[mask]
        dist = np.linalg.norm(reference_point[candidates] - center,axis=1)
        best = candidates[np.argmin(dist)]
        selected.append(best)

    # If some angle sector are empty, completition with closer neighbour
    if len(selected) < min(max_neighbours, len(neighbours)):
        remaining = [idx for idx in neighbours if idx not in selected]

        if len(remaining) > 0:
            remaining = np.asarray(remaining)
            dist = np.linalg.norm(reference_point[remaining] - center,axis=1)
            order = np.argsort(dist)

            n_missing = min(max_neighbours - len(selected),len(remaining))

            selected.extend(remaining[order[:n_missing]].tolist())

    selected = np.asarray(selected, dtype=int)

    dx = reference_point[selected, 0] - center[0]
    dy = reference_point[selected, 1] - center[1]
    theta = np.arctan2(dy, dx)
    order = np.argsort(theta)

    return selected[order].tolist()

def compute_strain_surface_Contour_2d(P_ref_2d, P_def_2d, grid_nx, grid_ny):
    """
    P_ref_2d, P_def_2d: (N,2) avec N = grid_nx*grid_ny
    retourne strainField (N,3) = [E11,E22,E12] en valeur brute (pas %)
    """
    shape = (grid_ny, grid_nx)

    X1_ref = P_ref_2d[:, 0].reshape(grid_ny, grid_nx)
    X2_ref = P_ref_2d[:, 1].reshape(grid_ny, grid_nx)

    x1_def = P_def_2d[:, 0].reshape(grid_ny, grid_nx)
    x2_def = P_def_2d[:, 1].reshape(grid_ny, grid_nx)

    F11 = np.full(shape, np.nan); F12 = np.full(shape, np.nan)
    F21 = np.full(shape, np.nan); F22 = np.full(shape, np.nan)

    for j in range(grid_ny):
        for i in range(grid_nx):

            contour = get_contour_Contour_center(i, j, grid_nx, grid_ny)
            if len(contour) < 3:
                continue

            X1 = np.array([X1_ref[jj, ii] for ii, jj in contour])
            X2 = np.array([X2_ref[jj, ii] for ii, jj in contour])
            x1 = np.array([x1_def[jj, ii] for ii, jj in contour])
            x2 = np.array([x2_def[jj, ii] for ii, jj in contour])

            D = 0.5 * np.sum(X1*np.roll(X2, -1) - X2*np.roll(X1, -1))
            if abs(D) < 1e-12:
                continue

            dX1 = np.roll(X1, -1) - np.roll(X1, 1)
            dX2 = np.roll(X2, -1) - np.roll(X2, 1)

            S11 = np.sum(x1 * dX2)
            S21 = np.sum(x2 * dX2)
            S12 = np.sum(x1 * dX1)
            S22 = np.sum(x2 * dX1)

            F11[j, i] =  S11 / (2*D)
            F21[j, i] =  S21 / (2*D)
            F12[j, i] = -S12 / (2*D)
            F22[j, i] = -S22 / (2*D)

    E11 = 0.5 * (F11**2 + F21**2 - 1)
    E22 = 0.5 * (F12**2 + F22**2 - 1)
    E12 = 0.5 * (F11*F12 + F21*F22)

    return np.stack([E11.ravel(), E22.ravel(), E12.ravel()], axis=1)

def compute_strain_surface_Contour_2d_nn(P_ref,P_def,distance_factor=1.8,min_neighbours=3,max_neighbours=8,debug_file=None):

    N = P_ref.shape[0]
    tree = KDTree(P_ref)
    pitch_median, pitch_local = compute_reference_pitch(P_ref)
    dmax = pitch_median * distance_factor

    print("========== NN DEBUG ==========")
    print(f"N points         : {N}")
    print(f"Pitch median     : {pitch_median:.3f}")
    print(f"Pitch std        : {np.std(pitch_local):.3f}")
    print(f"Distance factor  : {distance_factor:.3f}")
    print(f"Distance max     : {dmax:.3f}")
    print("==============================")

    strainField = np.full((N,3), np.nan)
    bad_points = []
    reversed_count = 0
    
    debug = {
        "pitch_median": float(pitch_median),
        "pitch_min": float(np.min(pitch_local)),
        "pitch_max": float(np.max(pitch_local)),
        "pitch_std": float(np.std(pitch_local)),
        "distance_factor": float(distance_factor),
        "distance_max": float(dmax),
        "sample_points": []
    }

    for point_id in range(N):

        contour = get_contour_nearest_neighbour(
            point_id,
            P_ref,
            tree,
            pitch_median,
            distance_factor=distance_factor,
            min_neighbours=min_neighbours,
            max_neighbours=max_neighbours
        )
        if len(contour) < min_neighbours:
            bad_points.append(point_id)
            continue

        X1 = P_ref[contour,0]
        X2 = P_ref[contour,1]

        x1 = P_def[contour,0]
        x2 = P_def[contour,1]

        D = 0.5 * np.sum(X1 * np.roll(X2,-1)-X2 * np.roll(X1,-1))

        reversed_flag = False

        if D < 0:
            contour = contour[::-1]
            X1 = P_ref[contour,0]
            X2 = P_ref[contour,1]
            x1 = P_def[contour,0]
            x2 = P_def[contour,1]
            D = -D

            reversed_flag = True
            reversed_count += 1

        area_threshold = 0.1 * pitch_median**2

        if D < area_threshold:
            bad_points.append(point_id)
            continue

        dX1 = np.roll(X1,-1) - np.roll(X1,1)
        dX2 = np.roll(X2,-1) - np.roll(X2,1)

        S11 = np.sum(x1 * dX2)
        S21 = np.sum(x2 * dX2)
        S12 = np.sum(x1 * dX1)
        S22 = np.sum(x2 * dX1)

        F11 =  S11/(2*D)
        F21 =  S21/(2*D)
        F12 = -S12/(2*D)
        F22 = -S22/(2*D)

        E11 = 0.5*(F11*F11 + F21*F21 - 1)
        E22 = 0.5*(F12*F12 + F22*F22 - 1)
        E12 = 0.5*(F11*F12 + F21*F22)

        strainField[point_id,:] = [E11,E22,E12]

        if point_id < 20:

            center = P_ref[point_id]

            dx = P_ref[contour,0] - center[0]
            dy = P_ref[contour,1] - center[1]

            theta = np.rad2deg(np.arctan2(dy,dx))

            debug["sample_points"].append({
                "point_id": int(point_id),
                "n_neighbours": int(len(contour)),
                "D": float(D),
                "reversed": bool(reversed_flag),
                "angles_deg": theta.tolist(),
                "neighbours": [int(v) for v in contour]
            })

    debug["failed_points"] = [int(v) for v in bad_points]
    debug["failed_count"] = len(bad_points)
    debug["success_count"] = N - len(bad_points)
    debug["reversed_count"] = reversed_count

    print("========== NN RESULTS ==========")
    print(f"Success  : {debug['success_count']}")
    print(f"Failed   : {debug['failed_count']}")
    print(f"Reversed : {reversed_count}")
    print("================================")

    if debug_file is not None:

        csv_file = debug_file.replace(".json",".csv")

        with open(csv_file,"w") as f:

            f.write("point_id,E11,E22,E12\n")

            for i in range(N):

                f.write(
                    f"{i},"
                    f"{strainField[i,0]},"
                    f"{strainField[i,1]},"
                    f"{strainField[i,2]}\n"
                )

        with open(debug_file,"w") as f:
            ujson.dump(debug,f,indent=4)

    return strainField
        


def compute_strain_surface_Contour_3d_nn(P_ref_rot,P_def_rot,distance_factor=1.8,min_neighbours=3,max_neighbours=8,debug_file=None):
    
        N = P_ref_rot.shape[0]
        # Neighbour calculated only on X/Y
        P_ref_xy = P_ref_rot[:, :2]
    
        tree = KDTree(P_ref_xy)
    
        pitch_median, pitch_local = compute_reference_pitch(
            P_ref_xy
        )
    
        strainField = np.full((N, 3), np.nan)
    
        bad_points = []
        reversed_count = 0
    
        for point_id in range(N):
    
            contour = get_contour_nearest_neighbour(point_id,P_ref_xy,tree,pitch_median,distance_factor=distance_factor,min_neighbours=min_neighbours,max_neighbours=max_neighbours)
    
            if len(contour) < min_neighbours:
                bad_points.append(point_id)
                continue
    
            X1 = P_ref_rot[contour, 0]
            X2 = P_ref_rot[contour, 1]
    
            x1 = P_def_rot[contour, 0]
            x2 = P_def_rot[contour, 1]
            x3 = P_def_rot[contour, 2]
    
            D = 0.5 * np.sum(
                X1 * np.roll(X2, -1)
                -
                X2 * np.roll(X1, -1)
            )

            if D < 0:
    
                contour = contour[::-1]
    
                X1 = P_ref_rot[contour, 0]
                X2 = P_ref_rot[contour, 1]
    
                x1 = P_def_rot[contour, 0]
                x2 = P_def_rot[contour, 1]
                x3 = P_def_rot[contour, 2]
    
                D = -D
                reversed_count += 1
    
            area_threshold = 0.1 * pitch_median**2
    
            if D < area_threshold:
                bad_points.append(point_id)
                continue
    
            dX1 = np.roll(X1, -1) - np.roll(X1, 1)
            dX2 = np.roll(X2, -1) - np.roll(X2, 1)
    
            S11 = np.sum(x1 * dX2)
            S21 = np.sum(x2 * dX2)
            S31 = np.sum(x3 * dX2)
    
            S12 = np.sum(x1 * dX1)
            S22 = np.sum(x2 * dX1)
            S32 = np.sum(x3 * dX1)
    
            F11 =  S11 / (2 * D)
            F21 =  S21 / (2 * D)
            F31 =  S31 / (2 * D)
    
            F12 = -S12 / (2 * D)
            F22 = -S22 / (2 * D)
            F32 = -S32 / (2 * D)
    
            E11 = 0.5 * (F11*F11 +F21*F21 +F31*F31 - 1)
    
            E22 = 0.5 * (F12*F12 +F22*F22 +F32*F32 - 1)
    
            E12 = 0.5 * (F11*F12 +F21*F22 +F31*F32)
    
            strainField[point_id, :] = [E11,E22,E12]
    
        print("========== 3D NN RESULTS ==========")
        print(f"Success  : {N - len(bad_points)}")
        print(f"Failed   : {len(bad_points)}")
        print(f"Reversed : {reversed_count}")
        print("===================================")
        if debug_file is not None:
    
            csv_file = debug_file.replace(".json", ".csv")
    
            with open(csv_file, "w") as f:
    
                f.write("point_id,E11,E22,E12\n")
    
                for i in range(N):
    
                    f.write(
                        f"{i},"
                        f"{strainField[i,0]},"
                        f"{strainField[i,1]},"
                        f"{strainField[i,2]}\n"
                    )
    
            debug = {
                "failed_points": [int(v) for v in bad_points],
                "failed_count": len(bad_points),
                "success_count": N - len(bad_points),
                "reversed_count": reversed_count,
                "pitch_median": float(pitch_median),
                "pitch_min": float(np.min(pitch_local)),
                "pitch_max": float(np.max(pitch_local)),
                "pitch_std": float(np.std(pitch_local))
            }
    
            with open(debug_file, "w") as f:
                ujson.dump(debug, f, indent=4)
        return strainField

def compute_strain_surface_Contour(P_ref_rot, P_def_rot, grid_nx, grid_ny):
    # retourne strainField (N,3) = [E11,E22,E12], calculus after rotation of the plan 
    shape = (grid_ny, grid_nx)

    X_ref = P_ref_rot[:, 0].reshape(grid_ny, grid_nx)
    Y_ref = P_ref_rot[:, 1].reshape(grid_ny, grid_nx)
    X_def = P_def_rot[:, 0].reshape(grid_ny, grid_nx)
    Y_def = P_def_rot[:, 1].reshape(grid_ny, grid_nx)
    Z_def = P_def_rot[:, 2].reshape(grid_ny, grid_nx)

    F11 = np.full(shape, np.nan); F12 = np.full(shape, np.nan)
    F21 = np.full(shape, np.nan); F22 = np.full(shape, np.nan)
    F31 = np.full(shape, np.nan); F32 = np.full(shape, np.nan)

    for j in range(grid_ny):
        for i in range(grid_nx):
            contour = get_contour_Contour_center(i, j, grid_nx, grid_ny)
            if len(contour) < 3:
                continue

            Xc = np.array([X_ref[jj, ii] for ii, jj in contour])
            Yc = np.array([Y_ref[jj, ii] for ii, jj in contour])


            x = np.array([X_def[jj, ii] for ii, jj in contour])
            y = np.array([Y_def[jj, ii] for ii, jj in contour])
            z = np.array([Z_def[jj, ii] for ii, jj in contour])

            D = 0.5 * np.sum(Xc*np.roll(Yc,-1) - Yc*np.roll(Xc,-1))
            if abs(D) < 1e-12:
                continue

            dX = np.roll(Xc,-1) - np.roll(Xc,1)
            dY = np.roll(Yc,-1) - np.roll(Yc,1)

            S11 = np.sum(x*dY); S21 = np.sum(y*dY); S31 = np.sum(z*dY)
            S12 = np.sum(x*dX); S22 = np.sum(y*dX); S32 = np.sum(z*dX)

            F11[j,i] =  S11/(2*D); F21[j,i] =  S21/(2*D); F31[j,i] =  S31/(2*D)
            F12[j,i] = -S12/(2*D); F22[j,i] = -S22/(2*D); F32[j,i] = -S32/(2*D)

    E11 = 0.5*(F11**2 + F21**2 + F31**2 - 1)
    E22 = 0.5*(F12**2 + F22**2 + F32**2 - 1)
    E12 = 0.5*(F11*F12 + F21*F22 + F31*F32)

    strain = np.stack([E11.ravel(), E22.ravel(), E12.ravel()], axis=1)
    return strain


def postprocess_current(ind_img, jsonPath_all, reference_point, indices_within_windows, resultFolderPath, scale, homography, postprocessDIC):
    
    _ = postprocessDIC.runPostprocessing(jsonPath_all, reference_point, indices_within_windows, resultFolderPath, ind_img, scale, homography)

    return ind_img,2

def postprocess_current_3D(ind_img, jsonPath_all, indices_within_windows, resultFolderPath, projectionMatrix1, projectionMatrix2, postprocessDIC):
    
    _ = postprocessDIC.runPostprocessing_3D(jsonPath_all, indices_within_windows, resultFolderPath, ind_img, projectionMatrix1, projectionMatrix2)

    return ind_img, 2

class PostprocessDIC:
    
    def __init__(self):
        # default bvalues (ereased by the UI)
        self.gridNx = 10
        self.gridNy = 10
        self.nnSearchK = 20
        self.nnDistanceFactor = 1.8
        self.nnMinNeighbours = 3
        self.nnMaxNeighbours = 8
            
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
    
    def createResultFolder(self, index=0):
    
        dic_folder = os.path.normpath(self.DICResultsPath)
    
        currentWorkingDir = os.path.dirname(dic_folder)
    
        resultFolder = os.path.join(
            currentWorkingDir,
            f"DIC_postprocessing{index}"
        )
    
        os.makedirs(resultFolder, exist_ok=True)
    
        self.outputName = resultFolder
    
        return resultFolder
        
    def savePostProcessingResult(self, DIC_PostProcessing, resultFolderPath, ind_images= None):
        if ind_images is not None:
            self.outputName = os.path.join(resultFolderPath, "DIC_postprocessing_%04d.json" % ind_images)
        else:
            self.outputName = os.path.join(resultFolderPath, 'DIC_processing.json')
        print("WRITING :", self.outputName)
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
    
    #calculus of the strain with a fucntion (the windows must be close enough)
    def compute_strain(self, reference_point, disp, indices_within_windows, scale=None, homography = None):
        num_points = reference_point.shape[0]
        #strainField = np.zeros((num_points, 3))
        strainField = np.full((num_points, 3), np.nan)   # NaN au lieu de 0
        n_skipped = 0
        if homography is not None:
           reference_point = cv2.perspectiveTransform(np.array([reference_point]), homography)[0]

        # Iterate through each reference point
        for i in range(num_points):
            indices_within_window = indices_within_windows[i]
            window_point = reference_point[indices_within_window, :]

            displacement = disp[indices_within_windows[i], :]

            if window_point.shape[0] < 3:  # Skip if not enough points for plane fitting
                n_skipped += 1
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
        print(f"compute_strain: {n_skipped}/{num_points} ignore points (window size is too small")
        return strainField
    
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
   
    def find_displacement(self, current_point, reference_point, scale = None, homography = None):
        if homography is not None:
            current_point = cv2.perspectiveTransform(np.array([current_point]), homography)[0]
            reference_point = cv2.perspectiveTransform(np.array([reference_point]), homography)[0]

        if scale is not None:
            current_point = current_point * scale
            reference_point = reference_point * scale

        return current_point - reference_point
    
    def triangulatePoint(self, projectionMatrix1, projectionMatrix2, point1, point2):
        point4D = cv2.triangulatePoints(projectionMatrix1, projectionMatrix2, point1.T, point2.T)
        point3D = point4D[:3] / point4D[3]
        point3D = point3D.T
        return point3D
    
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
        
    def runPostprocessing_2D_Contour(self, jsonPath_all, resultFolderPath, ind_img):
        # 2D points  (1 camera)
        current_point = self.readTrackingResult(jsonPath_all[ind_img])
        reference_point = self.readTrackingResult(jsonPath_all[0])
    
        # déplacement 2D
        displacementField = current_point - reference_point
    
        GRID_NX = int(getattr(self, "gridNx", 10))
        GRID_NY = int(getattr(self, "gridNy", 10))
    
        N = reference_point.shape[0]
        if N != GRID_NX * GRID_NY:
            raise ValueError(
                f"Grid mismatch 2D: N={N} points but GRID_NX*GRID_NY={GRID_NX*GRID_NY} "
                f"(NX={GRID_NX}, NY={GRID_NY})"
            )
    
        # strain 2D : (N,3) = [E11, E22, E12]
        strain2 = compute_strain_surface_Contour_2d(reference_point, current_point, GRID_NX, GRID_NY)
    
        postprocessingResults = {
            "currentPoint": current_point.tolist(),         # (N,2)
            "displacementField": displacementField.tolist(),# (N,2)
            "strainField": strain2.tolist()                 # (N,3)
            
        }
    
        self.savePostProcessingResult(postprocessingResults, resultFolderPath, ind_img)
        return postprocessingResults
    
    def runPostProcessingAll_2D_Contour(self, jsonPath_all, resultFolderPath, numProcess=1):
        print("jsonPath_all =", jsonPath_all)
        print("len(jsonPath_all) =", len(jsonPath_all))
        input_list = []
        for ind_img in range(len(jsonPath_all)):
            input_list.append((ind_img, jsonPath_all, resultFolderPath, self))
    
        pool = multiprocessing.Pool(processes=numProcess)

        list(tqdm(pool.imap(function_star_2D_Contour,input_list),total=len(input_list)))
        pool.close()
        pool.join()    
        

    def runPostprocessing_2D_NN(self,jsonPath_all,resultFolderPath,ind_img):

        current_point = self.readTrackingResult(jsonPath_all[ind_img])
        reference_point = self.readTrackingResult(jsonPath_all[0])    
        displacementField = (current_point - reference_point)
        debug_file = None
    
        if ind_img == 0:
            debug_file = os.path.join(resultFolderPath,"nearest_neighbour_debug.json") 
        strain2 = compute_strain_surface_Contour_2d_nn(
            reference_point,
            current_point,
            distance_factor=self.nnDistanceFactor,
            min_neighbours=self.nnMinNeighbours,
            max_neighbours=self.nnMaxNeighbours,
            debug_file=debug_file
        )
    
        postprocessingResults = {
            "currentPoint": current_point.tolist(),
            "displacementField": displacementField.tolist(),
            "strainField": strain2.tolist()
        }
        print("SAVE TO :", resultFolderPath)
        self.savePostProcessingResult(
            postprocessingResults,
            resultFolderPath,
            ind_img
        )
    
        return postprocessingResults
    
    def runPostProcessingAll_2D_NN(self,jsonPath_all,resultFolderPath,numProcess=1):
        
        print("jsonPath_all =", jsonPath_all)
        print("len(jsonPath_all) =", len(jsonPath_all))
        
        for ind_img in range(len(jsonPath_all)):
        
            print(f"Processing image {ind_img}")
        
            self.runPostprocessing_2D_NN(jsonPath_all=jsonPath_all,resultFolderPath=resultFolderPath,ind_img=ind_img)
        
        print("Postprocessing finished")
        
        return

    def runPostprocessing_3D(self, jsonPath_all, indices_within_windows, resultFolderPath, ind_img, projectionMatrix1, projectionMatrix2):
    
        # 1) Triangulation
        current_point1, current_point2 = self.readTrackingResult3D(jsonPath_all[ind_img])
        reference_point1, reference_point2 = self.readTrackingResult3D(jsonPath_all[0])
    
        P_ref = self.triangulatePoint(projectionMatrix1, projectionMatrix2, reference_point1, reference_point2)  # (N,3)
        P_def = self.triangulatePoint(projectionMatrix1, projectionMatrix2, current_point1, current_point2)      # (N,3)
    
        # 2) Fit plan + rotation (for the reference)
        dans_plan = (1, 5, 11)
        mark_in_plane = np.arange(P_ref.shape[0])
    
        M_x, M_y = fit_plane_rotation(P_ref, dans_plan=dans_plan, mark_in_plane=mark_in_plane)
        P_ref_rot, z0 = rotate_and_center(P_ref, M_x, M_y, z0=None, mark_in_plane=mark_in_plane)
        P_def_rot, _  = rotate_and_center(P_def, M_x, M_y, z0=z0,  mark_in_plane=mark_in_plane)
    
        # 3) Displacement (in the coordonate system “plan”)
        displacementField = (P_def_rot - P_ref_rot)
    
        # 4) Strain with a grid
        GRID_NX = int(getattr(self, "gridNx", 10))
        GRID_NY = int(getattr(self, "gridNy", 10))
    
        N = P_ref_rot.shape[0]
        if N != GRID_NX * GRID_NY:
            raise ValueError(f"Grid mismatch: N={N} points but GRID_NX*GRID_NY={GRID_NX*GRID_NY} (NX={GRID_NX}, NY={GRID_NY}).")
    
        strain3 = compute_strain_surface_Contour(P_ref_rot, P_def_rot, GRID_NX, GRID_NY)  # (N,3) -> [Exx,Eyy,Exy]
    
        postprocessingResults = {
            "currentPoint": current_point1.tolist(),             # 2D cam1 for display
            "displacementField": displacementField.tolist(),     # (N,3)
            "strainField": strain3.tolist(),                     # (N,3) = [E11,E22,E12]
            # 3D
            "referencePoint3D": P_ref.tolist(),                  # 3D triangulated (camera coordonate system)
            "currentPoint3D": P_def.tolist(),                    # 3D triangulated (camera coordonate system)
        
            # with the rotation of the plan in XY
            "referencePoint3D_rot": P_ref_rot.tolist(),          # 3D coordonate system with rotation (z=0)
            "currentPoint3D_rot": P_def_rot.tolist(),            # 3D coordonate system with rotation (z=0)
        }
        self.savePostProcessingResult(postprocessingResults, resultFolderPath, ind_img)
        return postprocessingResults


    def runPostprocessing_3D_NN(self,jsonPath_all,resultFolderPath,ind_img,projectionMatrix1,projectionMatrix2):
    
        current_point1, current_point2 = \
            self.readTrackingResult3D(jsonPath_all[ind_img])
    
        reference_point1, reference_point2 = \
            self.readTrackingResult3D(jsonPath_all[0])
    
        P_ref = self.triangulatePoint(projectionMatrix1,projectionMatrix2,reference_point1,reference_point2)
        P_def = self.triangulatePoint(projectionMatrix1,projectionMatrix2,current_point1,current_point2)
    
        dans_plan = (1,3,8)
    
        mark_in_plane = np.arange(P_ref.shape[0]) 
        M_x, M_y = fit_plane_rotation(P_ref,dans_plan=dans_plan,mark_in_plane=mark_in_plane)
        P_ref_rot, z0 = rotate_and_center(P_ref,M_x,M_y,mark_in_plane=mark_in_plane)
        P_def_rot, _ = rotate_and_center(P_def,M_x,M_y,z0=z0,mark_in_plane=mark_in_plane)  
        debug_file = None
        
        if ind_img == 0:
        
            debug_file = os.path.join(resultFolderPath,"nearest_neighbour_debug_3d.json")    
        displacementField = (P_def_rot - P_ref_rot)
    
        strain3 = compute_strain_surface_Contour_3d_nn(
            P_ref_rot,
            P_def_rot,
            distance_factor=self.nnDistanceFactor,
            min_neighbours=self.nnMinNeighbours,
            max_neighbours=self.nnMaxNeighbours,
            debug_file=debug_file
        )
    
        postprocessingResults = {
            "currentPoint": current_point1.tolist(),
            "displacementField": displacementField.tolist(),
            "strainField": strain3.tolist(),
    
            "referencePoint3D": P_ref.tolist(),
            "currentPoint3D": P_def.tolist(),
    
            "referencePoint3D_rot": P_ref_rot.tolist(),
            "currentPoint3D_rot": P_def_rot.tolist()
        }
    
        self.savePostProcessingResult(
            postprocessingResults,
            resultFolderPath,
            ind_img
        )
    
        return postprocessingResults


    def runPostProcessingAll_3D_NN(self, jsonPath_all,resultFolderPath,projectionMatrix1,projectionMatrix2,numProcess=1):
        
        input_list = []    
        for ind_img in range(len(jsonPath_all)):
    
            input_list.append((ind_img,jsonPath_all,resultFolderPath,projectionMatrix1,projectionMatrix2,self))
        pool = multiprocessing.Pool(processes=numProcess)
    
        list(tqdm(pool.imap(function_star_3D_NN,input_list),total=len(input_list)))
    
        pool.close()
        pool.join()
    

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
        
        list(tqdm(pool.imap(function_star_3D,input_list),total=len(input_list)))
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
        displacementTimeseries = np.zeros((len(jsonPath_all), displacementField.shape[1]))
        strainTimeseries = np.zeros((len(jsonPath_all), strainField.shape[1]))

        print("Extracting timeseries of point %d" % index)
        for i in tqdm(range(len(jsonPath_all))):
            currentPoint, displacementField, strainField = self.readPostProcessingResult(jsonPath_all[i])
            displacementTimeseries[i, :] = displacementField[index, :]
            strainTimeseries[i, :] = strainField[index, :]
        
        return displacementTimeseries, strainTimeseries