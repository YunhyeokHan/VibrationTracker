import json
import cv2
import os, sys
import PyQt5.QtWidgets as QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QWidget, QLabel, QAction
from PyQt5.QtCore import pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar


class InitializeTarget(QMainWindow):
    closed = pyqtSignal()

    def __init__(self, filePath=None, calibPath=None):
        super().__init__()
    
        self.setWindowTitle("Point Selection")
        self.setGeometry(100, 100, 800, 600)
        if filePath is not None:
            self.filePath = filePath
        else:
            self.filePath = ''
        
        if calibPath is not None:
            self.calibPath = calibPath
        else:
            self.calibPath = ''

        self.dialog = CornerDetectionSettings(self)
        self.dialog.apply_button.clicked.connect(self.selectPointsInImage_harris)

    def show(self):
        if self.filePath == '':
            print("Please select a file")
        else:        
            imagesNames = self.readImageNamesFromJson(self.filePath)
            self.image_path = imagesNames[0]
            print("Image path: ", self.image_path)
        
        if self.calibPath == '':
            print("No calibration file provided")
            self.calibResult = None
        else:
            self.calibResult = self.readCalibNameFromJson(self.calibPath)
        
        self.initUI()
        super().show()


    def initUI(self):

        menu_bar = self.menuBar()
        
        file_menu = menu_bar.addMenu("File")
        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)
        
        edit_menu = menu_bar.addMenu("Edit")
        corner_settings_action = QAction("Corner Detection Settings", self)
        corner_settings_action.triggered.connect(self.open_corner_settings)
        edit_menu.addAction(corner_settings_action)
        self.figure, self.ax = plt.subplots()

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)  # Add interactive toolbar for zoom and pan

        self.cid = []
        self.points = []

        main_layout = QHBoxLayout()
        # Left side: Matplotlib figure and buttons
        left_widget = QWidget()

        left_layout = QVBoxLayout()

        # Buttons
        button_layout = QHBoxLayout()
        self.manual_button = QPushButton("Manual Selection")
        button_layout.addWidget(self.manual_button)
        self.manual_button.clicked.connect(self.selectPointsInImage)

        self.corner_button = QPushButton("Corner Detection")
        button_layout.addWidget(self.corner_button)
        self.corner_button.clicked.connect(self.selectPointsInImage_harris)
        # Add figure

        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.toolbar)  # Add the interactive toolbar

        left_layout.addWidget(self.canvas, stretch=1)
        left_widget.setLayout(left_layout)

        # Right side: List of points and confirm button
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        self.point_list = QListWidget()
        self.clear_button = QPushButton("Clear Last Point")
        self.clear_button.clicked.connect(self.clearLastPoint)
        self.reset_button = QPushButton("Reset Points")
        self.reset_button.clicked.connect(self.resetPoints)

        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirmPoints)
        right_layout.addWidget(QLabel("Selected Points:"))
        right_layout.addWidget(self.point_list, stretch=1)

        right_layout.addWidget(self.clear_button)
        right_layout.addWidget(self.reset_button)
        right_layout.addWidget(self.confirm_button)

        right_widget.setLayout(right_layout)

        # Combine layouts
        main_layout.addWidget(left_widget, stretch=2)
        main_layout.addWidget(right_widget, stretch=1)

        # Set the main widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.refreshFigure()


    def open_corner_settings(self):
        self.dialog.exec_()
    
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
        
    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "InitializeTarget_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath
    
    def saveInitializationResults(self, posTrack, resultFolderPath):
        self.outputName = os.path.join(resultFolderPath, 'initializationResults.json')
        initializationResults = {"posTrack": posTrack.tolist()}
        with open(self.outputName, 'w') as f:
            json.dump(initializationResults, f)
        print("Initialization results saved in: ", self.outputName)
        return self.outputName
    
    def readInitializationResults(self, jsonPath):
        with open(jsonPath) as f:
            initializationResults = json.load(f)
        return initializationResults
    
    def updatePointList(self):
        self.point_list.clear()
        for idx, point in enumerate(self.points):
            item = QListWidgetItem(f"{idx}: ({point[0]:.2f}, {point[1]:.2f})")
            self.point_list.addItem(item)
        self.point_list.itemDoubleClicked.connect(self.removePoint)

    def removePoint(self, item):
        index = self.point_list.row(item)
        del self.points[index]
        self.refreshFigure()
        self.updatePointList()

    def resetPoints(self):
        self.points = []
        self.refreshFigure()
        self.updatePointList()

    def clearLastPoint(self):
        if self.points:
            self.points.pop()
            self.refreshFigure()
            self.updatePointList()

    def refreshFigure(self):

        self.ax.clear()
        ref_image = cv2.imread(self.image_path)
        if self.calibResult is not None:
            ref_image = self.undistortImage(ref_image, self.calibResult[0], self.calibResult[1])
        self.ax.imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        for idx, point in enumerate(self.points):
            self.ax.plot(point[0], point[1], 'ro')
            self.ax.text(point[0], point[1], str(idx), color='r', fontsize=12)
        self.canvas.draw()

    def confirmPoints(self):
        print("Confirmed points:", self.points)
        points_array = np.array(self.points)

        self.saveInitializationResults(points_array, self.resultFolder)

        self.close()

    def closeEvent(self, event):
        self.closed.emit()  # Emit signal when window is closed
        super().closeEvent(event)

    def selectPointsInImage(self):

        self.manual_button.setEnabled(False)
        self.corner_button.setEnabled(True)
        if len(self.cid) > 0:
            for i in range(len(self.cid)):
                self.canvas.mpl_disconnect(self.cid[i])
        print("Select points in image by right-clicking")

        def onclick(event):
            if event.button == 3:  # Right mouse button clicked
                point = (event.xdata, event.ydata)
                if point[0] is not None and point[1] is not None:
                    self.points.append(point)
                    self.ax.plot(point[0], point[1], 'rx')
                    self.ax.text(point[0], point[1], str(len(self.points)-1), color='r', fontsize=12)
                    self.canvas.draw()
                    self.updatePointList()

        ref_image = cv2.imread(self.image_path)
        if self.calibResult is not None:
            ref_image = self.undistortImage(ref_image, self.calibResult[0], self.calibResult[1])

        for artist in reversed(self.ax.artists):
            artist.remove()
        for line in reversed(self.ax.lines):
            line.remove()

        self.ax.imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        self.ax.set_title("Select points by right-clicking")

        for ind_point, point in enumerate(self.points):
            self.ax.plot(point[0], point[1], 'rx')
            self.ax.text(point[0], point[1], str(ind_point), color='r', fontsize=12)

        self.cid.append(self.canvas.mpl_connect('button_press_event', onclick))
        self.canvas.draw()

    def onSecondWindowClose(self):
        print("Second window closed")
        self.selectPointsInImage_harris()


    def selectPointsInImage_harris(self):

        self.manual_button.setEnabled(True)
        self.corner_button.setEnabled(False)

        if len(self.cid) > 0:
            for i in range(len(self.cid)):
                self.canvas.mpl_disconnect(self.cid[i])

        highlighted_point = None  # index of highlighted point
        def onclick(event):
            if event.button == 3:  # Right mouse button clicked
                mouse = np.array([event.xdata, event.ydata])
                distance_between_MousePoints = np.linalg.norm(points_ref - mouse, axis=1)
                ind_min = np.argmin(distance_between_MousePoints)

                if distance_between_MousePoints[ind_min] < 10:  # reference distance for close points
                    highlighted_point = ind_min
                    point = points_ref[ind_min]
                    if point[0] is not None and point[1] is not None:
                        self.points.append(point)
                        self.ax.plot(point[0], point[1], 'rx')
                        self.ax.text(point[0], point[1], str(len(self.points)-1), color='r', fontsize=12)
                        self.canvas.draw()
                        self.updatePointList()

        def onMove(event):
            nonlocal highlighted_point
            if event.inaxes:
                mouse = np.array([event.xdata, event.ydata])
                distance_between_MousePoints = np.linalg.norm(points_ref - mouse, axis=1)
                ind_min = np.argmin(distance_between_MousePoints)

                # Highlight the closest point
                if distance_between_MousePoints[ind_min] < 10:  # reference distance for close points
                    if highlighted_point != ind_min:
                        highlighted_point = ind_min
                        self.refreshFigureCurrent(points_ref, colors, highlight=highlighted_point)
                else:
                    if highlighted_point is not None:
                        highlighted_point = None
                        self.refreshFigureCurrent(points_ref, colors, highlight=None)

        ref_image = cv2.imread(self.image_path)
        if self.calibResult is not None:
            ref_image = self.undistortImage(ref_image, self.calibResult[0], self.calibResult[1])
        img_gray = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
        max_corners = self.dialog.max_corners
        quality_level = self.dialog.quality_level
        min_distance = self.dialog.min_distance
        window_size = self.dialog.window_size
        max_iterations = self.dialog.max_iterations
        epsilon = self.dialog.epsilon

        corners = cv2.goodFeaturesToTrack(img_gray, max_corners, quality_level, min_distance)
        corners = cv2.cornerSubPix(img_gray, corners, (window_size, window_size), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iterations, epsilon))
        points_ref = corners.reshape(-1, 2)

        colors = np.random.random((len(points_ref), 3))
        self.refreshFigureCurrent(points_ref, colors)

        self.ax.imshow(ref_image)
        self.ax.set_title("Select points by right-clicking")


        self.cid.append(self.canvas.mpl_connect('button_press_event', onclick))
        self.cid.append(self.canvas.mpl_connect('motion_notify_event', onMove))
        self.canvas.draw()

    def refreshFigureCurrent(self, points_ref, colors, highlight=None):
        # # current figure limits
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        # # all artists are removed
        for artist in reversed(self.ax.artists):
            artist.remove()
        for line in reversed(self.ax.lines):
            line.remove()

        # update the view
        for ind_point, point in enumerate(points_ref):
            if highlight == ind_point:
                self.ax.plot(point[0], point[1], 'x', color=colors[ind_point], markersize=10, markeredgewidth=2)  # highlight point
            else:
                self.ax.plot(point[0], point[1], 'o', color=colors[ind_point])  # normal point

        for ind_point, point in enumerate(self.points):
            self.ax.plot(point[0], point[1], 'rx')
            self.ax.text(point[0], point[1], str(ind_point), color='r', fontsize=12)

        # set the limits back
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)

        self.canvas.draw()

from PyQt5.QtWidgets import QDialog, QFormLayout, QLineEdit, QPushButton


class CornerDetectionSettings(QDialog):
    closed = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Corner Detection Settings")
        self.setGeometry(100, 100, 300, 200)

        self.initUI()
        # Default parameters
        self.max_corners = 300
        self.quality_level = 0.01
        self.min_distance = 3
        self.window_size = 5
        self.max_iterations = 30
        self.epsilon = 0.001

    def initUI(self):
        layout = QFormLayout()
        
        self.corners_edit = QLineEdit("300")
        self.quality_edit = QLineEdit("0.01")
        self.min_dist_edit = QLineEdit("3")
        self.window_size_edit = QLineEdit("5")
        self.max_iter_edit = QLineEdit("30")
        self.eps_edit = QLineEdit("0.001")
        
        layout.addRow("Max Corners:", self.corners_edit)
        layout.addRow("Quality Level:", self.quality_edit)
        layout.addRow("Min Distance:", self.min_dist_edit)
        layout.addRow("Window Size:", self.window_size_edit)
        layout.addRow("Max Iterations:", self.max_iter_edit)
        layout.addRow("Epsilon:", self.eps_edit)

        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_settings)
        layout.addWidget(self.apply_button)
        
        self.setLayout(layout)

    def apply_settings(self):
        try:
            self.max_corners = int(self.corners_edit.text())
            self.quality_level = float(self.quality_edit.text())
            self.min_distance = int(self.min_dist_edit.text())
            self.window_size = int(self.window_size_edit.text())
            self.max_iterations = int(self.max_iter_edit.text())
            self.epsilon = float(self.eps_edit.text())
            
            print(f"Updated Parameters: Max Corners={self.max_corners}, Quality={self.quality_level}, Min Distance={self.min_distance}, Window Size={self.window_size}, Max Iter={self.max_iterations}, Epsilon={self.epsilon}")
        except ValueError:
            print("Invalid input. Please enter correct values.")


if __name__ == '__main__':

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    imagePath = r"example_assemblage\ImportImages_2636550566624\imagesNames.json"
    calibPath = r'example_assemblage\CalibrateCamera_2636435126976\calibrationResults.json'

    initializeTarget = InitializeTarget(imagePath, calibPath)
    initializeTarget.resultFolder = initializeTarget.createResultFolder(index=0)

    initializeTarget.show()
    sys.exit(app.exec_())


