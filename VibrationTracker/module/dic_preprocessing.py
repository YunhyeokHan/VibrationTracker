import json
import cv2
import os, sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QListWidgetItem, QWidget, QLabel

import numpy as np
import matplotlib.pyplot as plt

try:
    from VibrationTracker.module.target_initialization import InitializeTarget
except ModuleNotFoundError:
    from target_initialization import InitializeTarget

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.path import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar

from matplotlib.patches import Polygon

class PreprocessDIC(InitializeTarget):

    def __init__(self, imagePath=None, calibPath=None):
        super().__init__(imagePath, calibPath)
        
        self.colors = np.random.rand(100, 3)


    def initUI(self):
        self.figure, self.ax = plt.subplots()

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)  # Add interactive toolbar for zoom and pan

        self.cid = []
        self.current_points = []
        self.polygon_list = []  # List to store all polygons

        main_layout = QHBoxLayout()

        # Left side: Matplotlib figure and buttons
        left_widget = QWidget()
        left_layout = QVBoxLayout()



        # Buttons
        button_layout = QHBoxLayout()
        self.add_polygon_mode_button = QPushButton("Add Polygon")
        button_layout.addWidget(self.add_polygon_mode_button)
        self.add_polygon_mode_button.setCheckable(True)
        self.add_polygon_mode_button.clicked.connect(self.activateAddPolygonMode)

        self.subtract_polygon_mode_button = QPushButton("Subtract Polygon")
        button_layout.addWidget(self.subtract_polygon_mode_button)
        self.subtract_polygon_mode_button.setCheckable(True)
        self.subtract_polygon_mode_button.clicked.connect(self.activateSubtractPolygonMode)

        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.toolbar)
        left_layout.addWidget(self.canvas, stretch=1)
        left_widget.setLayout(left_layout)

        # Right side: List of polygons and actions
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        self.polygon_list_widget = QListWidget()

        self.clear_last_point_button = QPushButton("Clear Last Point")
        self.clear_last_point_button.clicked.connect(self.clearLastPoint)

        self.reset_points_button = QPushButton("Reset Points")
        self.reset_points_button.clicked.connect(self.resetPoints)

        self.add_polygon_button = QPushButton("Confirm Polygon")
        self.add_polygon_button.clicked.connect(self.addPolygonToList)

        self.remove_selected_polygon_button = QPushButton("Remove Selected Polygon")
        self.remove_selected_polygon_button.clicked.connect(self.removeSelectedPolygon)

        self.remove_all_polygons_button = QPushButton("Remove All Polygons")
        self.remove_all_polygons_button.clicked.connect(self.removeAllPolygons)

        self.confirm_button = QPushButton("Confirm Selection")
        self.confirm_button.clicked.connect(self.confirmSelection)

        right_layout.addWidget(QLabel("Selected Polygons:"))

        right_layout.addWidget(self.polygon_list_widget, stretch=1)
        right_layout.addWidget(self.clear_last_point_button)
        right_layout.addWidget(self.reset_points_button)
        right_layout.addWidget(self.add_polygon_button)
        right_layout.addWidget(self.remove_selected_polygon_button)
        right_layout.addWidget(self.remove_all_polygons_button)
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

    def activateAddPolygonMode(self):
        """
        Enables Add Polygon mode, allowing user to select vertices for ROI.
        """
        self.deactivateOtherModes()
        self.add_polygon_mode_button.setChecked(True)
        self.current_mode = "add"
        self.current_points = []
        self.connectMouseEvents()

    def activateSubtractPolygonMode(self):
        """
        Enables Subtract Polygon mode, allowing user to remove areas from existing ROI.
        """
        self.deactivateOtherModes()
        self.subtract_polygon_mode_button.setChecked(True)
        self.current_mode = "subtract"
        self.current_points = []
        self.connectMouseEvents()

    def deactivateOtherModes(self):
        """ Deactivate other modes to avoid conflicts. """
        self.add_polygon_mode_button.setChecked(False)
        self.subtract_polygon_mode_button.setChecked(False)
        self.disconnectMouseEvents()

    def connectMouseEvents(self):
        """ Connect mouse click events for selecting points. """
        self.disconnectMouseEvents()
        self.cid.append(self.canvas.mpl_connect('button_press_event', self.onMouseClick))

    def disconnectMouseEvents(self):
        """ Disconnect any previous mouse event handlers. """
        for c in self.cid:
            self.canvas.mpl_disconnect(c)
        self.cid = []

    def onMouseClick(self, event):
        """ Handles mouse clicks to select polygon vertices. """
        if event.button == 3:  # Right mouse button
            current_polygon_index = len(self.polygon_list)
            point = (event.xdata, event.ydata)
            if point[0] is not None and point[1] is not None:
                self.current_points.append(point)
            # If the polygon is closed, add it to the list
            if hasattr(self, "temp_polygon") and self.temp_polygon in self.ax.patches:
                self.temp_polygon.remove()
            
            # Draw the polygon on the image
            if len(self.current_points) > 2:
                color = self.colors[current_polygon_index]
                
                if self.current_mode == "add":
                    self.temp_polygon = Polygon(self.current_points, closed=True, edgecolor=color, facecolor=color, alpha=0.3)
                else:
                    self.temp_polygon = Polygon(self.current_points, closed=True, edgecolor=color, facecolor=color, alpha=0.3)
                self.ax.add_patch(self.temp_polygon)  # 새 폴리곤 추가
            
            # draw the selected points
            self.ax.plot(point[0], point[1], 'bo')  # Blue points for selection
            
            # Draw the polygon on the image
            self.canvas.draw()



    def addPolygonToList(self):
        current_polygon_index = len(self.polygon_list)
        """ Adds the currently selected polygon to the list and displays it. """
        if len(self.current_points) < 3:
            print("At least 3 points are needed for a polygon.")
            return

        polygon_data = {"type": self.current_mode, "points": self.current_points}
        self.polygon_list.append(polygon_data)

        # Display in the list
        polygon_type = "Add" if self.current_mode == "add" else "Subtract"
        item_text = f"{polygon_type} Polygon - {len(self.current_points)} Points"
        self.polygon_list_widget.addItem(item_text)

        # Draw the polygon on the image

        color = self.colors[current_polygon_index]
        if self.current_mode == "add":
            polygon_patch = Polygon(self.current_points, closed=True, edgecolor=color, facecolor=color, alpha=0.3)
        else:
            polygon_patch = Polygon(self.current_points, closed=True, edgecolor=color, facecolor=color, alpha=1)
        self.ax.add_patch(polygon_patch)
        self.resetPoints()
        self.canvas.draw()

        # Reset current selection
        self.deactivateOtherModes()

    def clearLastPoint(self):
        """ Removes the last selected point from the current polygon. """
        if self.current_points:
            self.current_points.pop()
            self.refreshFigure()

    def resetPoints(self):
        """ Clears all points from the currently active polygon. """
        self.current_points = []
        self.refreshFigure()

    def removeSelectedPolygon(self):
        """ Removes the selected polygon from the list and clears the image. """
        selected_row = self.polygon_list_widget.currentRow()
        if selected_row >= 0:
            self.polygon_list.pop(selected_row)
            self.polygon_list_widget.takeItem(selected_row)
            self.refreshFigure()

    def removeAllPolygons(self):
        """ Removes all polygons from the list and clears the image. """
        self.polygon_list.clear()
        self.polygon_list_widget.clear()
        # self.ax.patches = []
        self.refreshFigure()

        self.canvas.draw()

    def refreshFigure(self):
        """ Clears and redraws the figure to reflect updates. """
        self.ax.clear()
        ref_image = cv2.imread(self.image_path)
        if self.calibResult is not None:
            ref_image = self.undistortImage(ref_image, self.calibResult[0], self.calibResult[1])
        self.ax.imshow(cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB))
        # Redraw all existing polygons
        for current_polygon_index, polygon_data in enumerate(self.polygon_list):
            color = self.colors[current_polygon_index]
            if polygon_data["type"] == "add":
                polygon_patch = Polygon(polygon_data["points"], closed=True, edgecolor=color, facecolor=color, alpha=0.3)    
                self.ax.add_patch(polygon_patch)

            else:
                polygon_patch = Polygon(polygon_data["points"], closed=True, edgecolor=color, facecolor=color, alpha=0.5)
                self.ax.add_patch(polygon_patch)

        # Draw current selected points
        for idx, point in enumerate(self.current_points):
            self.ax.plot(point[0], point[1], 'bo')

        self.canvas.draw()

    def confirmSelection(self):
        """ Outputs the final list of ROIs. """
        print("Final ROI List:", self.polygon_list)
    
        valid_points =  self.createMeshGrid(self.polygon_list, self.calibResult, meshSize=self.meshSize, stepSize=self.stepSize)

        self.saveDICPreprocessResults(valid_points, self.meshSize, self.resultFolder)

        self.close()



    def createMeshGrid(self, roi_data, calibResult=None, meshSize=21, stepSize=10):
        """
        Generates a uniform mesh grid within the specified ROI while ensuring each mesh cell is fully contained
        in the "add" regions and does not overlap with "subtract" regions.

        Parameters:
        - roi_data (list): A list of dictionaries containing polygon data with keys:
            - "type": "add" (for inclusion) or "subtract" (for exclusion)
            - "points": List of (x, y) coordinates defining the polygon
        - calibResult (tuple, optional): Camera calibration parameters (camera matrix, distortion coefficients).
        - meshSize (int, optional): Size of each mesh cell (default is 21).
        - stepSize (int, optional): Distance between mesh points (default is 10).

        Returns:
        - np.ndarray: Array of valid points (N, 2) where each row is (x, y).
        """

        # Load and undistort the image if calibration parameters are provided
        img = cv2.imread(self.image_path)
        if calibResult is not None:
            img = self.undistortImage(img, calibResult[0], calibResult[1])

        # Find the bounding box that encompasses all "add" polygons
        all_vertices = np.vstack([np.array(roi["points"]) for roi in roi_data if roi["type"] == "add"])
        x_min, x_max = np.ceil(np.min(all_vertices[:, 0])), np.floor(np.max(all_vertices[:, 0]))
        y_min, y_max = np.ceil(np.min(all_vertices[:, 1])), np.floor(np.max(all_vertices[:, 1]))

        # Generate a grid of points within the bounding box using the specified step size
        x = np.arange(x_min, x_max, stepSize)
        y = np.arange(y_min, y_max, stepSize)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel()]).T  # Convert to (N, 2) array of points

        halfMeshSize = (meshSize - 1) // 2  # Half-size of the mesh cells

        def is_rectangle_inside_polygon(center, polygon, flag="all"):
            """
            Checks if the rectangle centered at `center` with size `meshSize` fits entirely within `polygon`.
            """
            x, y = center
            leftcorner = (x - halfMeshSize, y - halfMeshSize)
            rightcorner = (x + halfMeshSize, y + halfMeshSize)
            # Check if all points of the rectangle are inside the polygon
            x_grid = np.linspace(leftcorner[0], rightcorner[0], int(2*halfMeshSize+1))
            y_grid = np.linspace(leftcorner[1], rightcorner[1], int(2*halfMeshSize+1))
            grid = np.meshgrid(x_grid, y_grid)

            rect_vertices = np.array([grid[0].flatten(), grid[1].flatten()]).T
            if flag == "all":   
                return np.all(polygon.contains_points(rect_vertices))
            else:
                return np.any(polygon.contains_points(rect_vertices))

      

        # Step 1: Keep only mesh squares fully inside "add" polygons
        valid_points = []
        for roi in roi_data:
            if roi["type"] == "add":
                add_polygon = Path(np.array(roi["points"]))
                for point in points:
                    if is_rectangle_inside_polygon(point, add_polygon):
                        valid_points.append(point)

        valid_points = np.array(valid_points)  # Convert to numpy array

        # Step 2: Remove mesh squares that overlap with "subtract" polygons
        filtered_points = []
        for point in valid_points:
            keep = True
            for roi in roi_data:
                if roi["type"] == "subtract":
                    subtract_polygon = Path(np.array(roi["points"]))
                    if is_rectangle_inside_polygon(point, subtract_polygon, flag='any'):
                        keep = False
                        break
            if keep:
                filtered_points.append(point)

        return np.array(filtered_points)  # Return the final set of valid mesh points




    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "DIC_preprocess_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath
    
    def saveDICPreprocessResults(self, posTrack, meshSize, resultFolderPath):
        self.outputName = os.path.join(resultFolderPath, 'DICpreprocessResults.json')
        posTrack = posTrack.tolist()
        dicpreprocessingResults = {"posTrack": posTrack, "meshSize": meshSize}
        with open(self.outputName, 'w') as f:
            json.dump(dicpreprocessingResults, f)
        return self.outputName
    
    def readDICPreprocessResults(self, jsonPath):
        with open(jsonPath) as f:
            dicpreprocessingResults = json.load(f)
        return dicpreprocessingResults
    

if __name__ == '__main__':

    app = QApplication(sys.argv)

    imagePath = r"example_assemblage\ImportImages_2636550566624\imagesNames.json"
    calibPath = r'example_assemblage\CalibrateCamera_2636435126976\calibrationResults.json'

    preprocessDIC = PreprocessDIC(imagePath, calibPath)

    preprocessDIC.resultFolder = preprocessDIC.createResultFolder(index=0)
    preprocessDIC.meshSize = 21
    preprocessDIC.stepSize = 10
    preprocessDIC.show()

    # mesh grid for DIC
    sys.exit(app.exec_())