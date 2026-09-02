import json
import cv2
import os
from PyQt5.QtWidgets import  QVBoxLayout, QHBoxLayout, QPushButton, QListWidget,  QWidget, QLabel
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import unary_union
import numpy as np
import matplotlib.pyplot as plt
try:
    from VibrationTracker.module.target_initialization import InitializeTarget
except ModuleNotFoundError:
    from target_initialization import InitializeTarget
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.patches import Polygon

class PreprocessDIC(InitializeTarget):
   
    def __init__(self, imagePath=None, calibPath=None):
        super().__init__(imagePath, calibPath)

        self.colors = np.random.rand(100, 3)
        self.gridNx = 10
        self.gridNy = 10


    def initUI(self):
        self.figure, self.ax = plt.subplots()

        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self) # Add interactive toolbar for zoom and pan

        self.cid = []
        self.current_points = []
        self.polygon_list = [] #List to store all polygons

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

    def readDICPreprocessResults(self, filepath):
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data

    #create the folder        
    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "DIC_preprocess_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        return resultFolderPath

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
        if event.button == 3: # Right mouse button
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
                self.temp_polygon = Polygon(self.current_points, closed=True, edgecolor=color, facecolor=color, alpha=0.3)
                self.ax.add_patch(self.temp_polygon) #add new polygon
                
            # draw the selected points
            self.ax.plot(point[0], point[1], 'bo')  # Blue points for selection
            # Draw the polygon on the image
            self.canvas.draw()

    def addPolygonToList(self):
        """ Adds the currently selected polygon to the list and displays it. """
        current_polygon_index = len(self.polygon_list)

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
        polygon_patch = Polygon(self.current_points, closed=True, edgecolor=color, facecolor=color, alpha=0.3)
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
            polygon_patch = Polygon(polygon_data["points"], closed=True, edgecolor=color, facecolor=color, alpha=0.3)
            self.ax.add_patch(polygon_patch)
        # Draw current selected points
        for point in self.current_points:
            self.ax.plot(point[0], point[1], 'bo')
        self.canvas.draw()

    def confirmSelection(self):
        """ Outputs the final list of ROIs. 2 modes : polygon ROI to draw subset 
        inside the polygon or Mesh mode to draw a grid with defined column and lign"""

        print("confirmSelection start")
    
        if self.type == "Polygon ROI":
    
            print("Polygon ROI mode")
    
            valid_points = self.createMeshGridPolygon(
                self.polygon_list,
                meshSize=self.meshSize,
                stepSize=self.stepSize
            )
    
        else:
    
            print("Mesh mode")
    
            valid_points = self.createMeshGrid(
                self.polygon_list,
                meshSize=self.meshSize,
                gridNx=self.gridNx,
                gridNy=self.gridNy
            )
    
        print("nb points =", len(valid_points))
    
          
        self.saveDICPreprocessResults(
            valid_points,
            self.meshSize,
            self.resultFolder
        )
        
        self.close()
    
        print("closed")
        

    #Outputs the final list of ROIs. 2 modes : polygon ROI to draw subset inside 
    #the polygon or Mesh mode to draw a grid with defined column and lign"

    def createMeshGrid(self,roi_data,meshSize=31,gridNx=10,gridNy=10):
  
        gridNx = int(gridNx)
        gridNy = int(gridNy)
    
        add_polygons = [roi for roi in roi_data if roi["type"] == "add"]
    
        if len(add_polygons) == 0:
            return np.zeros((0, 2))
    
        pts = np.array(add_polygons[0]["points"], dtype=float)
    
        if pts.shape[0] != 4:
            raise ValueError(
                "(This methode need a polygon with 4 vertex)."
            )
    
        P0 = pts[0]   # top left
        P1 = pts[1]   # top right
        #P2 = pts[2]   # bottom right
        P3 = pts[3]   # bottom left
    
        # horizontale direction 
        u = P1 - P0
        Lx = np.linalg.norm(u)
        u = u / Lx
    
        # verticale direction 
        v = P3 - P0
        Ly = np.linalg.norm(v)
        v = v / Ly
    
        # automatic calculated step
        stepX = Lx / max(gridNx - 1, 1)
        stepY = Ly / max(gridNy - 1, 1)
    
        points = []
    
        for iy in range(gridNy):
            for ix in range(gridNx):
    
                point = (
                    P0
                    + ix * stepX * u
                    + iy * stepY * v
                )
    
                points.append(point)
    
        points = np.array(points)
    
    
        return points
    
    def createMeshGridPolygon(self,roi_data, meshSize=21, stepSize=10,minOverlap=0.8):
    
        half = (meshSize - 1) / 2.0
    
        add_shapes = [
            ShapelyPolygon(np.array(roi["points"]))
            for roi in roi_data
            if roi["type"] == "add"
        ]
    
        if not add_shapes:
            return np.zeros((0, 2))
    
        roi_poly = unary_union(add_shapes)
    
        sub_shapes = [
            ShapelyPolygon(np.array(roi["points"]))
            for roi in roi_data
            if roi["type"] == "subtract"
        ]
    
        if sub_shapes:
            roi_poly = roi_poly.difference(
                unary_union(sub_shapes)
            )
    
        minx, miny, maxx, maxy = roi_poly.bounds
    
        xs = np.arange(minx, maxx + stepSize, stepSize)
        ys = np.arange(miny, maxy + stepSize, stepSize)
    
        square_area = meshSize * meshSize
    
        points = []
    
        for y in ys:
            for x in xs:
    
                square = ShapelyPolygon([
                    (x-half, y-half),
                    (x+half, y-half),
                    (x+half, y+half),
                    (x-half, y+half)
                ])
    
                overlap = roi_poly.intersection(square).area
    
                if overlap / square_area >= minOverlap:
                    points.append([x, y])
    
        return np.array(points)  
    

    # Draw the windows on the image
    def draw_windows(self, centers, meshSize, edgecolor="yellow", lw=1.2):
        half = (meshSize - 1) / 2.0

        for (x, y) in centers:
        
            corners = np.array([
                [x-half, y-half],
                [x+half, y-half],
                [x+half, y+half],
                [x-half, y+half]
            ])
        
            poly = Polygon(
                corners,
                fill=False,
                edgecolor=edgecolor,
                linewidth=lw
            )
        
            self.ax.add_patch(poly)
            

    
    # Save the data
    def saveDICPreprocessResults(self, posTrack, meshSize, resultFolderPath):
        self.outputName = os.path.join(resultFolderPath, 'DICpreprocessResults.json')
        posTrack = posTrack.tolist()
        dicpreprocessingResults = {"posTrack": posTrack, "meshSize": meshSize}
        with open(self.outputName, 'w') as f:
            json.dump(dicpreprocessingResults, f)
        return self.outputName
