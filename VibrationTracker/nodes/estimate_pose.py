from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QTableWidget, QTableWidgetItem,QMessageBox, QApplication, QGridLayout, QVBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from VibrationTracker.vib_conf import register_node, OP_NODE_ESTIMATEPOSE
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os
from VibrationTracker.module.pose_estimation import EstimatePose

class VibEstimatePoseContent(QDMNodeContentWidget):
    """
    This class defines the UI content for the VibEstimatePose node.
    It sets up the layout and widgets for input and output labels.
    """
    
    def initUI(self):
        """
        Initializes the UI by setting up the layout and adding widgets.
        """
        # Set the font size for the widget
        self.setStyleSheet(''' font-size: 14px; ''')

        # Create a grid layout with customized margins
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 20, 10, 30)

        # Add an empty label as a placeholder
        self.layout.addWidget(QLabel(""), 0, 0)

        # Input label for ImageNames
        self.inputLabel1 = QLabel("ImageNames")
        self.layout.addWidget(self.inputLabel1, 1, 0)

        # Input label for Calibration
        self.inputLabel2 = QLabel("Calibration")
        self.layout.addWidget(self.inputLabel2, 2, 0)

        # Spacer item to control layout spacing between elements
        spacer = QSpacerItem(40, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 1, 1, 1, 1)

        # Output label for EstimatedPose
        self.outputLabel = QLabel("EstimatedPose")
        self.layout.addWidget(self.outputLabel, 1, 2)

        # Add another empty label as a placeholder
        self.layout.addWidget(QLabel(""), 4, 2)

        # Set spacing between layout elements
        self.layout.setSpacing(1)
        
        # Apply the layout to the widget
        self.setLayout(self.layout)

    def serialize(self):
        """
        Serializes the node content and returns the dictionary containing data.
        """
        res = super().serialize()
        return res

    def deserialize(self, data, hashmap={}):
        """
        Deserializes the node content from the provided data.
        Returns True if successful, False otherwise.
        """
        res = super().deserialize(data, hashmap)
        try:
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_ESTIMATEPOSE)
class VibNode_EstimatePose(VibNode):
    """
    Node for estimating pose using different calibration patterns.

    This node supports different pose estimation methods, including:
    - Chessboard
    - Circle Grid
    - Charuco Pattern
    - Manual Selection (3D points assignment)

    Attributes:
        op_code (int): Node operation code.
        op_title (str): Display title of the node.
        content_label_objname (str): Internal label for node content.
    """

    op_code = OP_NODE_ESTIMATEPOSE
    op_title = "Estimate Pose"
    content_label_objname = "vib_node_estimatepose"

    def __init__(self, scene):
        """
        Initializes the pose estimation node with input and output slots.

        Args:
            scene (Scene): The graphical scene where the node is placed.
        """
        super().__init__(scene, inputs=[1, 1], outputs=[1, 2])        
        # self.eval()  # Delayed evaluation until necessary

    def initInnerClasses(self):
        """
        Initializes internal classes related to the node.
        """
        self.content = VibEstimatePoseContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_EstimatePose(self)
        self.mainWidget = VibNodeMain_EstimatePose(self)
        self.estimatePose = EstimatePose(self)
        
        # Connect the button click to runPoseEstimation method
        self.configWidget.buttonRun.clicked.connect(self.runPoseEstimation)

    def evalImplementation(self):
        """
        Evaluates the node's current state and updates its validity.

        Returns:
            str or None: Returns the output name if valid, otherwise None.
        """
        res = self.checkCurrentState()

        if res:
            self.markDirty(False)
            self.markInvalid(False)
            self.value = self.getResultName()
            print("Evaluation successful")
            self.markDescendantsInvalid(False)
            self.markDescendantsDirty()
            self.grNode.setToolTip("Pose estimation complete")
            return self.value
        else:
            self.markDirty()
            self.markInvalid()
            return None

    def runPoseEstimation(self):
        """
        Executes pose estimation based on the selected type in the configuration widget.
        """
        try:
            # Validate current state
            res = self.checkCurrentState()

            # Chessboard pose estimation
            if self.configWidget.type == "Chessboard":
                params = {
                    "numberCornersX": self.configWidget.numberCornersX,
                    "numberCornersY": self.configWidget.numberCornersY,
                    "sizeSquare": self.configWidget.sizeSquare,
                    "order": self.configWidget.order
                }
                print("Chessboard parameters:", params)
                img_res = self.estimatePose.estimatePose(
                    self.estimatePose.imagePath,
                    self.estimatePose.calibrationPath,
                    params=params,
                    result_path=self.resultFolder,
                    ind_img=self.configWidget.indexImg,
                    type_pose="Chessboard"
                )

            # Circle Grid pose estimation
            elif self.configWidget.type == "Circle Grid":
                params = {
                    "numberCirclesX": self.configWidget.numberCirclesX,
                    "numberCirclesY": self.configWidget.numberCirclesY,
                    "distanceBetweenCircles": self.configWidget.distanceBetweenCircles,
                    "order": self.configWidget.order
                }
                print("Circle Grid parameters:", params)
                img_res = self.estimatePose.estimatePose(
                    self.estimatePose.imagePath,
                    self.estimatePose.calibrationPath,
                    params=params,
                    result_path=self.resultFolder,
                    ind_img=self.configWidget.indexImg,
                    type_pose="Circle Grid"
                )

            # Charuco Pattern pose estimation
            elif self.configWidget.type == "Charuco Pattern":
                params = {
                    "numberSquaresX": self.configWidget.numberSquaresX,
                    "numberSquaresY": self.configWidget.numberSquaresY,
                    "sizeSquare": self.configWidget.sizeSquare,
                    "sizeMarker": self.configWidget.sizeMarker,
                    "markerType": self.configWidget.markerType,
                    "legacyPattern": self.configWidget.legacyPattern
                }
                print("Charuco Pattern parameters:", params)
                img_res = self.estimatePose.estimatePose(
                    self.estimatePose.imagePath,
                    self.estimatePose.calibrationPath,
                    params=params,
                    result_path=self.resultFolder,
                    ind_img=self.configWidget.indexImg,
                    type_pose="Charuco Pattern"
                )

            # Manual 3D Point Selection
            elif self.configWidget.type == "Manual Selection":
                print("Manual Selection - Assigning 3D points")
                point_2D = self.configWidget.points_2D
                print("Selected 2D points:", point_2D)
                point_3D = self.mainWidget.points_3D
                print("Selected 3D points:", point_3D)

                img_res = self.estimatePose.estimatePose_PointCorrespondences(
                    self.estimatePose.imagePath,
                    self.estimatePose.calibrationPath,
                    result_path=self.resultFolder,
                    ind_img=self.configWidget.indexImg,
                    points_2D=point_2D,
                    points_3D=point_3D
                )

            elif self.configWidget.type == "":
                print("Error: No pose estimation type selected")
                QMessageBox.critical(None, "Error", "No type selected")
                return

            self.img_res = img_res
            self.mainWidget.plotCalibration(img_res)
            self.estimatePose.outputName = os.path.join(self.resultFolder, 'poseEstimationResults.json')
            self.evalImplementation()
        except Exception as e:
            print("Pose estimation error:", e)
            QMessageBox.critical(None, "Error", f"Error in running pose estimation: {e}")

    def getResultName(self):
        """
        Retrieves the output file name for the pose estimation results.

        Returns:
            str: Output file name.
        """
        return self.estimatePose.outputName

    def checkCurrentState(self):
        """
        Checks the current state of the node by verifying input validity and result existence.

        Returns:
            bool: True if valid results exist, False otherwise.
        """
        self.estimatePose.imagePath = self.getInput(0).value
        self.estimatePose.filePath =  self.estimatePose.imagePath
 
        # Check if the second input (calibration file) is connected
        if self.getInput(1) is None:
            self.estimatePose.calibrationPath = None
        else:             
            self.estimatePose.calibrationPath = self.getInput(1).value
        
        self.resultFolder = self.estimatePose.createResultFolder(index=self.id)
        self.outputName = os.path.join(self.resultFolder, 'poseEstimationResults.json')

        print("Checking current state:", self.outputName)

        if os.path.exists(self.outputName):
            print("Existing results found")
            _, _, _, img_res = self.estimatePose.readPoseEstimationResults(self.outputName)
            self.mainWidget.plotCalibration(img_res)
            self.estimatePose.outputName = self.outputName
            return True
        else:
            return False

    def onInputChanged(self, socket):
        """
        Triggers evaluation when input values change.

        Args:
            socket (Socket): The input socket that changed.
        """
        print("Input changed:", socket)

        self.markDirty()
        # self.evalImplementation()



class VibNodeConfig_EstimatePose(QWidget):
    """
    Configuration widget for the camera pose estimation node.

    This widget provides UI elements to select and configure various pose estimation methods,
    including Chessboard, Circle Grid, Charuco Pattern, and Manual Selection.

    Attributes:
        node (VibNodeConfig_EstimatePose): The associated node instance.
        type (str): Selected pose estimation type.
        numberCornersX (int): Number of corners in X (for Chessboard).
        numberCornersY (int): Number of corners in Y (for Chessboard).
        sizeSquare (int): Size of a square (for Chessboard and Charuco).
        numberCirclesX (int): Number of circles in X (for Circle Grid).
        numberCirclesY (int): Number of circles in Y (for Circle Grid).
        distanceBetweenCircles (int): Distance between circles (for Circle Grid).
        numberSquaresX (int): Number of squares in X (for Charuco Pattern).
        numberSquaresY (int): Number of squares in Y (for Charuco Pattern).
        sizeMarker (int): Size of the marker (for Charuco Pattern).
        markerType (str): Type of the marker (for Charuco Pattern).
        legacyPattern (bool): Whether to use legacy pattern for Charuco.
        indexImg (int): Index of the image to process.
        order (bool): Reverse order flag.
    """

    def __init__(self, node):
        """
        Initializes the configuration widget with default parameters.

        Args:
            node: The associated node instance.
        """
        super().__init__()
        self.node = node

        # Initialize default parameters
        self.type = ''
        self.numberCornersX = 9
        self.numberCornersY = 5
        self.sizeSquare = 50
        self.numberCirclesX = 11
        self.numberCirclesY = 8
        self.distanceBetweenCircles = 15
        self.numberSquaresX = 10
        self.numberSquaresY = 6
        self.sizeMarker = 30
        self.markerType = 'DICT_5X5_1000'
        self.legacyPattern = True
        self.indexImg = 0
        self.order = False

        # Create the main layout only once
        self.layout = QGridLayout()

        self.initUI()

        self.setLayout(self.layout)


    def initUI(self):
        """
        Initializes the UI layout and common components.
        """
        # Clear any existing widgets (if necessary)
        self.clearLayout(self.layout)

        self.layout.addWidget(QLabel("Configurations for \nCamera Pose Estimation Node"), 0, 0)

        # Pose Estimation Type Selector
        self.layout.addWidget(QLabel("Pose Estimation Type"), 1, 0)
        self.typeSelector = QComboBox(self)
        self.typeSelector.addItems(['', 'Charuco Pattern', 'Chessboard', 'Circle Grid', 'Manual Selection'])
        self.layout.addWidget(self.typeSelector, 1, 1)
        self.typeSelector.activated[str].connect(self.onActivated_typeSelector)

        # Run Button (remains common for all configurations)
        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 9, 0, 1, 2)


    def validate_and_set_int(self, text, attribute_name, error_message):
        """
        Validates the text as an integer and sets it to the attribute if valid.
        Otherwise, shows a warning message.

        Args:
            text (str): The input text.
            attribute_name (str): The name of the attribute to update.
            error_message (str): Message to display if validation fails.
        """
        try:
            if text == '':
                value = 0
            else:
                value = int(text)
            setattr(self, attribute_name, value)
            print(f"Updated {attribute_name}: {value}")
        except ValueError:
            print(f"Invalid input for {attribute_name}: {text}")
            QMessageBox.warning(self, "Input Error", error_message)

    def onActivated_typeSelector(self, text):
        """
        Handles selection of the pose estimation type and updates the layout accordingly.

        Args:
            text (str): The selected type.
        """
        print("Activated", text)
        self.type = text
        print("Selected type:", self.type)

        # Clear current layout (except for top-level common widgets)
        self.clearLayout(self.layout)

        # self.initUI()

        if text == "Chessboard":
            self.layout_chessboard()

        elif text == "Circle Grid":
            self.layout_circleGrid()

        elif text == "Charuco Pattern":
            self.layout_charuco()

        elif text == "Manual Selection":
            self.layout_manual()

    def onChanged_numberCornersX(self, text):
        """Handles changes in the number of corners (X-axis) for Chessboard."""
        self.validate_and_set_int(text, "numberCornersX", "Please enter a valid integer for Number of Corners X.")

    def onChanged_numberCornersY(self, text):
        """Handles changes in the number of corners (Y-axis) for Chessboard."""
        self.validate_and_set_int(text, "numberCornersY", "Please enter a valid integer for Number of Corners Y.")

    def onChanged_sizeSquare(self, text):
        """Handles changes in the size of the square (for Chessboard & Charuco)."""
        self.validate_and_set_int(text, "sizeSquare", "Please enter a valid integer for Size of Square.")

    def onChanged_indexImg(self, text):
        """Handles changes in the selected image index."""
        self.validate_and_set_int(text, "indexImg", "Please enter a valid integer for Image Index.")

    def onChanged_numberCirclesX(self, text):
        """Handles changes in the number of circles in X for Circle Grid."""
        self.validate_and_set_int(text, "numberCirclesX", "Please enter a valid integer for Number of Circles in X.")

    def onChanged_numberCirclesY(self, text):
        """Handles changes in the number of circles in Y for Circle Grid."""
        self.validate_and_set_int(text, "numberCirclesY", "Please enter a valid integer for Number of Circles in Y.")

    def onChanged_distanceBetweenCircles(self, text):
        """Handles changes in the distance between circles (for Circle Grid)."""
        self.validate_and_set_int(text, "distanceBetweenCircles", "Please enter a valid integer for Distance Between Circles.")

    def onChanged_numberSquaresX(self, text):
        """Handles changes in the number of squares in X for Charuco Pattern."""
        self.validate_and_set_int(text, "numberSquaresX", "Please enter a valid integer for Number of Squares in X.")

    def onChanged_numberSquaresY(self, text):
        """Handles changes in the number of squares in Y for Charuco Pattern."""
        self.validate_and_set_int(text, "numberSquaresY", "Please enter a valid integer for Number of Squares in Y.")

    def onChanged_sizeMarker(self, text):
        """Handles changes in the marker size for Charuco Pattern."""
        self.validate_and_set_int(text, "sizeMarker", "Please enter a valid integer for Size of Marker.")

    def onActivated_markerType(self, text):
        """
        Handles selection of marker type for Charuco Pattern.

        Args:
            text (str): The selected marker type.
        """
        print("Activated markerType", text)
        self.markerType = text
        print("Marker type set to:", self.markerType)

    def onChanged_legacyPattern(self, state):
        """
        Handles changes in the legacy pattern checkbox.

        Args:
            state: The state of the checkbox.
        """
        print("Legacy pattern changed:", state)
        self.legacyPattern = (state == Qt.Checked)
        print("Legacy pattern set to:", self.legacyPattern)

    def onChanged_order(self, state):
        """
        Handles changes in the reverse order checkbox.

        Args:
            state: The state of the checkbox.
        """
        print("Order changed:", state)
        self.order = (state == Qt.Checked)
        print("Reverse order set to:", self.order)

    def layout_manual(self):
        """
        Sets up the UI layout for Manual Selection mode.
        """
        self.typeSelector.setCurrentText("Manual Selection")
        # Add 2D selection components
        self.layout.addWidget(QLabel("Select 2D points"), 2, 0)
        self.button_2D = QPushButton("Select", self)
        self.layout.addWidget(self.button_2D, 2, 1)

        # Add 3D assignment components
        self.layout.addWidget(QLabel("Assign 3D points"), 3, 0)
        self.button_3D = QPushButton("Assign", self)
        self.layout.addWidget(self.button_3D, 3, 1)

        # Additional dummy widgets (if needed)
        # self.layout.addWidget(QLabel("."), 4, 0)
        # self.layout.addWidget(QLabel("."), 5, 0)
        # self.layout.addWidget(QLabel("."), 6, 0)
        # self.layout.addWidget(QLabel("."), 7, 0)
        # self.layout.addWidget(QLineEdit(self), 4, 1)
        # self.layout.addWidget(QLineEdit(self), 5, 1)
        # self.layout.addWidget(QComboBox(self), 6, 1)
        # self.layout.addWidget(QCheckBox(self), 7, 1)

        # Index input for image selection
        self.layout.addWidget(QLabel("Index of the image"), 8, 0)
        indexImg = QLineEdit(self)
        indexImg.setText(str(self.indexImg))
        self.layout.addWidget(indexImg, 8, 1)
        indexImg.textChanged[str].connect(self.onChanged_indexImg)

        self.button_2D.clicked.connect(self.runButton_2D_clicked)
        self.button_3D.clicked.connect(self.runButton_3D_clicked)

    def runButton_2D_clicked(self):
        """
        Slot for handling the 2D point selection button click.
        Reads the image and calibration paths, processes the image names and calibration results,
        and triggers the 2D point selection.
        """
        print("Button 2D clicked")
        self.node.estimatePose.imagePath = self.node.getInput(0).value
        self.node.estimatePose.filePath = self.node.estimatePose.imagePath
        self.node.estimatePose.calibrationPath = self.node.getInput(1).value

        imageNames = self.node.estimatePose.readImageNamesFromJson(self.node.estimatePose.imagePath)
        calibResult = self.node.estimatePose.readCalibrationResults(self.node.estimatePose.calibrationPath)
        print("imageNames:", imageNames)
        print("calibResult:", calibResult)
        # Extract required calibration parameters
        calibResult = (calibResult["cameraMatrix"], calibResult["distortionCoefficients"])

        self.node.mainWidget.resetLayout()
        self.node.estimatePose.selectPointsInImage(imageNames, self.indexImg, calibResult)
        self.button_2D.setText("Selected")

    def runButton_3D_clicked(self):
        """
        Slot for handling the 3D point assignment button click.
        Tries to create a table for assigning 3D points. If points are not selected, shows an error.
        """
        print("Button 3D clicked")
        try:
            self.node.mainWidget.createTable()
            self.button_3D.setText("Assigned")
        except Exception as e:
            print("No points selected:", e)
            QMessageBox.critical(None, "Error", f"No points selected in the image. Please select points first: {e}")


    def clearLayout(self, layout):
        """
        Clears the layout by removing widgest except first 3 elements.

        Args:
            layout (QLayout): The layout to clear.
        """
        numWidgets = layout.count()
        for i in reversed(range(4, numWidgets)):
            item = layout.itemAt(i)
            print("Removing item:", item.widget())
            if item.widget() is not None:
                item.widget().setParent(None)

    def layout_chessboard(self):
        """
        Sets up the UI layout for Chessboard pose estimation.
        """
        self.typeSelector.setCurrentText("Chessboard")

        self.layout.addWidget(QLabel("Number of corners in X"), 2, 0)
        numberCornersX = QLineEdit(self)
        numberCornersX.setText(str(self.numberCornersX))
        self.layout.addWidget(numberCornersX, 2, 1)
        numberCornersX.textChanged[str].connect(self.onChanged_numberCornersX)

        self.layout.addWidget(QLabel("Number of corners in Y"), 3, 0)
        numberCornersY = QLineEdit(self)
        numberCornersY.setText(str(self.numberCornersY))
        self.layout.addWidget(numberCornersY, 3, 1)
        numberCornersY.textChanged[str].connect(self.onChanged_numberCornersY)

        self.layout.addWidget(QLabel("Size of the square (mm)"), 4, 0)
        sizeSquare = QLineEdit(self)
        sizeSquare.setText(str(self.sizeSquare))
        self.layout.addWidget(sizeSquare, 4, 1)
        sizeSquare.textChanged[str].connect(self.onChanged_sizeSquare)

        self.layout.addWidget(QLabel("Reverse order"), 7, 0)
        order = QCheckBox(self)
        order.setChecked(self.order)
        self.layout.addWidget(order, 7, 1)
        order.stateChanged.connect(self.onChanged_order)

        # Dummy widgets to fill space if needed
        # self.layout.addWidget(QLabel("."), 6, 0)
        # self.layout.addWidget(QComboBox(self), 6, 1)
        # self.layout.addWidget(QLabel("."), 5, 0)
        # self.layout.addWidget(QLineEdit(self), 5, 1)

        self.layout.addWidget(QLabel("Index of the image"), 8, 0)
        indexImg = QLineEdit(self)
        indexImg.setText(str(self.indexImg))
        self.layout.addWidget(indexImg, 8, 1)
        indexImg.textChanged[str].connect(self.onChanged_indexImg)

    def layout_circleGrid(self):
        """
        Sets up the UI layout for Circle Grid pose estimation.
        """
        self.typeSelector.setCurrentText("Circle Grid")
        self.layout.addWidget(QLabel("Number of circles in X"), 2, 0)
        numberCirclesX = QLineEdit(self)
        numberCirclesX.setText(str(self.numberCirclesX))
        self.layout.addWidget(numberCirclesX, 2, 1)
        numberCirclesX.textChanged[str].connect(self.onChanged_numberCirclesX)

        self.layout.addWidget(QLabel("Number of circles in Y"), 3, 0)
        numberCirclesY = QLineEdit(self)
        numberCirclesY.setText(str(self.numberCirclesY))
        self.layout.addWidget(numberCirclesY, 3, 1)
        numberCirclesY.textChanged[str].connect(self.onChanged_numberCirclesY)

        self.layout.addWidget(QLabel("Distance between circles (mm)"), 4, 0)
        distanceBetweenCircles = QLineEdit(self)
        distanceBetweenCircles.setText(str(self.distanceBetweenCircles))
        self.layout.addWidget(distanceBetweenCircles, 4, 1)
        distanceBetweenCircles.textChanged[str].connect(self.onChanged_distanceBetweenCircles)

        self.layout.addWidget(QLabel("Reverse order"), 7, 0)
        order = QCheckBox(self)
        order.setChecked(self.order)
        self.layout.addWidget(order, 7, 1)
        order.stateChanged.connect(self.onChanged_order)

        # self.layout.addWidget(QLabel("."), 6, 0)
        # self.layout.addWidget(QLabel("."), 5, 0)
        # self.layout.addWidget(QComboBox(self), 6, 1)
        # self.layout.addWidget(QLineEdit(self), 5, 1)

        self.layout.addWidget(QLabel("Index of the image"), 8, 0)
        indexImg = QLineEdit(self)
        indexImg.setText(str(self.indexImg))
        self.layout.addWidget(indexImg, 8, 1)
        indexImg.textChanged[str].connect(self.onChanged_indexImg)

    def layout_charuco(self):
        """
        Sets up the UI layout for Charuco Pattern pose estimation.
        """
        self.typeSelector.setCurrentText("Charuco Pattern")
        self.layout.addWidget(QLabel("Number of squares in X"), 2, 0)
        numberSquaresX = QLineEdit(self)
        numberSquaresX.setText(str(self.numberSquaresX))
        self.layout.addWidget(numberSquaresX, 2, 1)
        numberSquaresX.textChanged[str].connect(self.onChanged_numberSquaresX)

        self.layout.addWidget(QLabel("Number of squares in Y"), 3, 0)
        numberSquaresY = QLineEdit(self)
        numberSquaresY.setText(str(self.numberSquaresY))
        self.layout.addWidget(numberSquaresY, 3, 1)
        numberSquaresY.textChanged[str].connect(self.onChanged_numberSquaresY)

        self.layout.addWidget(QLabel("Size of the square (mm)"), 4, 0)
        sizeSquare = QLineEdit(self)
        sizeSquare.setText(str(self.sizeSquare))
        self.layout.addWidget(sizeSquare, 4, 1)
        sizeSquare.textChanged[str].connect(self.onChanged_sizeSquare)

        self.layout.addWidget(QLabel("Size of the marker (mm)"), 5, 0)
        sizeMarker = QLineEdit(self)
        sizeMarker.setText(str(self.sizeMarker))
        self.layout.addWidget(sizeMarker, 5, 1)
        sizeMarker.textChanged[str].connect(self.onChanged_sizeMarker)

        self.layout.addWidget(QLabel("Marker type"), 6, 0)
        markerType = QComboBox(self)
        markerType.addItem('DICT_5X5_1000')
        self.layout.addWidget(markerType, 6, 1)
        markerType.activated[str].connect(self.onActivated_markerType)

        self.layout.addWidget(QLabel("Legacy Pattern"), 7, 0)
        legacyPattern = QCheckBox(self)
        legacyPattern.setChecked(self.legacyPattern)
        self.layout.addWidget(legacyPattern, 7, 1)
        legacyPattern.stateChanged.connect(self.onChanged_legacyPattern)

        self.layout.addWidget(QLabel("Index of the image"), 8, 0)
        indexImg = QLineEdit(self)
        indexImg.setText(str(self.indexImg))
        self.layout.addWidget(indexImg, 8, 1)
        indexImg.textChanged[str].connect(self.onChanged_indexImg)




class VibNodeMain_EstimatePose(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()
        


    def initUI(self):
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')

        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')
        self.ax.axis('off')

        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.layout = QGridLayout()
        
        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)
        


        self.setLayout(self.layout)

    def clearLayout(self):
        for i in reversed(range(self.layout.count())): 
            item = self.layout.itemAt(i)
            if item.widget() is not None:  
                item.widget().setParent(None)

    def resetLayout(self):
        self.clearLayout()
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')

        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')
        self.ax.axis('off')

        self.toolbar = NavigationToolbar(self.canvas, self)
        
        # self.layout = QGridLayout()
        
        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)
        # self.setLayout(self.layout)

    def plotCalibration(self, undistortedImg):
        
        self.figure.clear()
        self.figure.patch.set_facecolor('#666')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')

        self.ax.imshow(undistortedImg)
        self.canvas.draw()


    def createTable(self):
        numPoints = len(self.node.estimatePose.points_array)
        position = self.mapToGlobal(self.pos())
        self.position = (position.x(),position.y())

        self.newWindowTable = NewWindow(self, numPoints, 4, intpos=self.position)
        self.newWindowTable.show()



class NewWindow(QWidget):
    def __init__(self, mainWidget, num_row, num_col, intpos = None):
        super().__init__()
        self.mainWidget = mainWidget

        # Set up the new window
        self.setWindowTitle('Assign 3D points')
        if intpos is not None:
            self.setGeometry(intpos[0], intpos[1], 500, 500)
        else:
            self.setGeometry(500, 500, 500, 500)

        # find the mouse position
        # self.setMouseTracking(True)

        # Create QTableWidget
        self.table_widget = TableWidget(num_row, num_col)  # 5 rows, 3 columns
        self.table_widget.setHorizontalHeaderLabels(['PointIndex', 'X position (mm)', 'Y position (mm)', 'Z position (mm)'])

        # Add some sample data
        for row in range(num_row):
            self.table_widget.setItem(row, 0, QTableWidgetItem('Point '+str(row)))


        # Set font size for the entire table
        font = QFont()
        font.setPointSize(12)  # Set the desired font size
        self.table_widget.setFont(font)

        # Set layout
        layout = QGridLayout()
        layout.addWidget(self.table_widget, 0, 0)
        self.assignButton = QPushButton('Assign', self)
        layout.addWidget(self.assignButton, 1, 0)
        
        self.assignButton.clicked.connect(self.onAssign)

        self.setLayout(layout)

    def onAssign(self):
        print("Assign button clicked")
        # get the values from the table
        values = []
        for row in range(self.table_widget.rowCount()):
            row_values = []
            for col in range(1,self.table_widget.columnCount()):
                item = self.table_widget.item(row, col)
                if item is not None:
                    row_values.append(item.text())
                else:
                    row_values.append('')
            values.append(row_values)
        self.close()
        self.mainWidget.points_3D = values

class TableWidget(QTableWidget):
    def __init__(self, rows, columns, parent=None):
        super().__init__(rows, columns)
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_V and (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
            selection = self.selectedIndexes()

            if selection:
                row_anchor = selection[0].row()
                column_anchor = selection[0].column()

                clipboard = QApplication.clipboard()

                rows = clipboard.text().split('\n')
                for indx_row, row in enumerate(rows):
                    values = row.split('\t')
                    for indx_col, value in enumerate(values):
                        item = QTableWidgetItem(value)
                        self.setItem(row_anchor + indx_row, column_anchor + indx_col, item)
            super().keyPressEvent(event)

        # delete the selected cell
        # Handle delete operation
        elif event.key() == Qt.Key.Key_Delete:
            selection = self.selectedIndexes()
            for index in selection:
                item = self.item(index.row(), index.column())  # Get the QTableWidgetItem from the index
                if item:  # Check if item is valid
                    item.setText('')  # Clear the text of the item
            super().keyPressEvent(event)




