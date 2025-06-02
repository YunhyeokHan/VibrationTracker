from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy,QMessageBox
from PyQt5.QtCore import Qt
from VibrationTracker.vib_conf import register_node, OP_NODE_CALIBRATECAMERA
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from VibrationTracker.module.camera_calibration import *

class VibCalibrateCameraContent(QDMNodeContentWidget):
    """
    A class representing the content widget for camera calibration in a node-based editor.
    
    This widget contains UI elements such as labels for input and output, and manages 
    serialization and deserialization of node data.
    """

    def initUI(self):
        """
        Initializes the user interface components of the widget.
        
        - Sets the font size.
        - Creates a grid layout with spacing.
        - Adds labels for 'ImageNames' (input) and 'Calibration' (output).
        - Inserts an empty label to adjust spacing.
        """
        self.setStyleSheet(''' font-size: 14px; ''')

        # Initialize the layout as a grid layout
        self.layout = QGridLayout()
        self.layout.setContentsMargins(15, 10, 15, 10)  # Set margins for layout

        # Add a spacer to create some vertical space
        space = QSpacerItem(0, 40, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(space, 0, 0)

        # Create and add an input label for "ImageNames"
        self.inputLabel = QLabel("ImageNames")
        self.layout.addWidget(self.inputLabel, 1, 0)

        # Create and add an output label for "Calibration" with right alignment
        self.outputLabel = QLabel("Calibration")
        self.layout.addWidget(self.outputLabel, 1, 2, alignment=Qt.AlignRight)

        # Add an empty label to adjust spacing in the layout
        self.layout.addWidget(QLabel(""), 5, 2)

        # Set the layout for the widget
        self.setLayout(self.layout)

    def serialize(self):
        """
        Serializes the widget's state and returns the data.

        Returns:
            dict: Serialized data containing the widget's state.
        """
        res = super().serialize()
        return res

    def deserialize(self, data, hashmap={}):
        """
        Deserializes the widget's state from the provided data.

        Args:
            data (dict): The serialized data to restore the widget's state.
            hashmap (dict, optional): A hashmap for tracking deserialization references. Defaults to an empty dictionary.

        Returns:
            bool: True if deserialization was successful, False otherwise.
        """
        res = super().deserialize(data, hashmap)
        try:
            # Ensure the deserialization process returns a valid boolean result
            return True & res
        except Exception as e:
            # Handle and log any exceptions that occur during deserialization
            dumpException(e)
        return res

@register_node(OP_NODE_CALIBRATECAMERA)
class VibNode_CalibrateCamera(VibNode):
    """
    A node class for performing camera calibration in a node-based system.
    
    This class initializes the calibration node, connects the UI elements, and 
    executes different calibration methods based on the selected pattern type.
    """

    # Node-specific attributes
    op_code = OP_NODE_CALIBRATECAMERA
    op_title = "Calibrate Camera"
    content_label_objname = "vib_node_calibratecamera"

    def __init__(self, scene):
        """
        Initializes the calibration node with one input and one output.

        Args:
            scene: The scene to which this node belongs.
        """
        super().__init__(scene, inputs=[1], outputs=[1])

    def initInnerClasses(self):
        """
        Initializes internal components required for the node:
        - UI components (content, graphics, config, main widget)
        - Camera calibration logic
        - Connects UI button to the calibration function
        """
        self.content = VibCalibrateCameraContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_CalibrateCamera(self)
        self.mainWidget = VibNodeMain_CalibrateCamera(self)

        # Instantiate the camera calibration object
        self.calibrateCamera = CalibrateCamera()

        # Connect the calibration button click event to the run function
        self.configWidget.buttonRun.clicked.connect(self.runCameraCalibration)

    def evalImplementation(self):
        """
        Evaluates the current state of the node and updates its status accordingly.

        Returns:
            str: The result name if the node is valid, otherwise None.
        """
        res = self.checkCurrentState()

        if res:
            # Update node state to valid
            self.markDirty(False)
            self.markInvalid(False)
            self.value = self.getResultName()
            print(self.value)

            # Update descendant nodes
            self.markDescendantsInvalid(False)
            self.markDescendantsDirty()

            # Set tooltip information
            self.grNode.setToolTip("456")

            return self.value
        else:
            # Mark node as invalid if the state check fails
            self.markDirty()
            self.markInvalid()
            return None

    def runCameraCalibration(self):
        """
        Executes the camera calibration process based on user input.
        - Checks input validity
        - Reads image names from a JSON file
        - Runs calibration based on the selected pattern type
        - Handles errors and displays relevant messages
        """
        try:
            # Retrieve input node
            input_node = self.getInput(0)
            if input_node is None or not hasattr(input_node, 'value'):
                QMessageBox.critical(None, "Error", "No input connected")
                return

            self.calibrateCamera.filePath = input_node.value

            # Validate file path (should end with 'imagesNames.json')
            if not self.calibrateCamera.filePath.endswith('imagesNames.json'):
                QMessageBox.critical(None, "Error", "Please select a valid input")
                return

            print("File Path: ", self.calibrateCamera.filePath)

        except Exception as e:
            print("Error: ", e)
            QMessageBox.critical(None, "Error", "Please connect a valid input")
            return

        # Read image names from the JSON file
        self.imagesNames = self.calibrateCamera.readImageNamesFromJson(self.calibrateCamera.filePath)

        # Create a folder for storing calibration results
        self.resultFolder = self.calibrateCamera.createResultFolder(index=self.id)

        # Run the appropriate calibration method based on the selected pattern
        if self.configWidget.type == "Chessboard":
            params = {
                "numberCornersX": self.configWidget.numberCornersX, 
                "numberCornersY": self.configWidget.numberCornersY, 
                "sizeSquare": self.configWidget.sizeSquare
            }
            print("Params: ", params)
            try:
                undistortedImg = self.calibrateCamera.runCalibrationChessboard(
                    self.imagesNames, self.resultFolder, params
                )
            except Exception as e:
                self._handleCalibrationError(e)
                return
                
        elif self.configWidget.type == "Circle Grid":
            params = {
                "numberCirclesX": self.configWidget.numberCirclesX, 
                "numberCirclesY": self.configWidget.numberCirclesY, 
                "distanceBetweenCircles": self.configWidget.distanceBetweenCircles
            }
            print("Params: ", params)
            try:
                undistortedImg = self.calibrateCamera.runCalibrationCirclesGrid(
                    self.imagesNames, self.resultFolder, params
                )
            except Exception as e:
                self._handleCalibrationError(e)
                return

        elif self.configWidget.type == "Charuco Pattern":
            params = {
                "numberSquaresX": self.configWidget.numberSquaresX, 
                "numberSquaresY": self.configWidget.numberSquaresY, 
                "sizeSquare": self.configWidget.sizeSquare, 
                "sizeMarker": self.configWidget.sizeMarker, 
                "markerType": self.configWidget.markerType, 
                "legacyPattern": self.configWidget.legacyPattern
            }
            print("Params: ", params)
            try:
                undistortedImg = self.calibrateCamera.runCalibrationCharuco(
                    self.imagesNames, self.resultFolder, params
                )
            except Exception as e:
                self._handleCalibrationError(e)
                return

        elif not self.configWidget.type:
            QMessageBox.critical(None, "Error", "Please select a calibration panel")
            return

        # Evaluate the node and update the UI
        self.evalImplementation()
        self.undistortedImg = undistortedImg
        self.mainWidget.plotCalibration(undistortedImg)

    def getResultName(self):
        """
        Retrieves the name of the output calibration file.

        Returns:
            str: The output filename of the calibration results.
        """
        return self.calibrateCamera.outputName

    def checkCurrentState(self):
        """
        Checks if a valid calibration result exists and updates the UI accordingly.

        Returns:
            bool: True if calibration results are found, False otherwise.
        """
        if self.getInput(0) is None:
            return False
        self.calibrateCamera.filePath = self.getInput(0).value
        self.imagesNames = self.calibrateCamera.readImageNamesFromJson(self.calibrateCamera.filePath)
        self.resultFolder = self.calibrateCamera.createResultFolder(index=self.id)
        self.outputName = os.path.join(self.resultFolder, 'calibrationResults.json')

        if not self.outputName:
            return False
        
        if os.path.exists(self.outputName):
            # Read calibration results
            calibrationResult = self.calibrateCamera.readCalibrationResults(self.outputName)
            cameraMatrix = calibrationResult["cameraMatrix"]
            distortionCoefficients = calibrationResult["distortionCoefficients"]

            # Generate an undistorted image using the calibration data
            undistortedImg = self.calibrateCamera.undistortImage(
                self.imagesNames[0], cameraMatrix, distortionCoefficients
            )
            self.calibrateCamera.outputName = self.outputName

            # Display the undistorted image
            self.mainWidget.plotCalibration(undistortedImg)

            return True

        else:
            return False

    def _handleCalibrationError(self, error):
        """
        Handles errors occurring during the calibration process and displays appropriate error messages.
        
        Args:
            error (Exception): The exception raised during calibration.
        """
        print("Error: ", error)
        if "nimages > 0" in str(error):
            QMessageBox.critical(None, "Error", "No control points found. Please check the calibration panel")
        elif "_charucoIds.total() > 0" in str(error):
            QMessageBox.critical(None, "Error", "No control points found. Please check the calibration panel")
        else:
            QMessageBox.critical(None, "Error", "Please check the calibration panel")



class VibNodeConfig_CalibrateCamera(QWidget):
    """
    A QWidget subclass that provides a configuration panel for the camera calibration node.

    Users can select a calibration pattern (Chessboard, Circle Grid, or Charuco Pattern)
    and specify relevant parameters such as the number of squares, marker size, and other properties.
    """

    def __init__(self, node):
        """
        Initializes the configuration panel.

        Args:
            node: The parent node to which this configuration belongs.
        """
        super().__init__()
        self.node = node
        self.initUI()

        # Default calibration parameters
        self.type = ''
        self.numberCornersX = 9
        self.numberCornersY = 5
        self.sizeSquare = 50  # 중복 선언 제거
        self.numberCirclesX = 11
        self.numberCirclesY = 8
        self.distanceBetweenCircles = 15
        self.numberSquaresX = 10
        self.numberSquaresY = 6
        self.sizeMarker = 30
        self.markerType = 'DICT_5X5_1000'
        self.legacyPattern = True

    def initUI(self):
        """Sets up the user interface elements for the calibration configuration panel."""
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configuration of calibration node"), 0, 0)

        # Calibration Panel Selection
        self.layout.addWidget(QLabel("Calibration Panel"), 1, 0)
        typeSelector = QComboBox(self)
        typeSelector.addItems(['', 'Charuco Pattern', 'Chessboard', 'Circle Grid'])
        self.layout.addWidget(typeSelector, 1, 1)

        # Run button
        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 8, 0, 1, 2)

        # Connect selection event
        typeSelector.activated[str].connect(self.onActivated_typeSelector)

        self.setLayout(self.layout)

    def onActivated_typeSelector(self, text):
        """Handles calibration pattern selection changes."""
        if text is None:
            return

        print("Activated", text)
        self.type = text

        # Clear previous widgets
        for i in reversed(range(4, self.layout.count())):
            widget = self.layout.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()
        

        # Update UI layout based on selection
        if text == "Chessboard":
            self.layout_chessboard()
        elif text == "Circle Grid":
            self.layout_circleGrid()
        elif text == "Charuco Pattern":
            self.layout_charuco()
    
    
    def validate_and_set_int(self, text, attribute_name, error_message, isfloat = False):
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
                if isfloat:
                    value = float(text)
                else:
                    value = int(text)
            setattr(self, attribute_name, value)
            print(f"Updated {attribute_name}: {value}")
        except ValueError:
            print(f"Invalid input for {attribute_name}: {text}")
            QMessageBox.warning(self, "Input Error", error_message)

    def layout_chessboard(self):
        """Updates the UI for the Chessboard calibration panel."""
        self.layout.addWidget(QLabel("Number of corners in X"), 2, 0)
        numberCornersX = QLineEdit(self)
        numberCornersX.setText(str(self.numberCornersX))
        self.layout.addWidget(numberCornersX, 2, 1)
        numberCornersX.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "numberCornersX", "Please enter a valid integer"))

        self.layout.addWidget(QLabel("Number of corners in Y"), 3, 0)
        numberCornersY = QLineEdit(self)
        numberCornersY.setText(str(self.numberCornersY))
        self.layout.addWidget(numberCornersY, 3, 1)
        numberCornersY.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "numberCornersY", "Please enter a valid integer"))

        self.layout.addWidget(QLabel("Size of the square (mm)"), 4, 0)
        sizeSquare = QLineEdit(self)
        sizeSquare.setText(str(self.sizeSquare))
        self.layout.addWidget(sizeSquare, 4, 1)
        sizeSquare.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "sizeSquare", "Please enter a valid float", True))
        
    def layout_circleGrid(self):
        """Updates the UI for the Circle Grid calibration panel."""
        self.layout.addWidget(QLabel("Number of circles in X"), 2, 0)
        numberCirclesX = QLineEdit(self)
        numberCirclesX.setText(str(self.numberCirclesX))
        self.layout.addWidget(numberCirclesX, 2, 1)
        numberCirclesX.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "numberCirclesX", "Please enter a valid integer"))

        self.layout.addWidget(QLabel("Number of circles in Y"), 3, 0)
        numberCirclesY = QLineEdit(self)
        numberCirclesY.setText(str(self.numberCirclesY))
        self.layout.addWidget(numberCirclesY, 3, 1)
        numberCirclesY.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "numberCirclesY", "Please enter a valid integer"))

        self.layout.addWidget(QLabel("Distance between circles (mm)"), 4, 0)
        distanceBetweenCircles = QLineEdit(self)
        distanceBetweenCircles.setText(str(self.distanceBetweenCircles))
        self.layout.addWidget(distanceBetweenCircles, 4, 1)
        distanceBetweenCircles.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "distanceBetweenCircles", "Please enter a valid float", True))

    def layout_charuco(self):
        """Updates the UI for the Charuco Pattern calibration panel."""
        self.layout.addWidget(QLabel("Number of squares in X"), 2, 0)
        numberSquaresX = QLineEdit(self)
        numberSquaresX.setText(str(self.numberSquaresX))
        self.layout.addWidget(numberSquaresX, 2, 1)
        numberSquaresX.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "numberSquaresX", "Please enter a valid integer"))

        self.layout.addWidget(QLabel("Number of squares in Y"), 3, 0)
        numberSquaresY = QLineEdit(self)
        numberSquaresY.setText(str(self.numberSquaresY))
        self.layout.addWidget(numberSquaresY, 3, 1)
        numberSquaresY.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "numberSquaresY", "Please enter a valid integer"))

        self.layout.addWidget(QLabel("Size of the square (mm)"), 4, 0)
        sizeSquare = QLineEdit(self)
        sizeSquare.setText(str(self.sizeSquare))
        self.layout.addWidget(sizeSquare, 4, 1)
        sizeSquare.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "sizeSquare", "Please enter a valid float", True))

        self.layout.addWidget(QLabel("Size of the marker (mm)"), 5, 0)
        sizeMarker = QLineEdit(self)
        sizeMarker.setText(str(self.sizeMarker))
        self.layout.addWidget(sizeMarker, 5, 1)
        sizeMarker.textChanged[str].connect(lambda text: self.validate_and_set_int(text, "sizeMarker", "Please enter a valid float", True))

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


    def onActivated_markerType(self, text):
        """Handles changes in the marker type selection for Charuco calibration."""
        if text:
            self.markerType = text

    def onChanged_legacyPattern(self, state):
        """
        Handles changes in the legacy pattern checkbox.

        Args:
            state: The state of the checkbox.
        """
        print("Legacy pattern changed:", state)
        self.legacyPattern = (state == Qt.Checked)
        print("Legacy pattern set to:", self.legacyPattern)


class VibNodeMain_CalibrateCamera(QWidget):
    """
    A QWidget subclass that displays calibration results using Matplotlib.
    
    This class provides a Matplotlib figure and canvas for displaying the undistorted image 
    after camera calibration.
    """

    def __init__(self, node):
        """
        Initializes the main UI for displaying calibration results.

        Args:
            node: The parent node to which this widget belongs.
        """
        super().__init__()
        self.node = node
        self.initUI()

    def initUI(self):
        """Sets up the Matplotlib canvas and toolbar for displaying calibration images."""
        self.figure = plt.figure()  # Create a new figure
        self.figure.patch.set_facecolor('#666')

        self.canvas = FigureCanvas(self.figure)  # Create a canvas for rendering the figure
        self.ax = self.figure.add_subplot(111)  # Add a subplot for displaying images
        self.ax.set_facecolor('#666')
        self.ax.axis('off')  # Hide axis labels for better visualization

        self.toolbar = NavigationToolbar(self.canvas, self)  # Add a navigation toolbar
        
        self.layout = QGridLayout()
        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)
        self.setLayout(self.layout)

    def plotCalibration(self, undistortedImg):
        """
        Plots the undistorted image from the camera calibration results.

        Args:
            undistortedImg (numpy.ndarray): The undistorted image to be displayed.
        """
        try:
            if undistortedImg is None:
                raise ValueError("Received None instead of an image.")

            if not isinstance(undistortedImg, np.ndarray):
                raise TypeError(f"Invalid image type: Expected numpy.ndarray, but got {type(undistortedImg)}")

            if undistortedImg.size == 0:
                raise ValueError("Received an empty image.")

            # Clear the existing figure before plotting a new one
            self.figure.clear()
            self.figure.patch.set_facecolor('#666')

            # Create a new subplot and set background color
            self.ax = self.figure.add_subplot(111)
            self.ax.set_facecolor('#666')
            self.ax.axis('off')  # Hide axis

            # Display the image
            self.ax.imshow(undistortedImg)

            # Efficiently update the canvas
            self.canvas.draw_idle()

        except Exception as e:
            print(f"Error in plotCalibration: {e}")



