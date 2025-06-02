from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QSpacerItem, QSizePolicy, QLineEdit, QMessageBox
from VibrationTracker.vib_conf import register_node, OP_NODE_CALCULATEDISPLACEMENT
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os

from VibrationTracker.module.displacement_calculation import CalculateDisplacement

class VibCalculateDisplacementContent(QDMNodeContentWidget):
    """
    UI content for the Calculate Displacement node.
    This class manages the layout and widgets for different camera setups.
    """
    def initUI(self):
        """ Initializes the UI layout and applies default settings. """
        self.setStyleSheet(''' font-size: 14px; ''')
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 23, 10, 25)
        self.setupUI_2CAM()
        self.setLayout(self.layout)

    def clearLayout(self):
        """ Clears all widgets from the current layout. """
        for i in reversed(range(self.layout.count())): 
            item = self.layout.itemAt(i)
            if item.widget() is not None:  
                item.widget().setParent(None)

    def setupUI_2CAM(self):
        """ Configures UI layout for a 2-camera setup. """
        self.clearLayout()
        self.inputLabel1 = QLabel("Tracking1")
        self.layout.addWidget(self.inputLabel1, 0, 0)
        
        self.inputLabel2 = QLabel("Pose1")
        self.layout.addWidget(self.inputLabel2, 1, 0)
        
        self.inputLabel3 = QLabel("Tracking2")
        self.layout.addWidget(self.inputLabel3, 2, 0)
        
        self.inputLabel4 = QLabel("Pose2")
        self.layout.addWidget(self.inputLabel4, 3, 0)
        
        spacer = QSpacerItem(6, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 1, 1, 1, 1)
        
        self.outputLabel1 = QLabel("Displacement 3D")
        self.layout.addWidget(self.outputLabel1, 1, 2)
        self.layout.setSpacing(1)

    def setupUI_1CAM_SCALEFACTOR(self):
        """ Configures UI layout for a single-camera setup using a scale factor. """
        self.clearLayout()
        
        self.inputLabel1 = QLabel("Tracking1")
        self.layout.addWidget(self.inputLabel1, 0, 0)
        
        self.inputLabel2 = QLabel(".")
        self.layout.addWidget(self.inputLabel2, 1, 0)
        
        self.inputLabel3 = QLabel(".")
        self.layout.addWidget(self.inputLabel3, 2, 0)
        
        self.inputLabel4 = QLabel(".")
        self.layout.addWidget(self.inputLabel4, 3, 0)
        
        self.outputLabel1 = QLabel("Displacement 2D")
        self.layout.addWidget(self.outputLabel1, 1, 2)
        self.layout.setSpacing(1)

    def setupUI_1CAM_HOMOGRAPHY(self):
        """ Configures UI layout for a single-camera setup using homography. """
        self.clearLayout()
        
        self.inputLabel1 = QLabel("Tracking1")
        self.layout.addWidget(self.inputLabel1, 0, 0)
        
        self.inputLabel2 = QLabel("Homography")
        self.layout.addWidget(self.inputLabel2, 1, 0)
        
        self.inputLabel3 = QLabel(".")
        self.layout.addWidget(self.inputLabel3, 2, 0)
        
        self.inputLabel4 = QLabel(".")
        self.layout.addWidget(self.inputLabel4, 3, 0)
        
        self.outputLabel1 = QLabel("Displacement 2D")
        self.layout.addWidget(self.outputLabel1, 1, 2)
        self.layout.setSpacing(1)

    def serialize(self):
        """ Serializes the node content. """
        res = super().serialize()
        return res

    def deserialize(self, data, hashmap={}):
        """ Deserializes the node content. """
        res = super().deserialize(data, hashmap)
        try:
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_CALCULATEDISPLACEMENT)
class VibNode_CalculateDisplacement(VibNode):
    """
    Node class for calculating displacement in a vibration analysis system.
    
    This node takes tracking results from single or stereo camera setups and
    computes displacement based on different configurations.
    """

    op_code = OP_NODE_CALCULATEDISPLACEMENT
    op_title = "Calculate Displacement"
    content_label_objname = "vib_node_calculate_displacement"

    def __init__(self, scene):
        """
        Initializes the node with input and output slots.
        
        Args:
            scene (Scene): The scene to which this node belongs.
        """
        super().__init__(scene, inputs=[1, 2, 1, 2], outputs=[1, 3])

    def initInnerClasses(self):
        """Initializes inner classes for UI and computation."""
        self.content = VibCalculateDisplacementContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_CalculateDisplacement(self)
        self.mainWidget = VibNodeMain_CalculateDisplacement(self)
        self.CalculateDisplacement = CalculateDisplacement()

        # Connect button to run displacement calculation
        self.configWidget.buttonRun.clicked.connect(self.runDisplacementCalculation3D)

    def evalImplementation(self):
        """
        Evaluates the node state and updates outputs accordingly.

        Returns:
            str or None: The name of the output file if valid, otherwise None.
        """
        res = self.checkCurrentState()
        print("res: ", res)

        if res:
            self.markDirty(False)
            self.markInvalid(False)

            self.value = self.getResultName()
            print(self.value)
            self.markDescendantsInvalid(False)
            self.markDescendantsDirty()

            self.grNode.setToolTip("789")
            return self.value
        else:
            self.markDirty()
            self.markInvalid()
            self.grNode.setToolTip("Connect all inputs")
            return None

    def runDisplacementCalculation3D(self):
        """
        Runs the displacement calculation based on the selected camera setup.
        
        Uses either stereo vision, scale factor, or homography for displacement
        computation and visualizes the results.
        """



        try:
            self.CalculateDisplacement.filePath = self.getInputs(0)[0].value
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Tracking file is missing")
            msg.setWindowTitle("Error")
            msg.exec_()
            return
        print("filePath: ", self.CalculateDisplacement.filePath)

        # Create result folder
        self.resultFolder = self.CalculateDisplacement.createResultFolder(index=self.id)

        numInput = len(self.inputs)
        numCamera = numInput // 2  # Integer division for correct calculation

        # Process displacement calculation based on the camera setup type
        if self.configWidget.type == 'Stereo Setup':
            list_TrackResultsPath = []
            list_poseEstimationResultsPath = []
            try:
                for i in range(numCamera):
                    list_TrackResultsPath.append(self.getInputs(2 * i)[0].value)
                    list_poseEstimationResultsPath.append(self.getInputs(2 * i + 1)[0].value)
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Tracking or Pose Estimation file is missing")
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            self.resultDisplacement = self.CalculateDisplacement.calculate3DPosition(
                list_TrackResultsPath, list_poseEstimationResultsPath, self.resultFolder
            )

        elif self.configWidget.type == 'Single Camera Setup - SCALE FACTOR':
            try:
                TrackResultsPath = self.getInputs(0)[0].value
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Tracking file is missing")
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            self.resultDisplacement = self.CalculateDisplacement.calculate2DDisplacement(
                TrackResultsPath, scale_factor=self.configWidget.scalefactor, resultFolderPath=self.resultFolder
            )

        elif self.configWidget.type == 'Single Camera Setup - HOMOGRAPHY':
            TrackResultsPath = self.getInputs(0)[0].value
            try:
                HomographyResultsPath = self.getInputs(1)[0].value
            except:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText("Homography file is missing")
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            self.resultDisplacement = self.CalculateDisplacement.calculate2DDisplacementWithHomography(
                TrackResultsPath, HomographyResultsPath, resultFolderPath=self.resultFolder
            )

        # Plot the first frame of the displacement results
        self.mainWidget.plotImage(ind=0, resultDisplacement=self.resultDisplacement)

        # Re-evaluate the node state
        self.eval()

    def getResultName(self):
        """
        Returns the name of the output file.

        Returns:
            str: The output file name containing displacement results.
        """
        return self.CalculateDisplacement.outputName

    def checkCurrentState(self):
        """
        Checks if the required input data is available and valid.

        Returns:
            bool: True if the calculation is ready to proceed, False otherwise.
        """

        # Check if the input file path is available
        try:
            self.CalculateDisplacement.filePath = self.getInputs(0)[0].value
        except:
            return False

        nameShouldbe = "TrackResults.json"
        if self.CalculateDisplacement.filePath[-len(nameShouldbe):] != nameShouldbe:
            return False
        # Create a result folder
        self.resultFolder = self.CalculateDisplacement.createResultFolder(index=self.id)

        # Define output file path
        self.CalculateDisplacement.outputName = os.path.join(self.resultFolder, 'displacementResults.json')

        # Check if the output JSON file already exists
        if os.path.isfile(self.CalculateDisplacement.outputName):
            self.resultDisplacement, _ = self.CalculateDisplacement.readDisplacementResults(self.CalculateDisplacement.outputName)

            # Get displacement result dimensions
            self.numFrame = self.resultDisplacement.shape[0]
            self.numPoint = self.resultDisplacement.shape[1]
            self.numDim = self.resultDisplacement.shape[2]

            # Visualize displacement results
            self.mainWidget.plotImage(ind=0, resultDisplacement=self.resultDisplacement)

            # Update point selector dropdown if new points are found
            for i in range(self.numPoint):
                if self.mainWidget.pointSelector.findText(f'Point {i}') == -1:
                    self.mainWidget.pointSelector.addItem(f'Point {i}')

            return True
        else:
            return False


class VibNodeConfig_CalculateDisplacement(QWidget):
    """
    Configuration panel for the displacement calculation node.
    
    This class provides UI elements for selecting the calculation method 
    and configuring parameters like scale factor.
    """
    
    def __init__(self, node):
        """
        Initializes the configuration widget.

        Args:
            node (VibNode_CalculateDisplacement): The parent node.
        """
        super().__init__()
        self.node = node
        self.type = 'Stereo Setup'  # Default setup type
        self.scalefactor = 1  # Default scale factor
        self.initUI()

    def initUI(self):
        """Initializes the user interface for the configuration panel."""
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Configuration title
        self.layout.addWidget(QLabel("Configurations of displacement calculation node"), 0, 0, 1, 2)

        # Dropdown menu for selecting calculation type
        self.layout.addWidget(QLabel("Type"), 1, 0)
        self.typeSelector = QComboBox(self)
        self.typeSelector.addItems([
            'Stereo Setup',
            'Multi-Camera Setup',
            'Single Camera Setup - HOMOGRAPHY',
            'Single Camera Setup - SCALE FACTOR'
        ])
        self.layout.addWidget(self.typeSelector, 1, 1)

        # Connect dropdown menu to selection handler
        self.typeSelector.activated[str].connect(self.onActivated_typeSelector)

        # Run button
        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 8, 0, 1, 2)

    def onActivated_typeSelector(self, text):
        """
        Handles changes in the calculation type selector and updates UI accordingly.

        Args:
            text (str): The selected calculation type.
        """
        print("Activated: ", text)
        self.type = text


        # Update UI based on selected type
        if text in ['Stereo Setup', 'Multi-Camera Setup']:
            self.node.content.setupUI_2CAM()
            self.setStereoLayout()
        elif text == 'Single Camera Setup - HOMOGRAPHY':
            self.node.content.setupUI_1CAM_HOMOGRAPHY()
            self.setHomographyLayout()
        elif text == 'Single Camera Setup - SCALE FACTOR':
            self.node.content.setupUI_1CAM_SCALEFACTOR()
            self.addScaleFactorInput()

    def setStereoLayout(self):
        self.clearLayout(self.layout)

    def setHomographyLayout(self):
        self.clearLayout(self.layout)

    def setScaleFactorLayout(self):
        self.clearLayout(self.layout)
        self.addScaleFactorInput()

    def addScaleFactorInput(self):
        """Adds Scale Factor input field when needed."""
        label = QLabel("Scale factor (mm/px)")
        self.scalefactorInput = QLineEdit(self)
        
        # Connect scale factor input field to validation function
        self.scalefactorInput.textChanged[str].connect(
            lambda text: self.validate_and_set_int(
                text, "scalefactor", "Scale Factor must be a valid number.", isfloat=True
            )
        )

        self.layout.addWidget(label, 5, 0)
        self.layout.addWidget(self.scalefactorInput, 5, 1)

    def validate_and_set_int(self, text, attribute_name, error_message, isfloat=False):
        """
        Validates the text input as an integer or float and updates the attribute.

        Args:
            text (str): The input text.
            attribute_name (str): The name of the attribute to update.
            error_message (str): Error message to display if validation fails.
            isfloat (bool): Whether the input should be validated as a float (default: False for int).
        """
        try:
            # Default value if input is empty
            value = 0 if text.strip() == '' else (float(text) if isfloat else int(text))
            setattr(self, attribute_name, value)
            print(f"Updated {attribute_name}: {value}")

        except ValueError:
            print(f"Invalid input for {attribute_name}: {text}")
            QMessageBox.warning(self, "Input Error", error_message)

    def clearLayout(self, layout):
        """
        Clears widgets from the given layout except 4 first widgets.

        Args:
            layout (QLayout): The layout to clear.
        """
        for i in reversed(range(4, layout.count())):
            item = layout.itemAt(i)
            if item.widget() is not None:
                item.widget().setParent(None)





class VibNodeMain_CalculateDisplacement(QWidget):
    """
    Main visualization panel for displacement calculation results.
    
    This widget provides matplotlib figures to plot displacement 
    results along different axes based on the selected calculation method.
    """

    def __init__(self, node):
        """
        Initializes the main visualization widget.

        Args:
            node (VibNode_CalculateDisplacement): The parent node.
        """
        super().__init__()
        self.node = node
        self.initUI()

    def initUI(self):
        """Initializes the UI components including the figure and dropdown selector."""
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')

        # Matplotlib canvas
        self.canvas = FigureCanvas(self.figure)

        # Subplots (3 by default, dynamically updated later)
        self.ax1 = self.figure.add_subplot(311)
        self.ax2 = self.figure.add_subplot(312)
        self.ax3 = self.figure.add_subplot(313)

        # Hide axes by default
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.axis('off')
            ax.set_facecolor('#666')

        # Toolbar for figure navigation
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout configuration
        self.layout = QGridLayout()
        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)

        # Dropdown for selecting displacement points
        self.pointSelector = QComboBox(self)
        self.layout.addWidget(self.pointSelector, 2, 0)
        self.pointSelector.activated[str].connect(self.onActivated_pointSelector)

        self.setLayout(self.layout)

    def onActivated_pointSelector(self, text):
        """
        Handles selection of a displacement point and updates the plot.

        Args:
            text (str): The selected point label (e.g., "Point 0").
        """
        print("Activated: ", text)
        
        # Ensure displacement data is available
        if not hasattr(self.node, 'resultDisplacement') or self.node.resultDisplacement is None:
            print("No displacement data available.")
            return

        # Extract the selected point index
        try:
            self.pointIndex = int(text.split(' ')[1])  # Extract index from "Point X"
            self.plotImage(ind=self.pointIndex, resultDisplacement=self.node.resultDisplacement)
        except (IndexError, ValueError):
            print(f"Invalid selection: {text}")

    def plotImage(self, ind=0, resultDisplacement=None):
        """
        Plots displacement results for the selected point.

        Args:
            ind (int): The index of the point to plot.
            resultDisplacement (numpy.ndarray): The displacement data array.
        """
        if resultDisplacement is None:
            print("No displacement data available for plotting.")
            return

        # Get the number of displacement dimensions (2D or 3D)
        is_3D = resultDisplacement.shape[2] > 2

        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()

        # Plot X and Y displacement
        self.ax1.plot(resultDisplacement[:, ind, 0], "k", linewidth=2)
        self.ax2.plot(resultDisplacement[:, ind, 1], "k", linewidth=2)

        # If data includes Z displacement (3D), plot it
        if is_3D:
            if not hasattr(self, 'ax3'):
                self.ax3 = self.figure.add_subplot(313)  # 다시 추가
            self.ax3.clear()
            self.ax3.plot(resultDisplacement[:, ind, 2], "k", linewidth=2)
            self.ax3.set_ylabel('Z (mm)')
            self.ax3.set_facecolor('#FFFFFF')
            self.ax3.set_xlabel('Frame')
        else:
            # Check if ax3 exists before deleting it
            if hasattr(self, 'ax3') and self.ax3 in self.figure.axes:
                self.figure.delaxes(self.ax3)
                del self.ax3  # 삭제 후 속성 제거

        # Set axis labels and colors
        self.ax1.set_title('Displacement')
        self.ax1.set_ylabel('X (mm)')
        self.ax2.set_ylabel('Y (mm)')

        self.ax1.set_facecolor('#FFFFFF')
        self.ax2.set_facecolor('#FFFFFF')

        self.figure.patch.set_facecolor('#666')

        # Refresh canvas
        self.canvas.draw()  # 새로 그리기                                                                                           

