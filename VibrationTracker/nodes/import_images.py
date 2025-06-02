from PyQt5.QtWidgets import QPushButton, QVBoxLayout, QLabel, QWidget, QGridLayout, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5 import QtGui
from VibrationTracker.vib_conf import register_node, OP_NODE_IMPORTIMAGES, VIB_NODES
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget, QDMTextEdit
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from matplotlib.figure import Figure

import cv2

from VibrationTracker.module.image_importer import *

class VibImportImagesContent(QDMNodeContentWidget):
    """
    A content widget for the image import node.

    This class provides a user interface for importing images using a folder browser.
    It contains:
    - A label to display the output image file names.
    - A button to open the folder browser dialog for importing images.
    """

    def initUI(self):
        """
        Initializes the user interface components of the widget.
        Sets up the layout, label, and folder browsing button.
        """
        self.setStyleSheet(''' font-size: 14px; ''')  # Set the font size for the widget

        # Main layout for the widget
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(10, 30, 10, 10)  # Set margins for better spacing

        # Placeholder label for spacing
        self.layout.addWidget(QLabel(""))

        # Output label to display the list of imported image names
        self.outputLabel = QLabel("ImageNames")
        self.layout.addWidget(self.outputLabel, alignment=Qt.AlignRight)

        # Additional empty label for spacing
        self.layout.addWidget(QLabel(""))

        # Button to open the folder browser dialog
        self.buttonRun = QPushButton("Folder Browser", self)
        self.buttonRun.setObjectName(self.node.content_label_objname)

        # Add the button to the layout
        self.layout.addWidget(self.buttonRun)

        # Apply the layout to the widget
        self.setLayout(self.layout)

    def serialize(self):
        """
        Serializes the widget's state for saving or exporting.

        Returns:
            dict: A dictionary containing the serialized state of the widget.
        """
        res = super().serialize()
        # Future serialization of additional state can be added here
        return res

    def deserialize(self, data, hashmap={}):
        """
        Deserializes the widget's state from saved data.

        Args:
            data (dict): Data to restore the state.
            hashmap (dict): Optional hashmap for ID mapping during deserialization.

        Returns:
            bool: True if deserialization was successful, False otherwise.
        """
        res = super().deserialize(data, hashmap)
        try:
            return True & res  # Combine the result with superclass deserialization result
        except Exception as e:
            dumpException(e)  # Log the exception for debugging
        return res


@register_node(OP_NODE_IMPORTIMAGES)
class VibNode_ImportImages(VibNode):
    """
    Node class for importing images into the VibrationTracker application.

    This node allows users to select a folder, import images, and visualize them.
    """
    # Operation code for registering the node
    op_code = OP_NODE_IMPORTIMAGES

    # Display title of the node
    op_title = "Import Images"

    # Object name used for identifying the content label
    content_label_objname = "vib_node_importimages"

    def __init__(self, scene):
        """
        Initializes the Import Images node with its scene and sets up input/output connections.
        
        Args:
            scene: The scene in which the node will be displayed.
        """
        super().__init__(scene, inputs=[], outputs=[1])  # No input, single output connection # 1 as mandatory output

    def initInnerClasses(self):
        """
        Initializes the inner UI and functionality classes for the node.
        """
        self.content = VibImportImagesContent(self)  # Content widget with UI elements
        self.grNode = VibGraphicsNode(self)  # Graphical representation of the node
        self.configWidget = VibNodeConfig_ImportImages(self)  # Configuration settings
        self.mainWidget = VibNodeMain_ImportImages(self)  # Main widget for image display
        self.imageImporter = ImageImporter()  # Image importer instance

        # Connect the folder browser button to the import function
        self.content.buttonRun.clicked.connect(self.runImageImport)

    def evalImplementation(self):
        """
        Evaluates the current state of the node and updates its output.
        """
        res = self.checkCurrentState()  # Check if images have been imported
        
        if res:
            # Update the node's value with the imported image JSON path
            u_value = self.getResultName()
            self.markDirty(False)  # Mark the node as clean
            self.markInvalid(False)  # Mark the node as valid

            self.value = u_value  # Store the output value
            print(self.value)

            # Update descendants in the node graph
            self.markDescendantsInvalid(False)
            self.markDescendantsDirty()

            # Set tooltip text
            self.grNode.setToolTip("Images imported successfully")
            self.evalChildren()  # Re-evaluate child nodes

            return self.value
        else:
            self.markDirty()  # Mark node as dirty if evaluation failed
            self.markInvalid()  # Mark node as invalid*
            self.grNode.setToolTip("Images not imported")  # Set tooltip text
            return None

    def runImageImport(self):
        """
        Handles the folder selection and imports images from the selected directory.
        """
        # Create result folder based on the current scene filename
        self.imageNames = None
        if self.scene.filename:
            self.resultFolder = os.path.join(os.path.dirname(self.scene.filename),
                                            os.path.splitext(os.path.basename(self.scene.filename))[0])
            self.folderName = os.path.join(self.resultFolder, "ImportImages_" + str(self.id))
            self.createFolder(self.folderName)  # Create a folder for imported images

            # Run the image import process
            
            self.imageNames = self.imageImporter.runImageImport(self.folderName)
            
            self.mainWidget.plotImage()
        else:
            # Display an error message if no scene filename is available
            QMessageBox.warning(None, "Error", "Save file before importing images.", QMessageBox.Ok)
            self.markDirty()  # Mark the node as dirty
            return
        if not self.imageNames:
            # Display an error message if no images are imported
            QMessageBox.warning(None, "Error", "No images found in the selected folder.", QMessageBox.Ok)
            self.markDirty()  # Mark the node as dirty
            self.markInvalid()

            return
        # Attempt to plot the first imported image

        self.evalImplementation()  # Evaluate the node after importing images
        
    def getResultName(self):
        """
        Returns:
            str: Path to the JSON file containing the imported image names.
        """
        return self.imageImporter.outputName

    def checkCurrentState(self):
        """
        Checks if there are any previously imported images saved in a JSON file.

        Returns:
            bool: True if the JSON file exists and images are loaded successfully, False otherwise.
        """
        # Set up the result folder path based on the scene filename
        self.resultFolder = os.path.join(os.path.dirname(self.scene.filename),
                                         os.path.splitext(os.path.basename(self.scene.filename))[0])
        self.folderName = os.path.join(self.resultFolder, "ImportImages_" + str(self.id))

        # Define the expected output JSON file path
        self.outputName = os.path.join(self.folderName, 'imagesNames.json')

        # Check if the JSON file exists and load the image names
        if os.path.isfile(self.outputName):
            self.imageImporter.outputName = self.outputName
            self.imageNames = self.imageImporter.readFromJson(self.outputName)
            if len(self.imageNames) == 0:
                return False
            # Plot the first image from the loaded list
            self.mainWidget.plotImage()

            return True
        else:
            return False  # No JSON file found, images not imported yet

    

class VibNodeConfig_ImportImages(QWidget):
    """
    Configuration panel for the Import Images node in the VibrationTracker application.

    This widget provides a placeholder for future configuration options. 
    Currently, it only displays a basic label indicating where configurations would be set.
    """

    def __init__(self, node):
        """
        Initializes the configuration widget.

        Args:
            node: The parent node that this configuration belongs to.
        """
        super().__init__()
        self.node = node  # Reference to the parent node
        self.initUI()  # Initialize the user interface

    def initUI(self):
        """
        Sets up the layout and UI elements for the configuration panel.
        """
        self.layout = QGridLayout()  # Create a grid layout 
        
        # Add a title label for the configuration section
        self.layout.addWidget(QLabel("Configurations"), 0, 0)
        
        # Placeholder for future configuration elements
        self.layout.addWidget(QLabel(""), 1, 0)
        
        # Apply the layout to the widget
        self.setLayout(self.layout)


from PyQt5.QtWidgets import QSlider, QLabel
from PyQt5.QtCore import Qt

class VibNodeMain_ImportImages(QWidget):
    """
    Main widget for displaying imported images using Matplotlib in the VibrationTracker application.

    This widget contains a Matplotlib figure, canvas, toolbar, and slider for navigating through images.
    """

    def __init__(self, node):
        """
        Initializes the main image display widget.

        Args:
            node: The parent node that this display widget is associated with.
        """
        super().__init__()
        self.node = node  # Reference to the parent node
        self.current_image_index = 0  # Current image index for slider
        self.xlim = None
        self.ylim = None

        self.initUI()  # Initialize the user interface components

    def initUI(self):
        """
        Sets up the UI layout, Matplotlib figure, canvas, toolbar, and image slider.
        """
        # Create a Matplotlib figure for displaying images
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')  # Set background color for the figure

        # Create a canvas for rendering the Matplotlib figure
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)  # Add a subplot (single axis)
        self.ax.set_facecolor('#666')  # Set background color for the axis
        self.ax.axis('off')  # Turn off axis display
        # Add a toolbar for interactive features like zooming and panning
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Create a slider for browsing through images
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.onSliderChanged)

        # Label to display the current image index
        self.index_label = QLabel("Image: 0")
        self.index_label.setAlignment(Qt.AlignCenter)

        # Set up the layout and add widgets
        self.layout = QGridLayout()
        self.layout.addWidget(self.toolbar, 0, 0)  # Add toolbar at the top
        self.layout.addWidget(self.canvas, 1, 0)  # Add canvas below the toolbar
        self.layout.addWidget(self.slider, 2, 0)  # Add slider below the canvas
        self.index_label.setMaximumHeight(20)
        self.layout.addWidget(self.index_label, 3, 0)  # Add image index label below the slider

        self.setLayout(self.layout)

    def plotImage(self, ind=0, xlim=None, ylim=None):
        """
        Plots an image on the Matplotlib canvas from the imported images.

        Args:
            ind (int): Index of the image to display from the image list.
        """
        if not hasattr(self.node, 'imageNames') or not self.node.imageNames:
            self.ax.clear()  # Clear the axis before plotting
            self.ax.axis('off')  # Turn off axis display
            self.slider.setMaximum(0)  # Set the slider range to 0
            print("No images available.")

            self.canvas.draw()  # Redraw the canvas with the updated image
            return

        # Set the slider range based on the number of images
        self.slider.setMaximum(len(self.node.imageNames) - 1)

        self.ax.clear()  # Clear the axis before plotting

        if xlim is not None:
            self.ax.set_xlim(xlim)
        if ylim is not None:
            self.ax.set_ylim(ylim)

        # Read the image from the file path using OpenCV
        img = cv2.imread(self.node.imageNames[ind])

        if img is not None:
            # Display the image on the canvas using Matplotlib
            self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert from BGR to RGB
            self.ax.axis('on')  # Turn off axis display

        else:
            print(f"Error: Could not load image at index {ind}")

        self.index_label.setText(f"Image: {ind + 1} / {len(self.node.imageNames)}")  # Update index label
        self.canvas.draw()  # Redraw the canvas with the updated image

    def onSliderChanged(self, value):
        """
        Callback function triggered when the slider value changes.
        Updates the displayed image according to the slider position.

        Args:
            value (int): The new slider value indicating the selected image index.
        """
        self.current_image_index = value # Update the current image index
        xlim = self.ax.get_xlim() # Get the current x-axis limits
        ylim = self.ax.get_ylim() # Get the current y-axis limits
        self.plotImage(value, xlim=xlim, ylim=ylim)




    
