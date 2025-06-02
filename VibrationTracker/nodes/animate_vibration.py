from PyQt5.QtWidgets import (
    QPushButton,
    QGridLayout,
    QLabel,
    QWidget,
    QComboBox,
    QSpacerItem,
    QSizePolicy,
    QLineEdit,
    QMessageBox,
)
from PyQt5.QtCore import Qt
from VibrationTracker.vib_conf import register_node, OP_NODE_ANIMATEVIBRATION
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode 
from nodeeditor.node_content_widget import QDMNodeContentWidget 
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.animation as animation
from VibrationTracker.module.vibration_animation import AnimateVibration
import itertools


class VibAnimateVibrationContent(QDMNodeContentWidget):

    def initUI(self):

        self.setStyleSheet(""" font-size: 14px; """)
        self.layout = QGridLayout()
        self.layout.setContentsMargins(8, 23, 10, 25)
        self.setupUI()
        self.setLayout(self.layout)

    def clearLayout(self):
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            if item.widget() is not None:
                item.widget().setParent(None)

    def setupUI(self):

        self.clearLayout()

        self.inputLabel1 = QLabel("DisplacementResults")
        self.layout.addWidget(self.inputLabel1, 0, 0)

        spacer = QSpacerItem(1, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 1, 1, 1, 1)

        self.outputLabel1 = QLabel("")
        self.layout.addWidget(self.outputLabel1, 0, 2)
        self.layout.setSpacing(1)

    def serialize(self):
        res = super().serialize()
        # res['value'] = self.edit.text()
        return res

    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            # value = data['value']
            # self.edit.setText(value)
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_ANIMATEVIBRATION)
class VibNode_AnimateVibration(VibNode):
    # icon = "icons/in.png" TODO
    op_code = OP_NODE_ANIMATEVIBRATION
    op_title = "Animate Vibration"
    content_label_objname = "vib_node_animate_vibration"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1], outputs=[])
        # self.eval()

    def initInnerClasses(self):
        self.content = VibAnimateVibrationContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_AnimateVibration(self)
        self.mainWidget = VibNodeMain_AnimateVibration(self)
        self.animateVibration = AnimateVibration()

        self.configWidget.buttonRun.clicked.connect(self.AnimateDisplacement3D)

    def evalImplementation(self):

        res = self.checkCurrentState()
        print("res: ", res)
        if res == True:

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

    def AnimateDisplacement3D(self):

        # self.figure = self.mainWidget.figure
        self.mainWidget.ax1.clear()
        self.mainWidget.ax1.set_facecolor("#666")
        try:
            self.animateVibration.filePath = self.getInputs(0)[0].value
        except:
            return
        print("filePath: ", self.animateVibration.filePath)
        if self.animateVibration.filePath == None:
            return
        else:
            # create result folder
            self.resultFolder = self.animateVibration.createResultFolder(index=self.id)
            self.position_all = self.animateVibration.readDisplacementdata(
                self.animateVibration.filePath
            )

            self.mainWidget.plotAnimation(resultDisplacement=self.position_all)
            self.eval()

    def getResultName(self):

        return self.animateVibration.outputName

    def checkCurrentState(self):
        try:
            self.animateVibration.filePath = self.getInputs(0)[0].value
        except:
            return False
        self.resultFolder = self.animateVibration.createResultFolder(index=self.id)
        self.animateVibration.outputName = self.resultFolder
        return True


class VibNodeConfig_AnimateVibration(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of animation node"), 0, 0)

        self.layout.addWidget(QLabel("Animate frequency"), 1, 0)
        animateFrequencySelector = QLineEdit(self)
        self.animateFrequency = 1
        animateFrequencySelector.setText(str(self.animateFrequency))
        self.layout.addWidget(animateFrequencySelector, 1, 1)
        animateFrequencySelector.textChanged[str].connect(
            self.onChanged_animateFrequency
        )

        self.layout.addWidget(QLabel("Scale"), 2, 0)
        animateScaleSelector = QLineEdit(self)
        self.animateScale = 10
        animateScaleSelector.setText(str(self.animateScale))
        self.layout.addWidget(animateScaleSelector, 2, 1)
        animateScaleSelector.textChanged[str].connect(self.onChanged_animateScale)
        self.buttonRun = QPushButton("Run", self)

        self.layout.addWidget(self.buttonRun, 8, 0, 1, 2)

        self.setLayout(self.layout)

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
    

    def onChanged_animateFrequency(self, text):
        
        self.validate_and_set_int(
            text,
            "animateFrequency",
            "Animate Frequency must be a number"
        )


    def onChanged_animateScale(self, text):
        self.validate_and_set_int(
            text,
            "animateScale",
            "Animate Scale must be a number"
        )

class VibNodeMain_AnimateVibration(QWidget):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.ani = None
        self.initUI()

    def initUI(self):
        self.initPlot()

        self.setLayout(self.layout)
        # self.show()

    def initPlot(self):
        self.figure = plt.figure()
        self.figure.patch.set_facecolor("#666")

        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(111, projection="3d")
        # set aspect ratio
        self.ax1.set_box_aspect([1, 1, 1])

        self.ax1.set_facecolor("#666")

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QGridLayout()
        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)

    def plotAnimation(self, resultDisplacement=None):
        # Clear the axis initially
        self.ax1.clear()
        if self.ani is not None:
            self.ani.event_source.stop()
        numFrame = resultDisplacement.shape[0]
        
        # Define the update function
        def update(frame):
            # Clear the axis for the current frame
            self.ax1.clear()
            
            # Current frame's displacement data
            # check 3D data 
            if np.shape(resultDisplacement)[2] == 3:
                newpoint = resultDisplacement[frame, :, :]
                # Reference point (initial frame)
                originpoint = resultDisplacement[0, :, :]
            elif np.shape(resultDisplacement)[2] == 2:
                newpoint = np.hstack((resultDisplacement[frame, :, :], np.zeros((resultDisplacement.shape[1], 1))))
                originpoint = np.hstack((resultDisplacement[0, :, :], np.zeros((resultDisplacement.shape[1], 1))))
            
            # Scatter plot for the current frame
            self.ax1.scatter(
                newpoint[:, 0], newpoint[:, 1], newpoint[:, 2], c="r", marker="o"
            )

            # Set axis limits based on the initial frame
            scale = self.node.configWidget.animateScale
            self.ax1.set_xlim(
                [min(originpoint[:, 0]) - scale, max(originpoint[:, 0]) + scale]
            )
            self.ax1.set_ylim(
                [min(originpoint[:, 1]) - scale, max(originpoint[:, 1]) + scale]
            )
            self.ax1.set_zlim(
                [min(originpoint[:, 2]) - scale, max(originpoint[:, 2]) + scale]
            )

            self.ax1.set_xlabel("X (mm)")
            self.ax1.set_ylabel("Y (mm)")
            self.ax1.set_zlabel("Z (mm)")

        # Animation interval in milliseconds
        interval = (
            1000 / self.node.configWidget.animateFrequency
        )  # Convert frequency to milliseconds

        # Create an infinite loop for the frames using itertools.cycle
        frames = itertools.cycle(range(numFrame))

        # Create the animation
        self.ani = animation.FuncAnimation(
            self.figure,
            update,
            frames=frames,
            interval=interval,
            repeat=True,
            cache_frame_data=False,
        )


        # Render the canvas
        self.canvas.draw()  # Draw the canvas
