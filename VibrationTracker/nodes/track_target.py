from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMessageBox, QVBoxLayout
from PyQt5.QtCore import Qt
from VibrationTracker.vib_conf import register_node, OP_NODE_TRACKTARGET
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import cv2

from VibrationTracker.module.target_tracking import *

class VibITrackTargetContent(QDMNodeContentWidget):
    def initUI(self):
        self.setStyleSheet(''' font-size: 14px; ''')

        self.layout = QGridLayout()
        self.layout.setContentsMargins(10,10,10,10)
        self.layout.addWidget(QLabel(""), 0, 0)
        # self.layout.addWidget(QLabel(""), 1, 0)
        # self.layout.addWidget(QLabel(""), 2, 0)
        # self.layout.addWidget(QLabel(""), 1, 0)

        self.inputLabel1 = QLabel("ImageNames")
        self.layout.addWidget(self.inputLabel1, 1, 0)
        self.inputLabel2 = QLabel("Calibration")
        self.layout.addWidget(self.inputLabel2, 2, 0)
        self.inputLabel3 = QLabel("PosTrack")
        self.layout.addWidget(self.inputLabel3, 3, 0)

        spacer = QSpacerItem(55, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 2, 1, 1, 1)
        self.outputlabel = QLabel("TrackResults")
        self.layout.addWidget(self.outputlabel, 2, 2)

        self.layout.addWidget(QLabel(""), 4, 2)

        self.setLayout(self.layout)

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

@register_node(OP_NODE_TRACKTARGET)
class VibNode_TrackTarget(VibNode):
    # icon = "icons/in.png" TODO
    op_code = OP_NODE_TRACKTARGET
    op_title = "Track Target"
    content_label_objname = "vib_node_track_target"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1,2,3], outputs=[1])        
        # self.eval()

    def initInnerClasses(self):
        self.content = VibITrackTargetContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_TrackTarget(self)
        self.mainWidget = VibNodeMain_TrackTarget(self)

        self.trackTarget = TrackTarget()
        self.configWidget.buttonRun.clicked.connect(self.runTargetTracking)
        self.configWidget.buttonCheck.clicked.connect(self.setupTargetTracking)


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
            return None
        
    def setupTargetTracking(self):
        try:
            self.trackTarget.filePath = self.getInputs(0)[0].value
        except Exception as e:
            print("Error: ", e)
            QMessageBox.critical(None, "Error", f"Please connect the input {e}")
            return
        
        print("filePath: ", self.trackTarget.filePath)
        if len(self.getInputs(1)) > 0:
            self.trackTarget.calibPath = self.getInputs(1)[0].value
            print("calibPath: ", self.trackTarget.calibPath)
            self.calibResult = self.trackTarget.readCalibNameFromJson(self.trackTarget.calibPath)
        else:
            self.trackTarget.calibPath = None
            self.calibResult = None
        
        self.imagesNames = self.trackTarget.readImageNamesFromJson(self.trackTarget.filePath)
        self.resultFolder = self.trackTarget.createResultFolder(index=self.id)

        self.trackTarget.posTrackPath = self.getInputs(2)[0].value
        self.trackTarget.posTrack = self.trackTarget.readInitializeTarget(self.trackTarget.posTrackPath)
        self.mainWidget.plotCurrentState()
        self.markInvalid()

    def runTargetTracking(self):
        self.setupTargetTracking()

        if self.configWidget.method == "LK Optical Flow":
            TrackResults = self.trackTarget.trackTarget_LKOF(self.imagesNames, self.trackTarget.posTrack, self.resultFolder, winsize=self.configWidget.sizeWindow, maxLevel=self.configWidget.maxLevel, calibResult= self.calibResult, update = self.configWidget._update, show = self.configWidget._show)
        elif self.configWidget.method == "DIC with ZNSSD":
            try: 
                TrackResults = self.trackTarget.trackTarget_DIC(self.imagesNames, self.trackTarget.posTrack, self.resultFolder, winsize=self.configWidget.sizeWindow, search = self.configWidget.search, calibResult= self.calibResult, update=self.configWidget._update, show = self.configWidget._show, reinitialize = self.configWidget._reinitialize)
            except Exception as e:
                print("Error: ", e)
                if type(e) == ValueError:
                    QMessageBox.critical(None, "Error", f"Size of searching area is too small to detect large displacement {e}")
                return
        # self.trackTarget.plotTrackingResult(TrackResults)
        self.eval()


    def getResultName(self):
        return self.trackTarget.outputName

    def checkCurrentState(self):
    
        self.trackTarget.filePath = self.getInputs(0)[0].value
        print("filePath: ", self.trackTarget.filePath)
        if len(self.getInputs(1)) > 0:
            self.trackTarget.calibPath = self.getInputs(1)[0].value
            print("calibPath: ", self.trackTarget.calibPath)
            self.calibResult = self.trackTarget.readCalibNameFromJson(self.trackTarget.calibPath)
        else:
            self.trackTarget.calibPath = None
            self.calibResult = None
        
        self.imagesNames = self.trackTarget.readImageNamesFromJson(self.trackTarget.filePath)
        self.resultFolder = self.trackTarget.createResultFolder(index=self.id)
        
        self.trackTarget.posTrackPath = self.getInputs(2)[0].value
        self.trackTarget.posTrack = self.trackTarget.readInitializeTarget(self.trackTarget.posTrackPath)

        self.outputName = os.path.join(self.resultFolder, "TrackResults.json")
        # check if the output file exists

        if os.path.exists(self.outputName):
            self.trackTarget.outputName = self.outputName
            # read the output file
            TrackResults = self.trackTarget.readTrackingResult(self.outputName)
            # plot the tracking result
            self.mainWidget.plotTrackingResult(TrackResults, 0)
            return True
        else:
            return False

class VibNodeConfig_TrackTarget(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()
        self.method = ""
        self.sizeWindow = 21
        self.maxLevel = 1
        self._update = False
        self._show = False
        self.search = 30
        self._reinitialize = True

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of Target Tracking Node"), 0, 0)

        ## 
        self.layout.addWidget(QLabel("Method"), 1, 0)
        methodSelector = QComboBox(self)
        methodSelector.addItem('')
        methodSelector.addItem('LK Optical Flow')
        methodSelector.addItem('DIC with ZNSSD')
        self.layout.addWidget(methodSelector, 1, 1)
        methodSelector.activated[str].connect(self.onActivated_methodSelector)

        self.layout.addWidget(QLabel("."), 2, 0)
        self.layout.addWidget(QLabel("."), 3, 0)
        self.layout.addWidget(QLabel("."), 4, 0)
        self.layout.addWidget(QLabel("."), 5, 0)
        self.layout.addWidget(QLabel("."), 6, 0)


        self.buttonCheck = QPushButton("Check", self)
        self.layout.addWidget(self.buttonCheck, 7, 0, 1, 1)

        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 7, 1, 1, 1)

        self.setLayout(self.layout)

    def onActivated_methodSelector(self, text):
        print("Activated: ", text)
        self.method = text

        print("method: ", self.method)

        if text == "LK Optical Flow":
            self.layout_LK()

        elif text == "DIC with ZNSSD":
            self.layout_DIC()

    def layout_LK(self):
        self.layout.addWidget(QLabel("Size of the window"), 2, 0)
        sizeWindow = QLineEdit(self)
        sizeWindow.setText(str(self.sizeWindow))
        # sizeWindow.setAlignment(Qt.AlignRight)
        self.layout.addWidget(sizeWindow, 2, 1)
        sizeWindow.textChanged.connect(self.onChanged_sizeWindow)


        self.layout.addWidget(QLabel("Pylamide Level"), 3, 0)
        maxLevel = QLineEdit(self)
        maxLevel.setText(str(self.maxLevel))
        # maxLevel.setAlignment(Qt.AlignRight)
        self.layout.addWidget(maxLevel, 3, 1)
        maxLevel.textChanged.connect(self.onChanged_maxLevel)
        

        self.layout.addWidget(QLabel("Update Reference"), 4, 0)
        update = QCheckBox(self)
        update.setChecked(self._update)
        self.layout.addWidget(update, 4, 1)
        update.stateChanged.connect(self.onChanged_update)

        self.layout.addWidget(QLabel("Visualize Tracking"), 5, 0)
        show = QCheckBox(self)
        show.setChecked(self._show)
        self.layout.addWidget(show, 5, 1)
        show.stateChanged.connect(self.onChanged_show)

        self.layout.addWidget(QLabel("Reinitialize Search from last data"), 6, 0)
        reinitialize = QCheckBox(self)
        reinitialize.setChecked(self._reinitialize)
        self.layout.addWidget(reinitialize, 6, 1)
        reinitialize.stateChanged.connect(self.onChanged_reinitialize)

        

        # self.buttonCheck = QPushButton("Check", self)
        # self.layout.addWidget(self.buttonCheck, 4, 0, 1, 2)
        self.setLayout(self.layout)
    def onChanged_reinitialize(self, state):
        if state == Qt.Checked:
            self._reinitialize = True
        else:
            self._reinitialize = False
        print("reinitialize: ", self._reinitialize)


    def onChanged_update(self, state):
        if state == Qt.Checked:
            self._update = True
        else:
            self._update = False
        print("update: ", self._update)

    def onChanged_show(self, state):
        if state == Qt.Checked:
            self._show = True
        else:
            self._show = False
        print("show: ", self._show)


    def onChanged_sizeWindow(self, text):
        if text != "":
            self.sizeWindow = int(text)
            print("sizeWindow: ", self.sizeWindow)
                

    def onChanged_maxLevel(self, text):
        if text != "":
            self.maxLevel = int(text)
            print("maxLevel: ", self.maxLevel)

    def onChanged_search(self, text):
        if text != "":
            self.search = int(text)
            print("search: ", self.search)

    def layout_DIC(self):
        self.layout.addWidget(QLabel("Size of the window"), 2, 0)
        sizeWindow = QLineEdit(self)
        sizeWindow.setText(str(self.sizeWindow))
        # sizeWindow.setAlignment(Qt.AlignRight)
        self.layout.addWidget(sizeWindow, 2, 1)
        sizeWindow.textChanged.connect(self.onChanged_sizeWindow)

        self.layout.addWidget(QLabel("Search Size"), 3, 0)
        search = QLineEdit(self)
        search.setText(str(self.search))
        # search.setAlignment(Qt.AlignRight)
        self.layout.addWidget(search, 3, 1)
        search.textChanged.connect(self.onChanged_search)

        self.layout.addWidget(QLabel("Update Reference"), 4, 0)
        update = QCheckBox(self)
        update.setChecked(self._update)
        self.layout.addWidget(update, 4, 1)
        update.stateChanged.connect(self.onChanged_update)

        self.layout.addWidget(QLabel("Visualize Tracking"), 5, 0)
        show = QCheckBox(self)
        show.setChecked(self._show)
        self.layout.addWidget(show, 5, 1)
        show.stateChanged.connect(self.onChanged_show)

        self.layout.addWidget(QLabel("Reinitialize Search from last data"), 6, 0)
        reinitialize = QCheckBox(self)
        reinitialize.setChecked(self._reinitialize)
        self.layout.addWidget(reinitialize, 6, 1)
        reinitialize.stateChanged.connect(self.onChanged_reinitialize)

        self.setLayout(self.layout)


class VibNodeMain_TrackTarget(QWidget):
    """
    This class represents a QWidget-based visualization tool for tracking targets.
    It provides methods to plot the current state before processing and display tracking results dynamically.
    """
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()

    def initUI(self):
        """
        Initializes the user interface, including matplotlib figures, axes, and the navigation toolbar.
        """
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')

        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')
        self.ax.axis('off')
    
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout = QGridLayout()
        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)
        self.setLayout(self.layout)

        self.comboBox = None  # QComboBox is initially not present

    def plotCurrentState(self):
        """
        Plots the current state of the tracked target.
        This method reads an image, applies undistortion if needed, and overlays tracking positions.
        """
        # Maintain the existing layout
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.axis('off')
        self.ax2.axis('off')

        self.ax.clear()

        img = cv2.imread(self.node.imagesNames[0])
        if self.node.calibResult is not None:
            img = self.node.trackTarget.undistortImage(img, self.node.calibResult[0], self.node.calibResult[1])
    
        self.ax.imshow(img)
        sizeWindow = self.node.configWidget.sizeWindow
        posTrack = np.array(self.node.trackTarget.posTrack)

        for i in range(len(posTrack)):
            self.ax.plot(posTrack[i, 0], posTrack[i, 1], 'ro')
            rect = plt.Rectangle((posTrack[i, 0]-sizeWindow//2, posTrack[i, 1]-sizeWindow//2), 
                                 sizeWindow, sizeWindow, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
        
        self.canvas.draw()

    def plotTrackingResult(self, trackingResults, ind):
        """
        Plots the tracking results for a specific target point.
        A QComboBox is dynamically added to allow switching between different tracked points.
        """
        if self.comboBox is None:
            self.comboBox = QComboBox()
            for i in range(trackingResults.shape[1]):
                self.comboBox.addItem(f"Point {i}", i)
            self.comboBox.currentIndexChanged.connect(lambda index: self.plotTrackingResult(trackingResults, index))
            
            # Set up a new layout
            self.trackingLayout = QVBoxLayout()
            self.trackingLayout.addWidget(self.canvas)
            self.trackingLayout.addWidget(self.comboBox)

            self.trackingWidget = QWidget()
            self.trackingWidget.setLayout(self.trackingLayout)
            
            self.layout.addWidget(self.trackingWidget, 1, 0)

        self.ax1.clear()
        self.ax2.clear()
        self.ax.clear()
        self.ax.set_facecolor('#666')
        self.ax.axis('off')

        self.ax1.plot(trackingResults[:, ind, 0], color='k', linewidth=2)
        self.ax2.plot(trackingResults[:, ind, 1], color='k', linewidth=2)
        self.ax1.set_xlabel('Frame')
        self.ax1.set_ylabel('u position (pixels)')
        self.ax2.set_xlabel('Frame')
        self.ax2.set_ylabel('v position (pixels)')
        
        self.canvas.draw()


    





    