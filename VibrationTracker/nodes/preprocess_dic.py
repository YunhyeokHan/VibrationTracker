from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QSpacerItem, QSizePolicy, QLineEdit
from VibrationTracker.vib_conf import register_node, OP_NODE_PREPROCESSDIC
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import os
import numpy as np
import cv2
from matplotlib.patches import Rectangle
from VibrationTracker.module.dic_preprocessing import *

class VibDicPreprocessingContent(QDMNodeContentWidget):
    def initUI(self):
        self.setStyleSheet(''' font-size: 14px; ''')

        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 20, 10, 30)

        self.layout.addWidget(QLabel(""), 0, 0)
        #change the name of the box and folder
        self.tagEdit = QLineEdit(self)
        self.tagEdit.setPlaceholderText("Output name (ex: dicprep_cam1)")
        self.tagEdit.setText("dicprep")
        self.layout.addWidget(self.tagEdit, 0, 1, 1, 2)

        self.inputLabel1 = QLabel("ImageNames")
        self.layout.addWidget(self.inputLabel1, 1, 0)
        

        self.inputLabel2 = QLabel("Calibration")
        self.layout.addWidget(self.inputLabel2, 2, 0)

        spacer = QSpacerItem(80, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 1, 1, 1, 1)
 

        self.outputLabel = QLabel("posTrack")
        self.layout.addWidget(self.outputLabel, 1, 2)
        
        self.layout.addWidget(QLabel(""), 4, 2)

        self.layout.setSpacing(1)

        self.setLayout(self.layout)

    def serialize(self):
        res = super().serialize()
        res["tag"] = self.tagEdit.text()
        return res
    
    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.tagEdit.setText(data.get("tag", "dicprep"))
            return True & res
        except Exception as e:
            dumpException(e)
        return res

@register_node(OP_NODE_PREPROCESSDIC)
class VibNode_PreprocessDIC(VibNode):
    # icon = "icons/in.png" TODO
    op_code = OP_NODE_PREPROCESSDIC
    op_title = "Preprocess DIC"
    content_label_objname = "vib_node_preprocess_dic"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1,2], outputs=[1,3])        
        # self.eval()

    def initInnerClasses(self):
        self.content = VibDicPreprocessingContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_PreprocessDIC(self)
        self.mainWidget = VibNodeMain_PreprocessDIC(self)
        self.preprocessDIC = PreprocessDIC(self)
        
        self.configWidget.buttonRun.clicked.connect(self.runMesher)

    
        
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
    
    def runMesher(self):
        
        self.loadAllSetup()
        self.preprocessDIC.meshSize = self.configWidget.meshSize
        self.preprocessDIC.gridNx = self.configWidget.gridNx
        self.preprocessDIC.gridNy = self.configWidget.gridNy
        self.preprocessDIC.stepSize = self.configWidget.stepSize
        
        self.preprocessDIC.type = self.configWidget.type
        self.preprocessDIC.closed.connect(self.on_second_window_closed)  # signal from second window

        self.preprocessDIC.show()
        print("run evaluation")

    def on_second_window_closed(self):

        print("Second window closed")
        initializationResults = self.preprocessDIC.readDICPreprocessResults(self.preprocessDIC.outputName)
        posTrack = np.array(initializationResults['posTrack'])
        print("posTrack: ", posTrack)
        self.evalImplementation()

    def getResultName(self):
        return self.preprocessDIC.outputName

    def loadAllSetup(self):
        
        try:
            filePath =  self.getInputs(0)[0].value
        except:
            print(Exception)

        if len(self.getInputs(1)) > 0:
            calibPath = self.getInputs(1)[0].value
            print("calibPath: ", calibPath)
        else:
            calibPath = ''
    
        self.preprocessDIC.filePath = filePath
        self.preprocessDIC.calibPath = calibPath

        tag = self.content.tagEdit.text().strip() if hasattr(self.content, "tagEdit") else ""
        if not tag:
            tag = "dicprep"
        self.preprocessDIC.resultFolder = self.preprocessDIC.createResultFolder(index=f"{tag}_{self.id}")

        print("resultFolder: ", self.preprocessDIC.resultFolder)
        self.preprocessDIC.outputName = os.path.join(self.preprocessDIC.resultFolder, 'DICpreprocessResults.json')

    def checkCurrentState(self):

        self.loadAllSetup()
        # if jsonname is exsit
        if os.path.isfile(self.preprocessDIC.outputName):
            DICpreprocessResults = self.preprocessDIC.readDICPreprocessResults(self.preprocessDIC.outputName)
            posTrack = np.array(DICpreprocessResults['posTrack'])
            meshSize = DICpreprocessResults['meshSize']
            self.mainWidget.plotImage(posTrack = posTrack, meshSize = meshSize)
            return True
        else: 
            return False


class VibNodeConfig_PreprocessDIC(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node

        self.type = "Mesh Grid for DIC"

        # defaults 
        self.meshSize = 31  # odd
        self.gridNx = 10
        self.gridNy = 10
        self.stepSize = 31

        self.initUI()
        
    def updateWidgetsVisibility(self):
    
        isPolygon = (self.type == "Polygon ROI")
    
        # Mesh Grid mode
        self.GridNxLabel.setVisible(not isPolygon)
        self.GridNxEdit.setVisible(not isPolygon)
    
        self.GridNyLabel.setVisible(not isPolygon)
        self.GridNyEdit.setVisible(not isPolygon)
    
        # Polygon mode
        self.StepSizeLabel.setVisible(isPolygon)
        self.StepSizeEdit.setVisible(isPolygon)
        
    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of DIC preprocessing Node"), 0, 0)

        self.layout.addWidget(QLabel("Type"), 1, 0)
        typeSelector = QComboBox(self)
        typeSelector.addItem("Mesh Grid for DIC")
        typeSelector.addItem("Polygon ROI")
        self.layout.addWidget(typeSelector, 1, 1)
        typeSelector.activated[str].connect(self.onActivated_typeSelector)

        # Mesh size
        self.layout.addWidget(QLabel("Mesh Size (odd)"), 2, 0)
        self.MeshSizeEdit = QLineEdit(self)
        self.MeshSizeEdit.setText(str(self.meshSize))
        self.layout.addWidget(self.MeshSizeEdit, 2, 1)
        self.MeshSizeEdit.textChanged[str].connect(self.onChanged_MeshSize)

        # Grid NX/NY
        self.GridNxLabel = QLabel("Grid NX (columns)")
        self.layout.addWidget(self.GridNxLabel, 3, 0)
        self.GridNxEdit = QLineEdit(self)
        self.GridNxEdit.setText(str(self.gridNx))
        self.layout.addWidget(self.GridNxEdit, 3, 1)
        self.GridNxEdit.textChanged[str].connect(self.onChanged_GridNx)

        self.GridNyLabel = QLabel("Grid NY (rows)")
        self.layout.addWidget(self.GridNyLabel, 4, 0)
        self.GridNyEdit = QLineEdit(self)
        self.GridNyEdit.setText(str(self.gridNy))
        self.layout.addWidget(self.GridNyEdit, 4, 1)
        self.GridNyEdit.textChanged[str].connect(self.onChanged_GridNy)
        # Step size (Polygon ROI mode)

        self.StepSizeLabel = QLabel("Step Size")
        self.layout.addWidget(self.StepSizeLabel, 5, 0)
        
        self.StepSizeEdit = QLineEdit(self)
        self.StepSizeEdit.setText(str(self.stepSize))
        
        self.layout.addWidget(self.StepSizeEdit, 5, 1)
        
        self.StepSizeEdit.textChanged[str].connect(
            self.onChanged_StepSize
)

        # Run button (single)
        self.buttonRun = QPushButton("Run", self)
        self.buttonRun.setToolTip("Select vertices of the ROI by right-clicking")
        self.layout.addWidget(self.buttonRun, 6, 0, 1, 2)
        self.updateWidgetsVisibility()
        self.setLayout(self.layout)

        
    def onActivated_typeSelector(self, text):
        self.type = text
        self.updateWidgetsVisibility()

    def onChanged_MeshSize(self, text):
        if text != "":
            try:
                v = int(text)
                if v % 2 == 0 or v < 1:
                    print("Please input a positive odd number")
                    return
                self.meshSize = v
            except:
                print("Please input an integer")

    def onChanged_GridNx(self, text):
        if text != "":
            try:
                v = int(text)
                if v < 1:
                    print("NX must be >= 1")
                    return
                self.gridNx = v
            except:
                print("Please input an integer")

    def onChanged_GridNy(self, text):
        if text != "":
            try:
                v = int(text)
                if v < 1:
                    print("NY must be >= 1")
                    return
                self.gridNy = v
            except:
                print("Please input an integer")
                
    def onChanged_StepSize(self, text):
    
        if text != "":
    
            try:
    
                value = int(text)
    
                if value < 1:
                    return
    
                self.stepSize = value
    
            except:
                print("Please input an integer")

class VibNodeMain_PreprocessDIC(QWidget):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()

    def initUI(self):
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')

        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.axis('off')
        self.ax.set_facecolor('#666')

        self.toolbar = NavigationToolbar(self.canvas, self)
        
        self.layout = QGridLayout()
        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)
        self.setLayout(self.layout)
        # self.show()

    def plotImage(self, posTrack=None, meshSize=None):
    
        self.figure.clear()
        self.figure.patch.set_facecolor('#666')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')
        self.ax.axis('off')
    
        imageNames = self.node.preprocessDIC.readImageNamesFromJson(self.node.preprocessDIC.filePath)
        img = cv2.imread(imageNames[0])
    
        if self.node.preprocessDIC.calibPath != '':
            calibResult = self.node.preprocessDIC.readCalibNameFromJson(self.node.preprocessDIC.calibPath)
            img = self.node.preprocessDIC.undistortImage(img, calibResult[0], calibResult[1])
    
        # BGR -> RGB (difference between cv2 and matoplotlib)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.imshow(img)
    
        # draw points + subset
        half = (int(meshSize) - 1) / 2.0
        for (x, y) in posTrack:
            self.ax.plot(x, y, 'ro', markersize=3)
            r = Rectangle((x - half, y - half), 2*half, 2*half,
                          fill=False, edgecolor="yellow", linewidth=1.2)
            self.ax.add_patch(r)
    
        self.canvas.draw()


