from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt
from VibrationTracker.vib_conf import register_node, OP_NODE_INITIALIZETARGET
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from VibrationTracker.module.target_initialization import *

class VibInitializeTargetContent(QDMNodeContentWidget):
    def initUI(self):
        self.setStyleSheet(''' font-size: 14px; ''')

        self.layout = QGridLayout()
        self.layout.setContentsMargins(10, 20, 10, 30)

        self.layout.addWidget(QLabel(""), 0, 0)

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

@register_node(OP_NODE_INITIALIZETARGET)
class VibNode_InitializeTarget(VibNode):
    # icon = "icons/in.png" TODO
    op_code = OP_NODE_INITIALIZETARGET
    op_title = "Initialize Target"
    content_label_objname = "vib_node_initialize_target"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1,2], outputs=[1,3])        

    def initInnerClasses(self):
        self.content = VibInitializeTargetContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_InitializeTarget(self)
        self.mainWidget = VibNodeMain_InitializeTarget(self)
        self.initializeTarget = InitializeTarget()
        self.configWidget.buttonRun.clicked.connect(self.runTargetInitialization)


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
    
    def runTargetInitialization(self):
        
        # self.figure = self.mainWidget.figure
        self.mainWidget.ax.clear()

        try:
            filePath =  self.getInputs(0)[0].value
        except:
            print(Exception)

        if len(self.getInputs(1)) > 0:
            calibPath = self.getInputs(1)[0].value
            print("calibPath: ", calibPath)
        else:
            calibPath = ''
    
        self.initializeTarget.filePath = filePath
        self.initializeTarget.calibPath = calibPath
        try:
        
            self.initializeTarget.resultFolder = self.initializeTarget.createResultFolder(index=self.id)
        except:
            print("No result folder")
            return
        self.initializeTarget.closed.connect(self.on_second_window_closed)  # signal from second window
        self.initializeTarget.show()
        print("run evaluation")

    def on_second_window_closed(self):
        print("Second window closed")
        initializationResults = self.initializeTarget.readInitializationResults(self.outputName)
        posTrack = np.array(initializationResults['posTrack'])
        print("posTrack: ", posTrack)
        self.evalImplementation()

    def getResultName(self):
        return self.initializeTarget.outputName
    
    def checkCurrentState(self):
        
        self.initializeTarget.filePath =  self.getInputs(0)[0].value
        self.filePath = self.getInputs(0)[0].value

        if len(self.getInputs(1)) > 0:
            self.initializeTarget.calibPath = self.getInputs(1)[0].value
            self.calibPath = self.getInputs(1)[0].value
        else:
            self.initializeTarget.calibPath = ''
            self.calibPath = '' 

        self.imagesNames = self.initializeTarget.readImageNamesFromJson(self.initializeTarget.filePath)
        self.initializeTarget.resultFolder = self.initializeTarget.createResultFolder(index=self.id)
        self.outputName = os.path.join(self.initializeTarget.resultFolder, 'initializationResults.json')

        if os.path.isfile(self.outputName):
            self.initializeTarget.outputName = self.outputName
            initializationResults = self.initializeTarget.readInitializationResults(self.outputName)
            posTrack = np.array(initializationResults['posTrack'])
            print("posTrack: ", posTrack)
            self.mainWidget.plotImage(ind = 0, posTrack = posTrack)
            return True
        
        else: 
            return False


class VibNodeConfig_InitializeTarget(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()
        self.type = ''
        
    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of target initializaton node"), 0, 0)
        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 2, 0, 1, 2)
        self.setLayout(self.layout)


    
class VibNodeMain_InitializeTarget(QWidget):

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

    def plotImage(self, ind = 0, posTrack = None):

        self.figure.clear()
        self.figure.patch.set_facecolor('#666')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')

        
        img = cv2.imread(self.node.imagesNames[ind])

        if self.node.initializeTarget.calibPath == '':
            print("No calibration file provided")
        else:
            calibResult = self.node.initializeTarget.readCalibNameFromJson(self.node.initializeTarget.calibPath)
            img = self.node.initializeTarget.undistortImage(img, calibResult[0], calibResult[1])
        
        self.ax.imshow(img)
        for i in range(len(posTrack)):
            self.ax.plot(posTrack[i,0], posTrack[i,1], 'ro')
            self.ax.text(posTrack[i,0], posTrack[i,1], str(i), color='r', fontsize=12)
        self.canvas.draw()


