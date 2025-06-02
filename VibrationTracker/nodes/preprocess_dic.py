from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QSpacerItem, QSizePolicy, QLineEdit
from PyQt5.QtCore import Qt
from VibrationTracker.vib_conf import register_node, OP_NODE_PREPROCESSDIC
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from VibrationTracker.module.dic_preprocessing import *

class VibDicPreprocessingContent(QDMNodeContentWidget):
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
        self.preprocessDIC.stepSize = self.configWidget.stepSize

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

        self.preprocessDIC.resultFolder = self.preprocessDIC.createResultFolder(index=self.id)
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
        self.initUI()
        self.type = 'Mesh Grid for DIC'
        

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of DIC preprocessing Node"), 0, 0)

        self.layout.addWidget(QLabel("Type"), 1, 0)
        typeSelector = QComboBox(self)
        typeSelector.addItem('Mesh Grid for DIC')
        self.layout.addWidget(typeSelector, 1, 1)
        typeSelector.activated[str].connect(self.onActivated_typeSelector)

        self.layout.addWidget(QLabel("Mesh Size"), 2, 0)
        MeshSizeEdit = QLineEdit()
        self.layout.addWidget(MeshSizeEdit, 2, 1)
        MeshSizeEdit.textChanged[str].connect(self.onChanged_MeshSize)

        self.layout.addWidget(QLabel("Step Size"), 3, 0)
        StepSizeEdit = QLineEdit()
        self.layout.addWidget(StepSizeEdit, 3, 1)
        StepSizeEdit.textChanged[str].connect(self.onChanged_StepSize)

        self.buttonRun = QPushButton("Run", self)
        # tip for this button
        self.buttonRun.setToolTip("Select vertices of the ROI by right-clicking")
        
        self.layout.addWidget(self.buttonRun, 4, 0, 1, 2)
        
        self.setLayout(self.layout)


    def onActivated_typeSelector(self, text):
        print("Activated: ", text)
        self.type = text

    def onChanged_MeshSize(self, text):
        print("Mesh Size: ", text)
        if text != "":
            # check if the input is a number
            try:
                if int(text) % 2 == 0:
                    print("Please input an odd number")
                else:
                    self.meshSize = int(text)
            except:
                print("Please input an integer")

    def onChanged_StepSize(self, text):
        print("Step Size: ", text)
        if text != "":
            try:
                self.stepSize = int(text)
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

    def plotImage(self,  posTrack = None, meshSize=None):

        self.figure.clear()
        self.figure.patch.set_facecolor('#666')
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')
        print("IMAGE PATH")
        imageNames = self.node.preprocessDIC.readImageNamesFromJson(self.node.preprocessDIC.filePath)
        img = cv2.imread(imageNames[0])
        if self.node.preprocessDIC.calibPath != '':
            calibResult = self.node.preprocessDIC.readCalibNameFromJson(self.node.preprocessDIC.calibPath)
            print("Undistort image")
            img = self.node.preprocessDIC.undistortImage(img, calibResult[0], calibResult[1])
    
        self.ax.imshow(img)

        for i in range(len(posTrack)):
            self.ax.plot(posTrack[i,0], posTrack[i,1], 'ro')

            # self.ax.text(posTrack[i,0], posTrack[i,1], str(i), color='r', fontsize=12)
        self.canvas.draw()



