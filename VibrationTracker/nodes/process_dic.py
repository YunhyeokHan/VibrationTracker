from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMessageBox
from PyQt5.QtCore import Qt
from VibrationTracker.vib_conf import register_node, OP_NODE_PROCESSDIC
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import cv2

from VibrationTracker.module.dic_processing import *


class VibProcessDICContent(QDMNodeContentWidget):
    def initUI(self):
        self.setStyleSheet(''' font-size: 14px; ''')

        self.layout = QGridLayout()
        self.layout.setContentsMargins(10,12,10,15)
        self.setupUI_DIC2D()
        self.setLayout(self.layout)

    def setupUI_DIC2D(self):
        self.clearLayout()

        self.inputLabel1 = QLabel("ImageNames")
        self.layout.addWidget(self.inputLabel1, 1, 0)
        self.inputLabel2 = QLabel("Calibration")
        self.layout.addWidget(self.inputLabel2, 2, 0)
        self.inputLabel3 = QLabel("PosTrack")
        self.layout.addWidget(self.inputLabel3, 3, 0)
        self.inputLabel4 = QLabel("")
        self.layout.addWidget(self.inputLabel4, 4, 0)
        self.inputLabel5 = QLabel("")
        self.layout.addWidget(self.inputLabel5, 5, 0)
        self.layout.addWidget(QLabel(""), 6, 0)

        spacer = QSpacerItem(10, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 3, 1, 1, 1)
        self.outputlabel = QLabel("TrackResults")
        self.layout.addWidget(self.outputlabel, 3, 2)

    def setupUI_DIC3D(self):
        self.clearLayout()

        self.inputLabel1 = QLabel("ImageNames")
        self.layout.addWidget(self.inputLabel1, 1, 0)
        self.inputLabel2 = QLabel("Calibration")
        self.layout.addWidget(self.inputLabel2, 2, 0)
        self.inputLabel3 = QLabel("PosTrack")
        self.layout.addWidget(self.inputLabel3, 3, 0)
        self.inputLabel4 = QLabel("ImageNames_cam2")
        self.layout.addWidget(self.inputLabel4, 4, 0)
        self.inputLabel5 = QLabel("Calibration_cam2")
        self.layout.addWidget(self.inputLabel5, 5, 0)
        self.layout.addWidget(QLabel(""), 6, 0)

        spacer = QSpacerItem(10, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 3, 1, 1, 1)
        self.outputlabel = QLabel("TrackResults")
        self.layout.addWidget(self.outputlabel, 3, 2)

    def clearLayout(self):
        for i in reversed(range(self.layout.count())): 
            item = self.layout.itemAt(i)
            if item.widget() is not None:  
                item.widget().setParent(None)

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

@register_node(OP_NODE_PROCESSDIC)
class VibNode_ProcessDIC(VibNode):
    # icon = "icons/in.png" TODO
    op_code = OP_NODE_PROCESSDIC
    op_title = "Process DIC"
    content_label_objname = "vib_node_process_dic"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1,2,3,1,2], outputs=[1])        
        
    def initInnerClasses(self):
        self.content = VibProcessDICContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_ProcessDIC(self)
        self.mainWidget = VibNodeMain_ProcessDIC(self)

        self.processDIC = ProcessDIC()
        
        self.configWidget.buttonRun.clicked.connect(self.runDIC)
        self.configWidget.buttonCheck.clicked.connect(self.setupDIC)


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
        
    def setupDIC(self):
        self.setupNode()
        self.mainWidget.plotCurrentState()
        self.markInvalid()

    def setupNode(self):

        self.processDIC.filePath = self.getInputs(0)[0].value
        print("filePath: ", self.processDIC.filePath)
        if len(self.getInputs(1)) > 0:
            self.processDIC.calibPath = self.getInputs(1)[0].value
            print("calibPath: ", self.processDIC.calibPath)
            self.calibResult = self.processDIC.readCalibNameFromJson(self.processDIC.calibPath)
        else:
            self.processDIC.calibPath = None
            self.calibResult = None
        
        self.imagesNames = self.processDIC.readImageNamesFromJson(self.processDIC.filePath)
        self.resultFolder = self.processDIC.createResultFolder(index=self.id)
        
        self.processDIC.posTrackPath = self.getInputs(2)[0].value
        self.processDIC.posTrack, self.processDIC.meshSize = self.processDIC.readDICPreprocessResults(self.processDIC.posTrackPath)
    
    def runDIC(self):
        self.setupDIC()

        if self.configWidget.method == "DIC2D with ZNSSD":
            # try: 
            self.processDIC.trackTarget_DIC(self.imagesNames, self.processDIC.posTrack, self.resultFolder, winsize = self.processDIC.meshSize, search = self.configWidget.search, calibResult = self.calibResult, update = self.configWidget._update, show = self.configWidget._show)
            # except Exception as e:
            #     print("Error: ", e)
            #     if type(e) == ValueError:
            #         QMessageBox.critical(None, "Error", f"Size of searching area is too small to detect large displacement {e}")
            
        if self.configWidget.method == "DIC2D with MultiProcessing":
            # try:
            self.processDIC.trackTarget_DICMP(self.imagesNames, self.processDIC.posTrack, self.resultFolder, meshsize = self.processDIC.meshSize, searchSize = self.configWidget.search, calibResult = self.calibResult, numProcess = self.configWidget._numProcess, show = self.configWidget._show, update = self.configWidget._update)
            # except Exception as e:
            #     print("Error: ", e)
            #     if type(e) == ValueError:
            #         QMessageBox.critical(None, "Error", f"Size of searching area is too small to detect large displacement {e}")
            
            # self.trackTarget.plotTrackingResult(TrackResults)

        if self.configWidget.method == "DIC3D with MultiProcessing":
            self.imagesNames_cam2 = self.processDIC.readImageNamesFromJson(self.getInputs(3)[0].value)

            self.calibResult_cam2 = self.processDIC.readCalibNameFromJson(self.getInputs(4)[0].value)
            
            self.processDIC.trackTarget_DICMP3D(imageNames1=self.imagesNames, imageNames2=self.imagesNames_cam2, posTrack=self.processDIC.posTrack, resultFolderPath=self.resultFolder, meshsize=self.processDIC.meshSize, searchSize=self.configWidget.search, calibResult1=self.calibResult, calibResult2=self.calibResult_cam2,searchSize_twoimage = self.configWidget.search2, numProcess=self.configWidget._numProcess, show=self.configWidget._show, update=self.configWidget._update)

        self.eval()


    def getResultName(self):
        return self.processDIC.outputName

    def checkCurrentState(self):
    
        self.processDIC.filePath = self.getInputs(0)[0].value
        print("filePath: ", self.processDIC.filePath)
        if len(self.getInputs(1)) > 0:
            self.processDIC.calibPath = self.getInputs(1)[0].value
            print("calibPath: ", self.processDIC.calibPath)
            self.calibResult = self.processDIC.readCalibNameFromJson(self.processDIC.calibPath)
        else:
            self.processDIC.calibPath = None
            self.calibResult = None
        
        self.imagesNames = self.processDIC.readImageNamesFromJson(self.processDIC.filePath)
        self.resultFolder = self.processDIC.createResultFolder(index=self.id)
        
        self.processDIC.posTrackPath = self.getInputs(2)[0].value
        self.processDIC.posTrack, self.processDIC.meshSize = self.processDIC.readDICPreprocessResults(self.processDIC.posTrackPath)

        
        # find the number of jsonfiles  
        jsonPath_all = self.processDIC.readResultsNames(self.resultFolder)
        num_results = len(jsonPath_all)
        num_images = len(self.imagesNames)
        print("num_results: ", num_results)
        print("num_images: ", num_images)
        

        # check if the output file exists
        if num_results == num_images:
            self.processDIC.outputName = self.resultFolder
            self.mainWidget.plotCurrentState()
            # plot the tracking result
            return True
        else:
            return False

class VibNodeConfig_ProcessDIC(QWidget):
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
        self._numProcess = 1
        self.search2 = 150

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of DIC Process Node"), 0, 0)

        ## 
        self.layout.addWidget(QLabel("Method"), 1, 0)
        methodSelector = QComboBox(self)
        methodSelector.addItem('')
        methodSelector.addItem('DIC2D with ZNSSD')
        methodSelector.addItem('DIC2D with MultiProcessing')
        methodSelector.addItem('DIC3D with MultiProcessing')

        self.layout.addWidget(methodSelector, 1, 1)
        methodSelector.activated[str].connect(self.onActivated_methodSelector)

        self.layout.addWidget(QLabel("."), 2, 0)
        self.layout.addWidget(QLabel("."), 3, 0)
        self.layout.addWidget(QLabel("."), 4, 0)
        self.layout.addWidget(QLabel("."), 5, 0)
        self.layout.addWidget(QLabel("."), 6, 0)
        self.layout.addWidget(QLabel("."), 7, 0)


        self.buttonCheck = QPushButton("Check", self)
        self.layout.addWidget(self.buttonCheck, 8, 0, 1, 1)

        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 8, 1, 1, 1)


        self.setLayout(self.layout)

    def onActivated_methodSelector(self, text):
        print("Activated: ", text)
        self.method = text

        print("method: ", self.method)

        if text == "DIC2D with ZNSSD":
            self.layout_DIC()
            self.node.content.setupUI_DIC2D()
        elif text == "DIC2D with MultiProcessing":
            self.layout_DIC_MP()
            self.node.content.setupUI_DIC2D()
        elif text == "DIC3D with MultiProcessing":
            self.layout_DIC_MP()
            self.node.content.setupUI_DIC3D()
        

    def layout_DIC_MP(self):
        self.node.setupNode()
        self.layout.addWidget(QLabel("Size of the window"), 2, 0)
        sizeWindow = QLineEdit(self)
        if hasattr(self.node.processDIC, 'meshSize'):
            self.sizeWindow = self.node.processDIC.meshSize
        sizeWindow.setText(str(self.sizeWindow))
        # sizeWindow.setAlignment(Qt.AlignRight)
        self.layout.addWidget(sizeWindow, 2, 1)
        # disable the sizeWindow
        sizeWindow.setDisabled(True)        
        # sizeWindow.textChanged.connect(self.onChanged_sizeWindow)

        self.layout.addWidget(QLabel("Search Size"), 3, 0)
        search = QLineEdit(self)
        search.setText(str(self.search))
        # search.setAlignment(Qt.AlignRight)
        self.layout.addWidget(search, 3, 1)
        search.textChanged.connect(self.onChanged_search)


        self.layout.addWidget(QLabel("Search Size between camera"), 4, 0)
        search2 = QLineEdit(self)
        search2.setText(str(self.search2))
        if self.method == "DIC3D with MultiProcessing":
            search2.setDisabled(False)
        else:
            search2.setDisabled(True)   
        self.layout.addWidget(search2, 4, 1)
        search2.textChanged.connect(self.onChanged_search2)

        self.layout.addWidget(QLabel("Visualize Tracking"), 5, 0)
        show = QCheckBox(self)
        # self._show = False
        show.setChecked(self._show)
        # show.setDisabled(True)
        self.layout.addWidget(show, 5, 1)
        show.stateChanged.connect(self.onChanged_show)

        self.layout.addWidget(QLabel("Reinitialize Search from last data"), 6, 0)
        reinitialize = QCheckBox(self)
        self._reinitialize = False
        reinitialize.setChecked(self._reinitialize)
        reinitialize.setDisabled(True)
        self.layout.addWidget(reinitialize, 6, 1)

        reinitialize.stateChanged.connect(self.onChanged_reinitialize)

        self.layout.addWidget(QLabel("Number of Process"), 7, 0)
        tooltip = "Number of process to run the DIC tracking, Maximum number of process is " + str(os.cpu_count())
        

        
        numProcess = QLineEdit(self)
        
        numProcess.setToolTip(tooltip)
        numProcess.setText(str(self._numProcess))

        self.layout.addWidget(numProcess, 7, 1)
        numProcess.textChanged.connect(self.onChanged_numProcess)

        self.setLayout(self.layout)


        
    def onChanged_numProcess(self, text):
        if text != "":
            try:
                self._numProcess = int(text)
                print("numProcess: ", self._numProcess)

            except:
                print("Please enter a number")
                


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

    # def onChanged_sizeWindow(self, text):
    #     if text != "":
    #         self.sizeWindow = int(text)
    #         print("sizeWindow: ", self.sizeWindow)
                

    def onChanged_search(self, text):
        try:
            if text != "":
                self.search = int(text)
                print("search: ", self.search)
        except:
            print("Please enter a number")

    def onChanged_search2(self, text):
        try: 
            if text != "":
                self.search2 = int(text)
                print("search2: ", self.search2)
        except:
            print("Please enter a number")

    def layout_DIC(self):
        self.node.setupNode()
        self.layout.addWidget(QLabel("Size of the window"), 2, 0)
        sizeWindow = QLineEdit(self)
        if hasattr(self.node.processDIC, 'meshSize'):
            self.sizeWindow = self.node.processDIC.meshSize
        sizeWindow.setText(str(self.sizeWindow))
        sizeWindow.setDisabled(True)
        # sizeWindow.setAlignment(Qt.AlignRight)
        self.layout.addWidget(sizeWindow, 2, 1)
        # sizeWindow.textChanged.connect(self.onChanged_sizeWindow)

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


        self.layout.addWidget(QLabel("Number of Process"), 7, 0)
        numProcess = QLineEdit(self)
        numProcess.setText(str(self._numProcess))
        numProcess.setDisabled(True)
        self.layout.addWidget(numProcess, 7, 1)
        numProcess.textChanged.connect(self.onChanged_numProcess)

        self.setLayout(self.layout)

class VibNodeMain_ProcessDIC(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()

    def initUI(self, version = 1):
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')

        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(211)
        self.ax2 = self.figure.add_subplot(212)
        # self.ax1.set_facecolor('#666')
        # # self.ax1.axis('off')
        # self.ax2.set_facecolor('#666')
        # # self.ax2.axis('off')

        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#666')
        self.ax.axis('off')
    
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.layout = QGridLayout()

        self.layout.addWidget(self.toolbar, 0, 0)
        self.layout.addWidget(self.canvas, 1, 0)
        
        self.setLayout(self.layout)


    def plotCurrentState(self):

        self.ax1.clear()
        self.ax2.clear()
        # self.ax1.set_facecolor('#666')
        # self.ax2.set_facecolor('#666')
        self.ax1.axis('off')
        self.ax2.axis('off')

        self.ax.clear()

        img = cv2.imread(self.node.imagesNames[0])
        if self.node.calibResult is not None:
            img = self.node.processDIC.undistortImage(img, self.node.calibResult[0], self.node.calibResult[1])
    
        self.ax.imshow(img)

        sizeWindow = self.node.configWidget.sizeWindow

        posTrack = np.array(self.node.processDIC.posTrack)

        # for i in range(len(posTrack)):
        #     self.ax.plot(posTrack[i, 0], posTrack[i, 1], 'ro')
        #     # rect = plt.Rectangle((posTrack[i, 0]-sizeWindow//2, posTrack[i, 1]-sizeWindow//2), sizeWindow, sizeWindow, edgecolor='r', facecolor='none')
        #     # self.ax.add_patch(rect)
        self.canvas.draw()

    def plotTrackingResult(self, trackingResults, ind):

        self.ax1.clear()
        self.ax2.clear()
        self.ax.clear()
        self.ax.set_facecolor('#666')
        self.ax.axis('off')

        self.ax1.plot(trackingResults[:, ind, 0], color = 'k', linewidth = 2)
        self.ax2.plot(trackingResults[:, ind, 1], color = 'k', linewidth = 2)
        self.ax1.set_xlabel('Frame')
        self.ax1.set_ylabel("u position (pixels)")
        
        self.ax2.set_xlabel('Frame')
        self.ax2.set_ylabel("v position (pixels)")
        
        self.canvas.draw()


    





    