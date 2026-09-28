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
import numpy as np
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
        try:
            res["tag"] = self.node.configWidget.tagEdit.text()
        except:
            res["tag"] = "dic"
        return res
    
    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            tag = data.get("tag", "dic")
            if hasattr(self.node, "configWidget") and hasattr(self.node.configWidget, "tagEdit"):
                self.node.configWidget.tagEdit.setText(tag)
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
        self._mesh_user_override = False

        
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
        self.markDirty(False)
        self.markInvalid(False)
        
        self.evalImplementation()
        self.evalChildren()

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
        #self.resultFolder = self.processDIC.createResultFolder(index=self.id)
        tag = self.configWidget.tagEdit.text().strip() if hasattr(self.configWidget, "tagEdit") else ""
        if not tag:
            tag = "dic"
        self.resultFolder = self.processDIC.createResultFolder(index=f"{tag}_{self.id}")
    # --- PosTrack input (required) ---
        if len(self.getInputs(2)) == 0 or self.getInputs(2)[0].value is None:
            self.processDIC.posTrackPath = None
            return
        
        self.processDIC.posTrackPath = self.getInputs(2)[0].value
        
        # meshSize: file default + UI override
        self.processDIC.posTrack, mesh_file = self.processDIC.readDICPreprocessResults(self.processDIC.posTrackPath)
        
        # file default
        self.processDIC.meshSize = mesh_file
        
        # push file value into UI unless user already overridden it
        if not getattr(self.configWidget, "_mesh_user_override", False):
            self.configWidget.sizeWindow = int(mesh_file)
            if hasattr(self.configWidget, "sizeWindowEdit"):
                self.configWidget.sizeWindowEdit.setText(str(mesh_file))
        
        # UI override (if user changed)
        self.processDIC.meshSize = int(self.configWidget.sizeWindow)


    def runDIC(self):
        self.setupDIC()

        if self.configWidget.method == "DIC2D with single core":
            # try: 
            self.processDIC.interpType = self.configWidget.interpType

            self.processDIC.trackTarget_DIC(self.imagesNames, self.processDIC.posTrack, self.resultFolder, winsize = self.processDIC.meshSize, search = self.configWidget.search, calibResult = self.calibResult, update = self.configWidget._update, show = self.configWidget._show)
            # except Exception as e:
            #     print("Error: ", e)
            #     if type(e) == ValueError:
            #         QMessageBox.critical(None, "Error", f"Size of searching area is too small to detect large displacement {e}")
            
        if self.configWidget.method == "DIC2D with MultiProcessing":
            # try:
            self.processDIC.interpType = self.configWidget.interpType

            self.processDIC.trackTarget_DICMP(self.imagesNames, self.processDIC.posTrack, self.resultFolder, meshsize = self.processDIC.meshSize, searchSize = self.configWidget.search, calibResult = self.calibResult, numProcess = self.configWidget._numProcess, show = self.configWidget._show, update = self.configWidget._update)
            # except Exception as e:
            #     print("Error: ", e)
            #     if type(e) == ValueError:
            #         QMessageBox.critical(None, "Error", f"Size of searching area is too small to detect large displacement {e}")
            
            # self.trackTarget.plotTrackingResult(TrackResults)

        if self.configWidget.method == "DIC3D with MultiProcessing":
        
            self.processDIC.interpType = self.configWidget.interpType
        
            inputs3 = self.getInputs(3)
            inputs4 = self.getInputs(4)
        
            if len(inputs3) == 0 or inputs3[0].value is None:
                QMessageBox.warning(
                    None,
                    "Missing Input",
                    "Input 3 (ImageNames_cam2) is not connected."
                )
                return
        
            if len(inputs4) == 0 or inputs4[0].value is None:
                QMessageBox.warning(
                    None,
                    "Missing Input",
                    "Input 4 (Calibration_cam2) is not connected."
                )
                return
        
            self.imagesNames_cam2 = self.processDIC.readImageNamesFromJson(
                inputs3[0].value
            )
        
            self.calibResult_cam2 = self.processDIC.readCalibNameFromJson(
                inputs4[0].value
            )
        
            self.processDIC.trackTarget_DICMP3D(
                imageNames1=self.imagesNames,
                imageNames2=self.imagesNames_cam2,
                posTrack=self.processDIC.posTrack,
                resultFolderPath=self.resultFolder,
                meshsize=self.processDIC.meshSize,
                searchSize=self.configWidget.search,
                calibResult1=self.calibResult,
                calibResult2=self.calibResult_cam2,
                searchSize_twoimage=self.configWidget.search2,
                numProcess=self.configWidget._numProcess,
                show=self.configWidget._show,
                update=self.configWidget._update
            )
            # mettre à jour la sortie du node
        self.evalImplementation()
        
        # propager aux nodes connectés
        self.evalChildren()
        
        print("PROCESS VALUE =", self.value)
        print("PROCESS RESULT FOLDER =", self.resultFolder)

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
        tag = self.configWidget.tagEdit.text().strip() if hasattr(self.configWidget, "tagEdit") else ""
        if not tag:
            tag = "dic"
        self.resultFolder = self.processDIC.createResultFolder(index=f"{tag}_{self.id}")

        
        self.processDIC.posTrackPath = self.getInputs(2)[0].value
        self.processDIC.posTrack, self.processDIC.meshSize = self.processDIC.readDICPreprocessResults(self.processDIC.posTrackPath)
        # override meshSize with UI value (user editable)
        if hasattr(self.configWidget, "sizeWindow"):
            self.processDIC.meshSize = int(self.configWidget.sizeWindow)

        
        # find the number of jsonfiles  
        jsonPath_all = self.processDIC.readResultsNames(self.resultFolder)
        num_results = len(jsonPath_all)
        num_images = len(self.imagesNames)
        print("num_results: ", num_results)
        print("num_images: ", num_images)
        print("resultFolder =", self.resultFolder)       

        # check if the output file exists
        if num_results > 0:
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

        # ---- defaults before initUI ----
        self.method = ""
        self.sizeWindow = 21
        self.maxLevel = 1
        self._update = False
        self._show = False
        self.search = 30
        self._reinitialize = True
        self._numProcess = 1
        self.search2 = 150
        self.interpType = "Bicubic"   # <- default

        self.initUI()

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of DIC Process Node"), 0, 0, 1, 2)

        # Output name
        self.layout.addWidget(QLabel("Output name"), 1, 0)
        self.tagEdit = QLineEdit(self)
        self.tagEdit.setPlaceholderText("ex: dic_cam1")
        self.tagEdit.setText("dic")
        self.layout.addWidget(self.tagEdit, 1, 1)

        # Method
        self.layout.addWidget(QLabel("Method"), 2, 0)
        self.methodSelector = QComboBox(self)
        self.methodSelector.addItems([
            "",
            "DIC2D with single core",
            "DIC2D with MultiProcessing",
            "DIC3D with MultiProcessing"
        ])
        self.layout.addWidget(self.methodSelector, 2, 1)
        self.methodSelector.activated[str].connect(self.onActivated_methodSelector)

        # Interpolation (dropdown)
        self.layout.addWidget(QLabel("Interpolation"), 3, 0)
        self.interpCombo = QComboBox(self)
        self.interpCombo.addItems(["Bilinear", "Bicubic"])
        self.interpCombo.setCurrentText(self.interpType)
        self.layout.addWidget(self.interpCombo, 3, 1)
        self.interpCombo.currentTextChanged.connect(self.onChanged_interpType)
        self.buttonCheck = QPushButton("Check", self)
        self.layout.addWidget(self.buttonCheck, 9, 0, 1, 1)

        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 9, 1, 1, 1)

        self.setLayout(self.layout)

    def onChanged_interpType(self, text):
        self.interpType = text
        print("interpType:", self.interpType)

    def clearDynamic(self, row_start=4, row_end=20):
        for i in reversed(range(self.layout.count())):
            item = self.layout.itemAt(i)
            w = item.widget()
            if w is None:
                continue
    
            r, c, rs, cs = self.layout.getItemPosition(i)
    
            if 4 <= r < 9:
                w.setParent(None)

    def onActivated_methodSelector(self, text):
        print("Activated: ", text)
        self.method = text
        print("method: ", self.method)
    
        # reset menu (garde juste le dropdown)
        if text == "":
            self.clearDynamic(4, 15)
            return
        
        self.clearDynamic(4, 20)

    
        if text == "DIC2D with single core":
        
            self.layout_DIC()
            self.node.content.setupUI_DIC2D()
        
        elif text == "DIC2D with MultiProcessing":
        
            self.layout_DIC_MP_2D()
            self.node.content.setupUI_DIC2D()
        
        elif text == "DIC3D with MultiProcessing":
        
            self.layout_DIC_MP_3D()
            self.node.content.setupUI_DIC3D()


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

    def onChanged_sizeWindow(self, text):
        try:
            if text != "":
                self.sizeWindow = int(text)
                self._mesh_user_override = True
        except:
            pass

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
            
#Layout for single core
    def layout_DIC(self):
    
        self.clearDynamic()
    
        # Size Window
        self.layout.addWidget(QLabel("Size of the window"), 4, 0)
    
        self.sizeWindowEdit = QLineEdit(self)
    
        if hasattr(self.node.processDIC, 'meshSize'):
            self.sizeWindow = int(self.node.processDIC.meshSize)
    
        self.sizeWindowEdit.setText(str(self.sizeWindow))
    
        self.layout.addWidget(self.sizeWindowEdit, 4, 1)
    
        self.sizeWindowEdit.textChanged.connect(self.onChanged_sizeWindow)
    
        # Search
        self.layout.addWidget(QLabel("Search Size"), 5, 0)
    
        search = QLineEdit(self)
        search.setText(str(self.search))
    
        self.layout.addWidget(search, 5, 1)
    
        search.textChanged.connect(self.onChanged_search)
    
        # Update
        self.layout.addWidget(QLabel("Update Reference"), 6, 0)
    
        update = QCheckBox(self)
        update.setChecked(self._update)
    
        self.layout.addWidget(update, 6, 1)
    
        update.stateChanged.connect(self.onChanged_update)
    
        # Show
        self.layout.addWidget(QLabel("Visualize Tracking"), 7, 0)
    
        show = QCheckBox(self)
        show.setChecked(self._show)
    
        self.layout.addWidget(show, 7, 1)
    
        show.stateChanged.connect(self.onChanged_show)
    
        # Reinitialize
        self.layout.addWidget(QLabel("Reinitialize Search from last data"), 8, 0)
    
        reinitialize = QCheckBox(self)
    
        reinitialize.setChecked(self._reinitialize)
    
        self.layout.addWidget(reinitialize, 8, 1)
    
        reinitialize.stateChanged.connect(self.onChanged_reinitialize)

#Layout for DIC 2D multiprocess
    def layout_DIC_MP_2D(self):   
        self.layout.addWidget(QLabel("Size of the window"), 4, 0)
    
        self.sizeWindowEdit = QLineEdit(self)   
        if hasattr(self.node.processDIC, 'meshSize'):
            self.sizeWindow = int(self.node.processDIC.meshSize)
    
        self.sizeWindowEdit.setText(str(self.sizeWindow))
        self.layout.addWidget(self.sizeWindowEdit, 4, 1)
    
        self.sizeWindowEdit.textChanged.connect(self.onChanged_sizeWindow)
    
        # Search
        self.layout.addWidget(QLabel("Search Size"), 5, 0)   
        search = QLineEdit(self)
        search.setText(str(self.search))
        self.layout.addWidget(search, 5, 1)
        search.textChanged.connect(self.onChanged_search)
        # Show
        self.layout.addWidget(QLabel("Visualize Tracking"), 6, 0)
        show = QCheckBox(self)
        show.setChecked(self._show)
    
        self.layout.addWidget(show, 6, 1)
        show.stateChanged.connect(self.onChanged_show)  
        # Num Process
        self.layout.addWidget(QLabel("Number of Process"), 7, 0)
        numProcess = QLineEdit(self)
        numProcess.setText(
            str(self._numProcess))
        self.layout.addWidget(numProcess,7,1)
        numProcess.textChanged.connect(self.onChanged_numProcess)
   
#Layout for DIC 3D
    def layout_DIC_MP_3D(self):
        
        self.layout.addWidget(QLabel("Size of the window"), 4, 0)
    
        self.sizeWindowEdit = QLineEdit(self)
    
        if hasattr(self.node.processDIC, 'meshSize'):
            self.sizeWindow = int(self.node.processDIC.meshSize)
    
        self.sizeWindowEdit.setText(str(self.sizeWindow))
    
        self.layout.addWidget(
            self.sizeWindowEdit,4,1)
    
        self.sizeWindowEdit.textChanged.connect(self.onChanged_sizeWindow)
    
        # Search cam1
        self.layout.addWidget(QLabel("Search Size"),5,0)
        search = QLineEdit(self)
        search.setText(str(self.search))
    
        self.layout.addWidget( search, 5, 1)
    
        search.textChanged.connect(self.onChanged_search)
    
        # Search cam2
        self.layout.addWidget(
            QLabel("Search Size between camera"),6,0)  
        search2 = QLineEdit(self) 
        search2.setText(str(self.search2))
        self.layout.addWidget(search2,6,1)
        search2.textChanged.connect(self.onChanged_search2)
    
        # Show
        self.layout.addWidget(QLabel("Visualize Tracking"),7,0)
        show = QCheckBox(self)
        show.setChecked(self._show)
        self.layout.addWidget(show,7,1)
        show.stateChanged.connect(self.onChanged_show)
    
        # NumProcess
        self.layout.addWidget(QLabel("Number of Process"),8,0) 
        numProcess = QLineEdit(self)
        numProcess.setText(str(self._numProcess))
        self.layout.addWidget(numProcess, 8,1)
        numProcess.textChanged.connect(self.onChanged_numProcess)
        
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

        for i in range(len(posTrack)):
             self.ax.plot(posTrack[i, 0], posTrack[i, 1], 'ro')
             rect = plt.Rectangle((posTrack[i, 0]-sizeWindow//2, posTrack[i, 1]-sizeWindow//2), sizeWindow, sizeWindow, edgecolor='r', facecolor='none')
             self.ax.add_patch(rect)
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


    





    