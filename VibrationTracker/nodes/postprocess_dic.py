from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMenu, QSlider, QSpinBox
from PyQt5.QtCore import Qt
from VibrationTracker.vib_conf import register_node, OP_NODE_POSTPROCESSDIC
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
from matplotlib import cm
from VibrationTracker.module.dic_postprocessing import *
from PyQt5.Qt import QCursor
import matplotlib.tri as tri
from matplotlib.patches import Rectangle
from PyQt5.QtWidgets import QHBoxLayout, QDoubleSpinBox



class VibPostprocessDICContent(QDMNodeContentWidget):
    def initUI(self):
        self.setStyleSheet(''' font-size: 14px; ''')
    
        self.layout = QGridLayout()
        self.layout.setContentsMargins(10,10,10,30)
    
        self.inputLabel1 = QLabel("ImageNames")
        self.layout.addWidget(self.inputLabel1, 1, 0)
    
        self.inputLabel2 = QLabel("Calibration")
        self.layout.addWidget(self.inputLabel2, 2, 0)
    
        self.inputLabel3 = QLabel("TrackResults")
        self.layout.addWidget(self.inputLabel3, 3, 0)
    
        self.inputLabel4 = QLabel("EstimatePose")
        self.layout.addWidget(self.inputLabel4, 4, 0)
    
        self.inputLabel5 = QLabel("EstimatePose2")
        self.layout.addWidget(self.inputLabel5, 5, 0)
    
        spacer = QSpacerItem(80, 0, QSizePolicy.Minimum, QSizePolicy.Minimum)
        self.layout.addItem(spacer, 3, 1, 1, 1)
    
        self.outputlabel = QLabel("Output")
        self.layout.addWidget(self.outputlabel, 3, 2)
    
        self.layout.addWidget(QLabel(""), 4, 2)
    
        self.setLayout(self.layout)
    
    def serialize(self):
        res = super().serialize()
        res["resultFolder"] = getattr(self.node, "resultFolder", "")
        return res
    
    def deserialize(self, data, hashmap={}):
        res = super().deserialize(data, hashmap)
        try:
            self.node.resultFolder = data.get("resultFolder", "")
            return True & res
        except Exception as e:
            dumpException(e)
        return res


@register_node(OP_NODE_POSTPROCESSDIC)
class VibNode_PostprocessDIC(VibNode):
    # icon = "icons/in.png" TODO
    op_code = OP_NODE_POSTPROCESSDIC
    op_title = "Postprocess DIC"
    content_label_objname = "vib_node_postprocess_dic"

    def __init__(self, scene):
        super().__init__(scene, inputs=[1,2,3,4,4], outputs=[1])        

    def initInnerClasses(self):
        self.content = VibPostprocessDICContent(self)
        self.grNode = VibGraphicsNode(self)
        self.configWidget = VibNodeConfig_PostprocessDIC(self)
        self.mainWidget = VibNodeMain_PostprocessDIC(self)
        self.postprocessDIC = PostprocessDIC()
        
        self.configWidget.buttonRun.clicked.connect(self.runPostDIC)
        # self.configWidget.buttonCheck.clicked.connect(self.setupDIC)
                
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
        
    def setupPostDIC(self):
        if len(self.getInputs(0)) == 0:
            return False
        file_input = self.getInputs(0)[0]

        if file_input.value is None:
            return False

        self.postprocessDIC.filePath = self.getInputs(0)[0].value
        print("filePath: ", self.postprocessDIC.filePath)
    
        if len(self.getInputs(1)) > 0:
            self.postprocessDIC.calibPath = self.getInputs(1)[0].value
            print("calibPath: ", self.postprocessDIC.calibPath)
            self.calibResult = self.postprocessDIC.readCalibNameFromJson(
                self.postprocessDIC.calibPath
            )
        else:
            self.postprocessDIC.calibPath = None
            self.calibResult = None
    
        self.imagesNames = self.postprocessDIC.readImageNamesFromJson(
            self.postprocessDIC.filePath
        )
        inputs2 = self.getInputs(2)
        
        if len(inputs2) == 0:
            print("No ProcessDIC connected")
            return False
        
        if inputs2[0].value is None:
            print("ProcessDIC output is None")
            return False
        
        self.postprocessDIC.DICResultsPath = inputs2[0].value
        
        print("POSTPROCESS INPUT =", self.postprocessDIC.DICResultsPath)
        
        self.resultFolder = self.postprocessDIC.createResultFolder(index=self.id)
        
        self.postprocessDIC.jsonPath_all = self.postprocessDIC.readResultsNames(
            self.postprocessDIC.DICResultsPath
        )
    
        if self.getInputs(3) != []:
            self.postprocessDIC.poseEstimatePath = self.getInputs(3)[0].value
            self.poseEstimate = self.postprocessDIC.readHomography(
                self.postprocessDIC.poseEstimatePath
            )
        else:
            self.poseEstimate = None
    
        if self.getInputs(4) != []:
            self.postprocessDIC.poseEstimatePath1 = self.getInputs(3)[0].value
            self.postprocessDIC.poseEstimatePath2 = self.getInputs(4)[0].value
    
            self.projectionMatrix1 = self.postprocessDIC.readProjectionMatrix(
                self.postprocessDIC.poseEstimatePath1
            )
            self.projectionMatrix2 = self.postprocessDIC.readProjectionMatrix(
                self.postprocessDIC.poseEstimatePath2
            )
        else:
            self.poseEstimate2 = None
    
        self.markInvalid()
        
        return True
    
    def runPostDIC(self):
    
        if not self.setupPostDIC():
            print("Postprocess setup failed")
            return
    
        res = None
    
        if self.configWidget.method == "2D DIC (Contour strain: grid)":
            # passer la grille au backend (comme en 3D)
            self.postprocessDIC.gridNx = int(self.configWidget.gridNx)
            self.postprocessDIC.gridNy = int(self.configWidget.gridNy)
    
            # lancer le postprocess 2D
            self.postprocessDIC.runPostProcessingAll_2D_Contour(
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                resultFolderPath=self.resultFolder,
                numProcess=self.configWidget._numProcess
            )
    
        if self.configWidget.method == "2D DIC with Scale factor":
            self.postprocessDIC.initPostprocessing(
                windowsize_pixel=self.configWidget.sizeWindow,
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                index_reference=0
            )
    
            self.postprocessDIC.runPostProcessingAll(
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                reference_point=self.postprocessDIC.reference_point,
                indices_within_windows=self.postprocessDIC.indices_within_windows,
                resultFolderPath=self.resultFolder,
                scale=self.configWidget.scale,
                numProcess=self.configWidget._numProcess
            )
    
        if self.configWidget.method == "2D DIC with Homography":
            self.postprocessDIC.initPostprocessing(
                windowsize_pixel=self.configWidget.sizeWindow,
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                index_reference=0
            )
    
            self.postprocessDIC.runPostProcessingAll(
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                reference_point=self.postprocessDIC.reference_point,
                indices_within_windows=self.postprocessDIC.indices_within_windows,
                resultFolderPath=self.resultFolder,
                homography=self.poseEstimate,
                numProcess=self.configWidget._numProcess
            )
    
        if self.configWidget.method == "2D DIC (Contour strain: closer points)":
    
            self.postprocessDIC.nnDistanceFactor = self.configWidget.nnDistanceFactor
            self.postprocessDIC.nnMinNeighbours = self.configWidget.nnMinNeighbours
            self.postprocessDIC.nnMaxNeighbours = self.configWidget.nnMaxNeighbours
            self.postprocessDIC.debugNearestNeighbour = self.configWidget.debugNearestNeighbour
    
            self.postprocessDIC.runPostProcessingAll_2D_NN(
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                resultFolderPath=self.resultFolder,
                numProcess=self.configWidget._numProcess
            )
    
            print("AFTER RUN")
            print("FILES =", os.listdir(self.resultFolder))
    
        if self.configWidget.method == "3D DIC (Contour strain: grid)":
    
            self.postprocessDIC.initPostprocessing_3D(
                windowsize_pixel=self.configWidget.sizeWindow,
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                index_reference=0
            )
    
            # >>> AJOUT: passer la grille au backend
            self.postprocessDIC.gridNx = int(self.configWidget.gridNx)
            self.postprocessDIC.gridNy = int(self.configWidget.gridNy)
    
            self.postprocessDIC.runPostProcessingAll_3D(
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                reference_point=self.postprocessDIC.reference_point,
                indices_within_windows=self.postprocessDIC.indices_within_windows,
                resultFolderPath=self.resultFolder,
                projectionMatrix1=self.projectionMatrix1,
                projectionMatrix2=self.projectionMatrix2,
                numProcess=self.configWidget._numProcess
            )
    
        if self.configWidget.method == "3D DIC (Contour strain: closer points)":
    
            self.postprocessDIC.nnDistanceFactor = self.configWidget.nnDistanceFactor
            self.postprocessDIC.nnMinNeighbours = self.configWidget.nnMinNeighbours
            self.postprocessDIC.nnMaxNeighbours = self.configWidget.nnMaxNeighbours
    
            self.postprocessDIC.runPostProcessingAll_3D_NN(
                jsonPath_all=self.postprocessDIC.jsonPath_all,
                resultFolderPath=self.resultFolder,
                projectionMatrix1=self.projectionMatrix1,
                projectionMatrix2=self.projectionMatrix2,
                numProcess=self.configWidget._numProcess
            )
    
            print("====================================")
            print("NODE :", self.title)
            print("METHOD :", self.configWidget.method)
            print("RESULT FOLDER :", self.resultFolder)
            print("====================================")
    
        res = self.eval()
    
        if res is not None:
            self.mainWidget.plotCurrentState()
        else:
            print("Postprocess results not available")
            
    def getResultName(self):
        return self.postprocessDIC.outputName
    
    def checkCurrentState(self):
            if not self.setupPostDIC():
                return False   
            # setup (inputs, imagesNames, calib, etc.)
            self.setupPostDIC()
        
            # --- 1) choisir le dossier de résultats à relire ---
            resultFolder = getattr(self, "resultFolder", None)
        
            # si on n'a pas de folder sauvegardé / invalide => fallback dossier par défaut
            if (not resultFolder) or (not os.path.isdir(resultFolder)):
                resultFolder = self.postprocessDIC.createResultFolder(index=self.id)
                self.resultFolder = resultFolder
        
            # --- 2) lire les json postprocess existants ---
            jsonPath_all = [p for p in self.postprocessDIC.readResultsNames(resultFolder)
        if os.path.basename(p).startswith("DIC_postprocessing_")]
            
            print("=== JSON FILES FOUND ===")
            for p in jsonPath_all:
                print(p)
            
            num_results = len(jsonPath_all)
            
            num_images = len(self.imagesNames) if hasattr(self, "imagesNames") else 0
            
            print("resultFolder:", resultFolder)
            print("num_results:", num_results)
            print("num_images:", num_images)
        
            if num_results == 0:
                return False
        
            # Si tu veux exiger "autant de results que d'images", garde le check strict:
            if num_images > 0 and num_results != num_images:
                return False
        
            # --- 3) reconstruire l'UI + afficher sans recalcul ---
            print("Postprocessing already done -> reloading graphs")
        
            PostResults = self.postprocessDIC.readPostProcessingResult(jsonPath_all[0])
            if PostResults[1].shape[1] == 3:
                self.configWidget.method = "3D DIC (Contour strain: grid)"
        
            self.mainWidget.setLayoutContent()
            self.postprocessDIC.outputName = resultFolder
        
            # remet au frame 0 et affiche
            self.mainWidget.ind_image = 0
            self.mainWidget.timeSerise = False
            self.mainWidget.plotCurrentState()
        
            self.markDirty(False)
            self.markInvalid(False)
            return True
    

class VibNodeConfig_PostprocessDIC(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node

        # paramètres par défaut
        self.method = ""
        self.sizeWindow = 50
        self._numProcess = 1
        self.scale = 1

        # >>> AJOUT: taille de grille pour strain (NX colonnes, NY lignes)
        self.gridNx = 5
        self.gridNy = 54
        self.nnDistanceFactor = 1.8
        self.nnMinNeighbours = 3
        self.nnMaxNeighbours = 8
        self.debugNearestNeighbour = True
        self.initUI()

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of DIC Postprocess Node"), 0, 0)

        self.layout.addWidget(QLabel("Method"), 1, 0)
        self.methodSelector = QComboBox(self)
        self.methodSelector.addItem('')
        self.methodSelector.addItem('2D DIC with Scale factor')
        self.methodSelector.addItem('2D DIC with Homography')
        self.methodSelector.addItem('2D DIC (Contour strain: grid)')
        self.methodSelector.addItem('2D DIC (Contour strain: closer points)')
        self.methodSelector.addItem('3D DIC (Contour strain: grid)')
        self.methodSelector.addItem('3D DIC (Contour strain: closer points)')
        self.layout.addWidget(self.methodSelector, 1, 1)
        self.methodSelector.activated[str].connect(self.onActivated_methodSelector)

        self.layout.addWidget(QLabel("."), 2, 0)
        self.layout.addWidget(QLabel("."), 3, 0)

        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 12, 0, 1, 2)

        self.setLayout(self.layout)

    def onActivated_methodSelector(self, text):
    
        print("Activated:", text)
        self.method = text
    
        self.clearConfigWidgets()
    
        if text == "2D DIC with Scale factor":
            self.layout_PostDIC_SF()
    
        elif text == "2D DIC with Homography":
            self.layout_PostDIC_Homography()
    
        elif text == "2D DIC (Contour strain: grid)":
            self.layout_PostDIC_Grid()
    
        elif text == "3D DIC (Contour strain: grid)":
            self.layout_PostDIC_Grid()
    
        elif text == "2D DIC (Contour strain: closer points)":
            self.layout_PostDIC_NN()
        
        elif text == "3D DIC (Contour strain: closer points)":
            self.layout_PostDIC_NN()
            
    def clearConfigWidgets(self):
    
        for row in range(2, 12):
    
            for col in range(2):
    
                item = self.layout.itemAtPosition(row, col)
    
                if item is not None:
    
                    widget = item.widget()
    
                    if widget is not None:
                        widget.deleteLater()      

    # ---------- callbacks ----------
    def onChanged_scalefactor(self, text):
        if text != "":
            try:
                self.scale = float(text)
                print("scale: ", self.scale)
            except:
                print("Error: Please enter a number")

    def onChanged_numProcess(self, text):
        if text != "":
            try:
                self._numProcess = int(text)
                print("numProcess: ", self._numProcess)
            except:
                print("Error: Please enter a number")

    def onChanged_sizeWindow(self, text):
        if text != "":
            try:
                self.sizeWindow = int(text)
                print("sizeWindow: ", self.sizeWindow)
            except:
                print("Error: Please enter an integer for sizeWindow")

    # ---------- layouts ----------
    def layout_PostDIC_SF(self):

        self.layout.addWidget(QLabel("Square size (px)"), 2, 0)
        sizeWindow = QLineEdit(self)
        sizeWindow.setText(str(self.sizeWindow))
        self.layout.addWidget(sizeWindow, 2, 1)
        sizeWindow.textChanged.connect(self.onChanged_sizeWindow)

        scalefactor = QLineEdit(self)
        scalefactor.setText(str(self.scale))
        self.layout.addWidget(QLabel("Scale factor (mm/px)"), 3, 0)
        self.layout.addWidget(scalefactor, 3, 1)
        scalefactor.textChanged.connect(self.onChanged_scalefactor)

        self.layout.addWidget(QLabel("Number of Process"), 4, 0)
        tooltip = "Number of process to run the Postprocessing, Maximum number of process is " + str(os.cpu_count())
        numProcess = QLineEdit(self)
        numProcess.setToolTip(tooltip)
        numProcess.setText(str(self._numProcess))
        self.layout.addWidget(numProcess, 4, 1)
        numProcess.textChanged.connect(self.onChanged_numProcess)

        self.setLayout(self.layout)

    def layout_PostDIC_Homography(self):
    
        self.layout.addWidget(QLabel("Square size (px)"), 2, 0)
    
        sizeWindow = QLineEdit(self)
        sizeWindow.setText(str(self.sizeWindow))
        sizeWindow.textChanged.connect(self.onChanged_sizeWindow)
    
        self.layout.addWidget(sizeWindow, 2, 1)
    
        self.layout.addWidget(QLabel("Number of Process"), 3, 0)
    
        numProcess = QLineEdit(self)
        numProcess.setText(str(self._numProcess))
        numProcess.textChanged.connect(self.onChanged_numProcess)
    
        self.layout.addWidget(numProcess, 3, 1)
    
        self.layout.addWidget(self.buttonRun, 12, 0, 1, 2)
        
    def layout_PostDIC_Grid(self):
    
        self.layout.addWidget(QLabel("Size Window"), 2, 0)
    
        sizeWindow = QLineEdit(self)
        sizeWindow.setText(str(self.sizeWindow))
        sizeWindow.textChanged.connect(self.onChanged_sizeWindow)
    
        self.layout.addWidget(sizeWindow, 2, 1)
    
        self.layout.addWidget(QLabel("Number of Process"), 3, 0)
    
        numProcess = QLineEdit(self)
        numProcess.setText(str(self._numProcess))
        numProcess.textChanged.connect(self.onChanged_numProcess)
    
        self.layout.addWidget(numProcess, 3, 1)
    
        self.layout.addWidget(QLabel("Grid NX"), 4, 0)
    
        self.gridNxSpin = QSpinBox(self)
        self.gridNxSpin.setRange(1, 5000)
        self.gridNxSpin.setValue(int(self.gridNx))
        self.gridNxSpin.valueChanged.connect(self._onChanged_gridNxSpin)
    
        self.layout.addWidget(self.gridNxSpin, 4, 1)
    
        self.layout.addWidget(QLabel("Grid NY"), 5, 0)
    
        self.gridNySpin = QSpinBox(self)
        self.gridNySpin.setRange(1, 5000)
        self.gridNySpin.setValue(int(self.gridNy))
        self.gridNySpin.valueChanged.connect(self._onChanged_gridNySpin)
    
        self.layout.addWidget(self.gridNySpin, 5, 1)
    
        self.layout.addWidget(self.buttonRun, 12, 0, 1, 2)
            
    def layout_PostDIC_NN(self):
    
        self.layout.addWidget(QLabel("Size Window"), 2, 0)
    
        sizeWindow = QLineEdit(self)
        sizeWindow.setText(str(self.sizeWindow))
        sizeWindow.textChanged.connect(self.onChanged_sizeWindow)
    
        self.layout.addWidget(sizeWindow, 2, 1)
    
        self.layout.addWidget(QLabel("Number of Process"), 3, 0)
    
        numProcess = QLineEdit(self)
        numProcess.setText(str(self._numProcess))
        numProcess.textChanged.connect(self.onChanged_numProcess)
    
        self.layout.addWidget(numProcess, 3, 1)
    
        self.layout.addWidget(QLabel("Distance factor"), 4, 0)
    
        self.nnDistanceFactorSpin = QDoubleSpinBox(self)
        self.nnDistanceFactorSpin.setRange(0.5, 10.0)
        self.nnDistanceFactorSpin.setSingleStep(0.1)
        self.nnDistanceFactorSpin.setValue(self.nnDistanceFactor)
    
        self.nnDistanceFactorSpin.valueChanged.connect(
            lambda v: setattr(self, "nnDistanceFactor", float(v))
        )
    
        self.layout.addWidget(self.nnDistanceFactorSpin, 4, 1)
    
        self.layout.addWidget(QLabel("Minimum neighbours"), 5, 0)
    
        self.nnMinNeighboursSpin = QSpinBox(self)
        self.nnMinNeighboursSpin.setRange(3, 20)
        self.nnMinNeighboursSpin.setValue(self.nnMinNeighbours)
    
        self.nnMinNeighboursSpin.valueChanged.connect(
            lambda v: setattr(self, "nnMinNeighbours", int(v))
        )
    
        self.layout.addWidget(self.nnMinNeighboursSpin, 5, 1)
        
        self.layout.addWidget(QLabel("Maximum neighbours"), 6, 0)

        self.nnMaxNeighboursSpin = QSpinBox(self)
        self.nnMaxNeighboursSpin.setRange(3, 50)
        self.nnMaxNeighboursSpin.setValue(self.nnMaxNeighbours)
        
        self.nnMaxNeighboursSpin.valueChanged.connect(
            lambda v: setattr(self, "nnMaxNeighbours", int(v))
        )
        
        self.layout.addWidget(self.nnMaxNeighboursSpin, 6, 1)

        self.debugNNCheck = QCheckBox("Write debug report")
    
        self.debugNNCheck.setChecked(
            self.debugNearestNeighbour
        )
    
        self.debugNNCheck.stateChanged.connect(
            lambda v: setattr(
                self,
                "debugNearestNeighbour",
                bool(v)
            )
        )
    
        self.layout.addWidget(self.debugNNCheck, 7, 0, 1, 2)
    
        self.layout.addWidget(self.buttonRun, 12, 0, 1, 2)

    def _onChanged_gridNxSpin(self, v):
        self.gridNx = int(v)
        print("gridNx: ", self.gridNx)

    def _onChanged_gridNySpin(self, v):
        self.gridNy = int(v)
        print("gridNy: ", self.gridNy)


from matplotlib.backend_bases import MouseButton

class VibNodeMain_PostprocessDIC(QWidget):

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()
        self.ind_image = 0
        self.timeSerise = False
    
    def openContextMenu(self, event):
        # event from matplotlib
        menu = QMenu()
        action = menu.addAction("Point data at (%.2f, %.2f)" % (event.xdata, event.ydata))
        qpoint = QCursor.pos()

        # connect the action to a function
        action.triggered.connect(self.plotTimeSeries)
        menu.exec_(qpoint)

    def plotTimeSeries(self):
        self.timeSerise = True
        print("Plot time series")
        self.removeUI()
        print("Remove UI")

        self.setLayoutImageContent()

        # find the timeseries data of the point query
        self.displacementTimeSeries, self.strainTimeSeries = self.node.postprocessDIC.readTimeseries(self.point_query, self.ind_image, self.node.resultFolder)

        self.setLayout(self.layout)
        print("Init UI")
        self.plotCurrentState()



    def on_button_press(self, event):

        if event.button == MouseButton.RIGHT:  
            print('press right')
            if event.inaxes:
                print(f'button {event.button} pressed at ({event.xdata}, {event.ydata})')
                self.point_query = np.array([event.xdata, event.ydata]).astype(int)
                print("point_query: ", self.point_query)
                self.openContextMenu(event)


    def initUI(self):

        self.initializePlot()
        
        self.layout = QGridLayout()

        self.setLayoutContent()

        self.setLayout(self.layout)
        # self.show()

    def onChanged_scaleControls(self, *args):
        auto = self.cbAutoScale.isChecked()
        self.spinVmin.setEnabled(not auto)
        self.spinVmax.setEnabled(not auto)
        self.plotCurrentState()

    def setLayoutImageContent(self):
        self.layout.addWidget(self.toolbar, 0, 0)

        self.layout.addWidget(self.canvas, 1, 0)

    def setLayoutContent(self):

        self.setLayoutImageContent()
        self.typeSelector = QComboBox(self)
        
        
        if "3D DIC" in self.node.configWidget.method:
            self.typeSelector.addItem("Displacement X (mm)")
            self.typeSelector.addItem("Displacement Y (mm)")
            self.typeSelector.addItem("Displacement Z (mm)")
            self.typeSelector.addItem("Strain XX")
            self.typeSelector.addItem("Strain YY")
            self.typeSelector.addItem("Strain XY")
     
        else:
            self.typeSelector.addItem("Displacement X (mm)")
            self.typeSelector.addItem("Displacement Y (mm)")
            self.typeSelector.addItem("Strain XX")
            self.typeSelector.addItem("Strain YY")
            self.typeSelector.addItem("Strain XY")

        self.layout.addWidget(self.typeSelector, 2, 0)
        # ---- Color scale controls (vmin/vmax) ----
        scale_row = QWidget(self)
        scale_layout = QHBoxLayout(scale_row)
        scale_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cbAutoScale = QCheckBox("Auto scale", self)
        self.cbAutoScale.setChecked(True)
        self.cbAutoScale.stateChanged.connect(self.onChanged_scaleControls)
        scale_layout.addWidget(self.cbAutoScale)
        
        scale_layout.addWidget(QLabel("Min:", self))
        self.spinVmin = QDoubleSpinBox(self)
        self.spinVmin.setRange(-1e12, 1e12)
        self.spinVmin.setDecimals(6)
        self.spinVmin.setSingleStep(0.1)
        self.spinVmin.setValue(0.0)
        self.spinVmin.setEnabled(False)
        self.spinVmin.editingFinished.connect(self.onChanged_scaleControls)
        scale_layout.addWidget(self.spinVmin)
        
        scale_layout.addWidget(QLabel("Max:", self))
        self.spinVmax = QDoubleSpinBox(self)
        self.spinVmax.setRange(-1e12, 1e12)
        self.spinVmax.setDecimals(6)
        self.spinVmax.setSingleStep(0.1)
        self.spinVmax.setValue(1.0)
        self.spinVmax.setEnabled(False)
        self.spinVmax.editingFinished.connect(self.onChanged_scaleControls)
        scale_layout.addWidget(self.spinVmax)
        
        self.layout.addWidget(scale_row, 6, 0)
        
        self.typeSelector.activated[str].connect(self.onActivated_typeSelector)

         # Label for slider
        self.image_slider_label = QLabel("Select Image:", self)
        self.layout.addWidget(self.image_slider_label, 3, 0)

        # Slider for selecting images
        self.imageSelectorSlider = QSlider(Qt.Horizontal)
        self.imageSelectorSlider.setMinimum(0)  # First image index
        self.imageSelectorSlider.setMaximum(10)  # Last image index
        self.imageSelectorSlider.setTickInterval(1)
        self.imageSelectorSlider.setTickPosition(QSlider.NoTicks)
        self.imageSelectorSlider.setValue(0)
        self.imageSelectorSlider.setTracking(True)
        self.imageSelectorSlider.valueChanged.connect(self.onActivated_imageSlider)

        # Add slider to layout
        self.layout.addWidget(self.imageSelectorSlider, 4, 0)

        # Label for displaying the selected image number
        self.current_image_label = QLabel(f"Image: 0", self)
        self.layout.addWidget(self.current_image_label, 5, 0)



    def onActivated_imageSlider(self, value):
        """Updates the selected image index based on the slider value and re-plots the data."""
        
        self.ind_image = value  # Update the current image index
        self.current_image_label.setText(f"Image: {value}")  # Update label

        print("Activated Image:", value)

        self.readResult()  # Load the new image data
        self.timeSerise = False
        self.plotCurrentState()  # Re-plot with the new image


    def removeUI(self):
        self.layout.removeWidget(self.canvas)
        self.layout.removeWidget(self.toolbar)
        # self.layout.removeWidget(self.imageSelector)
        # self.layout.removeWidget(self.typeSelector)


    def readResult(self):

        resultFile = os.path.join(self.node.resultFolder, f"DIC_postprocessing_{self.ind_image:04d}.json")
        self.currentPoint, self.displacementField, self.strainField = self.node.postprocessDIC.readPostProcessingResult(resultFile)
        # triangulation of points
        x = self.currentPoint[:, 0]
        y = self.currentPoint[:, 1]
        step_approx = self.node.configWidget.sizeWindow
        triang = tri.Triangulation(x, y)
        tri_pts = np.array([x[triang.triangles], y[triang.triangles]])  # (2, num_triangles, 3)
        edges = np.linalg.norm(tri_pts[:, :, [1, 2, 0]] - tri_pts[:, :, [0, 1, 2]], axis=0)  # (num_triangles, 3)
        mask = np.any(edges > 5.0*step_approx, axis=1)
        self.valid_triangles = triang.triangles[~mask]  # select only the valid triangles
        self.changeImage(self.ind_image)


    def changeImage(self, ind):
        curImg = cv2.imread(self.node.imagesNames[ind])
        if self.node.calibResult is not None:
            mtx, dist = self.node.calibResult
            self.curImg = self.node.postprocessDIC.undistortImage(curImg, mtx, dist)
        else:
            self.curImg = curImg


    def initializePlot(self):
        self.figure = plt.figure()
        self.figure.patch.set_facecolor('#666')

        self.colorbar = None
        self.canvas = FigureCanvas(self.figure)
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.axis('off')


        self.ax1.set_facecolor('#666')

        self.toolbar = NavigationToolbar(self.canvas, self)

        self.ax1.set_facecolor('#666')
        # plt.connect('motion_notify_event', self.on_move)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)



    def onActivated_typeSelector(self, text):
        print("Activated: ", text)
        self.plotCurrentState()
        
    # Julien Archez modif heatmap    
    def _show_grid_as_heatmap_percent(self, values_1d, title):
        GRID_NX = int(self.node.configWidget.gridNx)
        GRID_NY = int(self.node.configWidget.gridNy)
    
        Z = np.array(values_1d).reshape(GRID_NY, GRID_NX)
        Z = 100.0 * Z  # affichage en %
    
        im = self.ax1.imshow(Z, origin="lower", interpolation="nearest", aspect="auto", cmap=cm.jet)
        self.ax1.set_title(title + " (%)")
        self.ax1.set_xlabel("i")
        self.ax1.set_ylabel("j")
        self.colorbar = self.ax1.figure.colorbar(im, ax=self.ax1)


    def _overlay_squares_on_image(self, values_1d, title, unit, as_percent=False, vmin=None, vmax=None):

        """
        Affiche l'image + une grille de carrés (Nx*Ny) colorés par values_1d.
        - unit: "(mm)" ou "(%)"
        - as_percent : True => *100 avant affichage
        """
        # image
        self.ax1.imshow(self.curImg, cmap="gray")
        
        vals = np.array(values_1d, dtype=float).copy()
        if as_percent:
            vals *= 100.0
    
        
        centers = np.array(self.currentPoint[:, :2], dtype=float)
    
        # taille carré (en px) = meshSize de preprocess si dispo dans le json postprocess
        # -> sinon fallback: estimate spacing
        mesh_px = None
        try:
            # si tu as sauvegardé meshSize dans le json postprocess: self.meshSize_current
            mesh_px = float(getattr(self, "meshSize_current", None))
        except Exception:
            mesh_px = None
    
        if mesh_px is None:
            mesh_px = float(self.node.configWidget.sizeWindow)
    
        half = 0.5 * mesh_px
    
        # colormap + normalisation (ignore NaN)
        finite = np.isfinite(vals)
        
        if vmin is None or vmax is None:
            if np.any(finite):
                vmin2 = np.nanmin(vals)
                vmax2 = np.nanmax(vals)
        
                if vmin2 == vmax2:
                    vmax2 = vmin2 + 1e-12
            else:
                vmin2, vmax2 = 0.0, 1.0
                        
        else:
            vmin2, vmax2 = float(vmin), float(vmax)
            if vmin2 == vmax2:
                vmax2 = vmin2 + 1e-12
        
        norm = plt.Normalize(vmin=vmin2, vmax=vmax2)
        cmap = cm.jet

    
        for (cx, cy), v in zip(centers, vals):
        
            if not np.isfinite(v):
                continue
        
            color = cmap(norm(v))
        
            r = Rectangle(
                (cx-half, cy-half),
                2*half,
                2*half,
                facecolor=color,
                edgecolor="none",
                alpha=0.35
            )
        
            self.ax1.add_patch(r)
    
        # title + colorbar
        self.ax1.set_title(f"{title} {unit}")
        mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        self.colorbar = self.ax1.figure.colorbar(mappable, ax=self.ax1)
    
    def resetPlot(self):
        self.figure.clear()
        self.figure.patch.set_facecolor('#666')
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.axis('off')
        self.ax1.set_facecolor('#666')

    def plotCurrentState(self):
        self.readResult()
    
        # slider max = nb images
        self.imageSelectorSlider.setMaximum(len(self.node.imagesNames) - 1)
        self.imageSelectorSlider.update()
        self.imageSelectorSlider.repaint()
    
        ind_type = self.typeSelector.currentText()
    
        # clear + remove old colorbar
        self.ax1.clear()
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except Exception:
                pass
            self.colorbar = None
    
        # ---- manual/auto scale from UI ----
        vmin = None
        vmax = None
        if hasattr(self, "cbAutoScale") and (not self.cbAutoScale.isChecked()):
            vmin = float(self.spinVmin.value())
            vmax = float(self.spinVmax.value())
    
        # ---------------- Displacements (squares overlay) ----------------
        if ind_type == "Displacement X (mm)":
            self.resetPlot()
            if self.timeSerise is False:
                self._overlay_squares_on_image(
                    self.displacementField[:, 0],
                    "Displacement X",
                    "(mm or pixel)",
                    as_percent=False,
                    vmin=vmin, vmax=vmax
                )
            else:
                self.ax1.plot(self.displacementTimeSeries[:, 0])
                self.ax1.set_title("Displacement X Time Series")
                self.ax1.set_xlabel("Frame")
                self.ax1.set_ylabel("Displacement X (mm)")
    
        elif ind_type == "Displacement Y (mm)":
            self.resetPlot()
            if self.timeSerise is False:
                self._overlay_squares_on_image(
                    self.displacementField[:, 1],
                    "Displacement Y",
                    "(mm or pixel)",
                    as_percent=False,
                    vmin=vmin, vmax=vmax
                )
            else:
                self.ax1.plot(self.displacementTimeSeries[:, 1])
                self.ax1.set_title("Displacement Y Time Series")
                self.ax1.set_xlabel("Frame")
                self.ax1.set_ylabel("Displacement Y (mm)")
    
        elif "3D DIC" in self.node.configWidget.method and ind_type == "Displacement Z (mm)":
            self.resetPlot()
            if self.timeSerise is False:
                self._overlay_squares_on_image(
                    self.displacementField[:, 2],
                    "Displacement Z",
                    "(mm)",
                    as_percent=False,
                    vmin=vmin, vmax=vmax
                )
            else:
                self.ax1.plot(self.displacementTimeSeries[:, 2])
                self.ax1.set_title("Displacement Z Time Series")
                self.ax1.set_xlabel("Frame")
                self.ax1.set_ylabel("Displacement Z (mm)")
    
        # ---------------- Strains (squares overlay + % ; time series in %) ----------------
        elif ind_type == "Strain XX":
            self.resetPlot()
            if self.timeSerise is False:
                self._overlay_squares_on_image(
                    self.strainField[:, 0],
                    "Strain XX",
                    "(%)",
                    as_percent=True,
                    vmin=vmin, vmax=vmax
                )
            else:
                self.ax1.plot(100.0 * self.strainTimeSeries[:, 0])
                self.ax1.set_title("Strain XX Time Series (%)")
                self.ax1.set_xlabel("Frame")
                self.ax1.set_ylabel("Strain XX (%)")
    
        elif ind_type == "Strain YY":
            self.resetPlot()
            if self.timeSerise is False:
                self._overlay_squares_on_image(
                    self.strainField[:, 1],
                    "Strain YY",
                    "(%)",
                    as_percent=True,
                    vmin=vmin, vmax=vmax
                )
            else:
                self.ax1.plot(100.0 * self.strainTimeSeries[:, 1])
                self.ax1.set_title("Strain YY Time Series (%)")
                self.ax1.set_xlabel("Frame")
                self.ax1.set_ylabel("Strain YY (%)")
    
        elif ind_type == "Strain XY":
            self.resetPlot()
            if self.timeSerise is False:
                self._overlay_squares_on_image(
                    self.strainField[:, 2],
                    "Strain XY",
                    "(%)",
                    as_percent=True,
                    vmin=vmin, vmax=vmax
                )
            else:
                self.ax1.plot(100.0 * self.strainTimeSeries[:, 2])
                self.ax1.set_title("Strain XY Time Series (%)")
                self.ax1.set_xlabel("Frame")
                self.ax1.set_ylabel("Strain XY (%)")
    
        # ---------------- theme UI ----------------
        if self.timeSerise is False:
            self.ax1.axis("off")
            self.ax1.set_facecolor("#666")
            self.figure.patch.set_facecolor("#666")
        else:
            self.ax1.axis("on")
            self.ax1.set_facecolor("#fff")
            self.figure.patch.set_facecolor("#fff")
    
        self.canvas.draw()


    