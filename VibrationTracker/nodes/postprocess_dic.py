from PyQt5.QtWidgets import QPushButton, QGridLayout, QLabel, QWidget, QComboBox, QLineEdit, QCheckBox, QSpacerItem, QSizePolicy, QMessageBox, QMenu, QSlider
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QPoint
from VibrationTracker.vib_conf import register_node, OP_NODE_POSTPROCESSDIC
from VibrationTracker.vib_node_base import VibNode, VibGraphicsNode
from nodeeditor.node_content_widget import QDMNodeContentWidget
from nodeeditor.utils import dumpException
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
import cv2
import os
from matplotlib import cm
from VibrationTracker.module.dic_postprocessing import *
from PyQt5.Qt import QCursor
import matplotlib.tri as tri

class VibPostprocessDICContent(QDMNodeContentWidget):
    def initUI(self):
        self.setStyleSheet(''' font-size: 14px; ''')

        self.layout = QGridLayout()
        self.layout.setContentsMargins(10,10,10,30)
        # self.layout.addWidget(QLabel(""), 1, 0)
        # self.layout.addWidget(QLabel(""), 2, 0)
        # self.layout.addWidget(QLabel(""), 1, 0)

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
        self.postprocessDIC.filePath = self.getInputs(0)[0].value
        print("filePath: ", self.postprocessDIC.filePath)
        if len(self.getInputs(1)) > 0:
            self.postprocessDIC.calibPath = self.getInputs(1)[0].value
            print("calibPath: ", self.postprocessDIC.calibPath)
            self.calibResult = self.postprocessDIC.readCalibNameFromJson(self.postprocessDIC.calibPath)
        else:
            self.postprocessDIC.calibPath = None
            self.calibResult = None
        
        self.imagesNames = self.postprocessDIC.readImageNamesFromJson(self.postprocessDIC.filePath)
        self.resultFolder = self.postprocessDIC.createResultFolder(index=self.id)

        self.postprocessDIC.DICResultsPath = self.getInputs(2)[0].value
        self.postprocessDIC.jsonPath_all = self.postprocessDIC.readResultsNames(self.postprocessDIC.DICResultsPath)
        if self.getInputs(3) != []:
            self.postprocessDIC.poseEstimatePath = self.getInputs(3)[0].value
            self.poseEstimate = self.postprocessDIC.readHomography(self.postprocessDIC.poseEstimatePath)
        else:
            self.poseEstimate = None
        if self.getInputs(4) != []:
            self.postprocessDIC.poseEstimatePath1 = self.getInputs(3)[0].value
            self.postprocessDIC.poseEstimatePath2 = self.getInputs(4)[0].value
            self.projectionMatrix1 = self.postprocessDIC.readProjectionMatrix(self.postprocessDIC.poseEstimatePath1)
            self.projectionMatrix2 = self.postprocessDIC.readProjectionMatrix(self.postprocessDIC.poseEstimatePath2)
        else:
            self.poseEstimate2 = None

        # self.mainWidget.plotCurrentState()
        self.markInvalid()

    def runPostDIC(self):
        
        self.setupPostDIC()
        
        if self.configWidget.method == "2D DIC with Scale factor":
            self.postprocessDIC.initPostprocessing(windowsize_pixel=self.configWidget.sizeWindow, jsonPath_all=self.postprocessDIC.jsonPath_all, index_reference=0)


            self.postprocessDIC.runPostProcessingAll(jsonPath_all = self.postprocessDIC.jsonPath_all, reference_point=self.postprocessDIC.reference_point, indices_within_windows=self.postprocessDIC.indices_within_windows, resultFolderPath = self.resultFolder, scale = self.configWidget.scale, numProcess = self.configWidget._numProcess)
            #     print("Error: ", e)
            #     if type(e) == ValueError:
            #         QMessageBox.critical(None, "Error", f"Size of searching area is too small to detect large displacement {e}")
            
        if self.configWidget.method == "2D DIC with Homography":
            self.postprocessDIC.initPostprocessing(windowsize_pixel=self.configWidget.sizeWindow, jsonPath_all=self.postprocessDIC.jsonPath_all, index_reference=0)

            # try:
            self.postprocessDIC.runPostProcessingAll(jsonPath_all = self.postprocessDIC.jsonPath_all, reference_point=self.postprocessDIC.reference_point, indices_within_windows=self.postprocessDIC.indices_within_windows, resultFolderPath = self.resultFolder, homography = self.poseEstimate, numProcess = self.configWidget._numProcess)
            #     print("Error: ", e)
            #     if type(e) == ValueError:
            #         QMessageBox.critical(None, "Error", f"Size of searching area is too small to detect large displacement {e}")
        if self.configWidget.method == "3D DIC":
            self.postprocessDIC.initPostprocessing_3D(windowsize_pixel=self.configWidget.sizeWindow, jsonPath_all=self.postprocessDIC.jsonPath_all, index_reference=0)

            self.postprocessDIC.runPostProcessingAll_3D(jsonPath_all = self.postprocessDIC.jsonPath_all, reference_point=self.postprocessDIC.reference_point, indices_within_windows=self.postprocessDIC.indices_within_windows, resultFolderPath = self.resultFolder, projectionMatrix1 = self.projectionMatrix1, projectionMatrix2 = self.projectionMatrix2, numProcess = self.configWidget._numProcess)
            # self.trackTarget.plotTrackingResult(TrackResults)
        self.eval()
        self.mainWidget.plotCurrentState()


    def getResultName(self):
        return self.postprocessDIC.outputName

    def checkCurrentState(self):
    
        self.setupPostDIC()

        
        # find the number of jsonfiles  
        jsonPath_all = self.postprocessDIC.readResultsNames(self.resultFolder)
        num_results = len(jsonPath_all)
        num_images = len(self.imagesNames)
        print("num_results: ", num_results)
        print("num_images: ", num_images)
        # check if the output file exists

        if num_results == num_images:

            print("Postprocessing is done")
            # read the first result
            PostResults = self.postprocessDIC.readPostProcessingResult(jsonPath_all[0])
            print("PostResults: ", PostResults[1].shape)
            
            if PostResults[1].shape[1] == 3:
                print("3D DIC")
                self.configWidget.method = "3D DIC"
            self.mainWidget.setLayoutContent()

            self.postprocessDIC.outputName = self.resultFolder
            
            # mainwidgetSetting

            self.mainWidget.plotCurrentState()

            # plot the tracking result
            self.markDirty(False)
            self.markInvalid(False)
            return True
        else:
            return False

class VibNodeConfig_PostprocessDIC(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.initUI()
        self.method = ""
        self.sizeWindow = 50
        self._numProcess = 1
        self.scale = 1

    def initUI(self):
        self.layout = QGridLayout()
        self.layout.addWidget(QLabel("Configurations of DIC Postprocess Node"), 0, 0)

        ## 
        self.layout.addWidget(QLabel("Method"), 1, 0)
        methodSelector = QComboBox(self)
        methodSelector.addItem('')
        methodSelector.addItem('2D DIC with Scale factor')
        methodSelector.addItem('2D DIC with Homography')
        methodSelector.addItem('3D DIC')
        self.layout.addWidget(methodSelector, 1, 1)
        methodSelector.activated[str].connect(self.onActivated_methodSelector)

        self.layout.addWidget(QLabel("."), 2, 0)
        self.layout.addWidget(QLabel("."), 3, 0)


        self.buttonRun = QPushButton("Run", self)
        self.layout.addWidget(self.buttonRun, 5, 0, 1, 2)

        self.setLayout(self.layout)

    def onActivated_methodSelector(self, text):
        print("Activated: ", text)
        self.method = text

        print("method: ", self.method)

        if text == "2D DIC with Scale factor":
            self.layout_PostDIC_SF()
        elif text == "2D DIC with Homography":
            self.layout_PostDIC_Homography()
        elif text == "3D DIC":
            self.layout_PostDIC_Homography()

    def layout_PostDIC_SF(self):
        
        self.layout.addWidget(QLabel("Size of the window for smoothing"), 2, 0)
        sizeWindow = QLineEdit(self)
        sizeWindow.setText(str(self.sizeWindow))
        # sizeWindow.setAlignment(Qt.AlignRight)
        self.layout.addWidget(sizeWindow, 2, 1)
        # disable the sizeWindow
        # sizeWindow.setDisabled(True)        
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

    def onChanged_scalefactor(self, text):
        if text != "":
            try:
                self.scale = float(text)
                print("scale: ", self.scale)
            except:
                print("Error: Please enter a number")
        
    def onChanged_numProcess(self, text):
        if text != "":
            try :
                self._numProcess = int(text)
                print("numProcess: ", self._numProcess)
            except:
                print("Error: Please enter a number")


    def onChanged_sizeWindow(self, text):
        if text != "":
            self.sizeWindow = int(text)
            print("sizeWindow: ", self.sizeWindow)
                
    def layout_PostDIC_Homography(self):
        self.layout.addWidget(QLabel("Size of the window for smoothing"), 2, 0)
        sizeWindow = QLineEdit(self)
        self.sizeWindow = 50
        sizeWindow.setText(str(self.sizeWindow))
        # sizeWindow.setAlignment(Qt.AlignRight)
        self.layout.addWidget(sizeWindow, 2, 1)
        # disable the sizeWindow
        # sizeWindow.setDisabled(True)        
        sizeWindow.textChanged.connect(self.onChanged_sizeWindow)

        scalefactor = QLineEdit(self)
        scalefactor.setText("1") 
        label = QLabel("Scale factor (mm/px)")
        f = label.font()
        f.setStrikeOut(True)
        label.setFont(f)
        self.layout.addWidget(label, 3, 0)
        scalefactor.setDisabled(True)
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

    def setLayoutImageContent(self):
        self.layout.addWidget(self.toolbar, 0, 0)

        self.layout.addWidget(self.canvas, 1, 0)

    def setLayoutContent(self):

        self.setLayoutImageContent()
        self.typeSelector = QComboBox(self)
        
        
        if self.node.configWidget.method == "3D DIC":
            self.typeSelector.addItem("Displacement X (mm)")
            self.typeSelector.addItem("Displacement Y (mm)")
            self.typeSelector.addItem("Displacement Z (mm)")
            self.typeSelector.addItem("Strain XX")
            self.typeSelector.addItem("Strain YY")
            self.typeSelector.addItem("Strain XY")
            self.typeSelector.addItem("Strain ZZ")
            self.typeSelector.addItem("Strain XZ")
            self.typeSelector.addItem("Strain YZ")
        else:
            self.typeSelector.addItem("Displacement X (mm)")
            self.typeSelector.addItem("Displacement Y (mm)")
            self.typeSelector.addItem("Strain XX")
            self.typeSelector.addItem("Strain YY")
            self.typeSelector.addItem("Strain XY")

        self.layout.addWidget(self.typeSelector, 2, 0)
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

        resultFile = self.node.resultFolder + "\DIC_postprocessing_%04d.json" % self.ind_image
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
        plt.connect('button_press_event', self.on_button_press)


    def resetPlot(self):
        
        self.figure.clear()
        self.figure.patch.set_facecolor('#666')
        self.ax1 = self.figure.add_subplot(111)
        self.ax1.axis('off')
        self.ax1.set_facecolor('#666')


    def onActivated_typeSelector(self, text):
        print("Activated: ", text)
        self.plotCurrentState()


    def plotCurrentState(self):
        self.readResult()

        self.imageSelectorSlider.setMaximum(len(self.node.imagesNames) - 1)
        self.imageSelectorSlider.update()
        self.imageSelectorSlider.repaint()
        ind_type = self.typeSelector.currentText()

        self.ax1.clear()    
        
        if self.colorbar is not None:
            try:
                self.colorbar.remove()
            except AttributeError:
                pass
            self.colorbar = None

        if ind_type == "Displacement X (mm)":
            self.resetPlot()
            if self.timeSerise == False:
                self.ax1.imshow(self.curImg, cmap='gray')
                sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.valid_triangles, self.displacementField[:,  0], cmap=cm.jet, alpha=0.5)
                # sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.displacementField[:,  0], cmap=cm.jet, alpha=0.5, shading='gouraud')
                min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.displacementField[:, 0])
                sc.set_clim(min_confidence, max_confidence)
                self.ax1.set_title('Displacement X')
                self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)

            else:
                self.ax1.plot(self.displacementTimeSeries[:, 0])
                self.ax1.set_title('Displacement X Time Series')
                self.ax1.set_xlabel('Frame')
                self.ax1.set_ylabel('Displacement X (mm)')


        if ind_type == "Displacement Y (mm)":
            self.resetPlot()
            if self.timeSerise == False:
                self.ax1.imshow(self.curImg, cmap='gray')
                sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.valid_triangles, self.displacementField[:,  1], cmap=cm.jet, alpha=0.5)
                min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.displacementField[:, 1])
                sc.set_clim(min_confidence, max_confidence)
                self.ax1.set_title('Displacement Y')
                self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)
            
            else:
                self.ax1.plot(self.displacementTimeSeries[:, 1])
                self.ax1.set_title('Displacement Y Time Series')
                self.ax1.set_xlabel('Frame')
                self.ax1.set_ylabel('Displacement Y (mm)')


        if ind_type == "Strain XX":
            self.resetPlot()
            if self.timeSerise == False:
                self.ax1.imshow(self.curImg, cmap='gray')
                sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.valid_triangles, self.strainField[:,  0], cmap=cm.jet, alpha=0.5)

                min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.strainField[:, 0])
                sc.set_clim(min_confidence, max_confidence)
                self.ax1.set_title('Strain XX')
                self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)

            else:
                self.ax1.plot(self.strainTimeSeries[:, 0])
                self.ax1.set_title('Strain XX Time Series')
                self.ax1.set_xlabel('Frame')
                self.ax1.set_ylabel('Strain XX')



        if ind_type == "Strain YY":
            self.resetPlot()
            if self.timeSerise == False:
                self.ax1.imshow(self.curImg, cmap='gray')
                sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.valid_triangles, self.strainField[:,  1], cmap=cm.jet, alpha=0.5)
                min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.strainField[:, 1])
                sc.set_clim(min_confidence, max_confidence)     
                self.ax1.set_title('Strain YY')
                self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)

            else: 
                self.ax1.plot(self.strainTimeSeries[:, 1])
                self.ax1.set_title('Strain YY Time Series')
                self.ax1.set_xlabel('Frame')
                self.ax1.set_ylabel('Strain YY')


        if ind_type == "Strain XY":
            self.resetPlot()

            if self.timeSerise == False:
                self.ax1.imshow(self.curImg, cmap='gray')
                sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.valid_triangles, self.strainField[:,  2], cmap=cm.jet, alpha=0.5)
                min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.strainField[:, 2])
                sc.set_clim(min_confidence, max_confidence)          
                self.ax1.set_title('Strain XY')
                self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)

            else:
                self.ax1.plot(self.strainTimeSeries[:, 2])
                self.ax1.set_title('Strain XY Time Series')
                self.ax1.set_xlabel('Frame')
                self.ax1.set_ylabel('Strain XY')

        if self.node.configWidget.method == "3D DIC":
            if ind_type == "Displacement Z (mm)":
                self.resetPlot()
                if self.timeSerise == False:
                    self.ax1.imshow(self.curImg, cmap='gray')

                    sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.valid_triangles, self.displacementField[:,  2], cmap=cm.jet, alpha=0.5)

                    min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.displacementField[:, 2])
                    sc.set_clim(min_confidence, max_confidence)
                    self.ax1.set_title('Displacement Z')
                    self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)
                
                else:
                    self.ax1.plot(self.displacementTimeSeries[:, 2])
                    self.ax1.set_title('Displacement Z Time Series')
                    self.ax1.set_xlabel('Frame')
                    self.ax1.set_ylabel('Displacement Z (mm)')

            if ind_type == "Strain ZZ":
                self.resetPlot()
                if self.timeSerise == False:
                    self.ax1.imshow(self.curImg, cmap='gray')
                    sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.valid_triangles, self.strainField[:,  3], cmap=cm.jet, alpha=0.5)
                    min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.strainField[:, 3])
                    sc.set_clim(min_confidence, max_confidence)
                    self.ax1.set_title('Strain ZZ')
                    self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)

                else:
                    self.ax1.plot(self.strainTimeSeries[:, 3])
                    self.ax1.set_title('Strain ZZ Time Series')
                    self.ax1.set_xlabel('Frame')
                    self.ax1.set_ylabel('Strain ZZ')
            
            if ind_type == "Strain XZ":
                self.resetPlot()
                if self.timeSerise == False:
                    self.ax1.imshow(self.curImg, cmap='gray')
                    sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.strainField[:,  4], cmap=cm.jet, alpha=0.5)
                    min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.strainField[:, 4])
                    sc.set_clim(min_confidence, max_confidence)
                    self.ax1.set_title('Strain XZ')
                    self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)

                else:
                    self.ax1.plot(self.strainTimeSeries[:, 4])
                    self.ax1.set_title('Strain XZ Time Series')
                    self.ax1.set_xlabel('Frame')
                    self.ax1.set_ylabel('Strain XZ')
            
            if ind_type == "Strain YZ":
                self.resetPlot()
                if self.timeSerise == False:
                    self.ax1.imshow(self.curImg, cmap='gray')
                    sc = self.ax1.tripcolor(self.currentPoint[:,  0], self.currentPoint[:,  1], self.strainField[:,  5], cmap=cm.jet, alpha=0.5)
                    min_confidence, max_confidence = self.node.postprocessDIC.findConfidence(self.strainField[:, 5])
                    sc.set_clim(min_confidence, max_confidence)
                    self.ax1.set_title('Strain YZ')
                    self.colorbar = self.ax1.figure.colorbar(sc, ax=self.ax1)

                else:
                    self.ax1.plot(self.strainTimeSeries[:, 5])
                    self.ax1.set_title('Strain YZ Time Series')
                    self.ax1.set_xlabel('Frame')
                    self.ax1.set_ylabel('Strain YZ')

        if self.timeSerise == False:
            self.ax1.set_facecolor('#666')
            self.figure.patch.set_facecolor('#666')
        else:
            self.ax1.axis('on')
            self.ax1.set_facecolor('#fff')
            self.figure.patch.set_facecolor('#fff') 
        self.canvas.draw()

    





    