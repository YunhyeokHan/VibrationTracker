import os
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import QAbstractScrollArea, QWidget, QDockWidget, QAction, QMessageBox, QFileDialog
from PyQt5.QtCore import Qt
from nodeeditor.utils import loadStylesheets
from nodeeditor.node_editor_window import NodeEditorWindow
from nodeeditor.node_node import Node
from nodeeditor.utils import dumpException, pp

from VibrationTracker.vib_sub_window import VibrationTrakerSubWindow
from VibrationTracker.vib_drag_listbox import QDMDragListbox

from VibrationTracker.vib_conf import VIB_NODES

# Enabling edge validators
from nodeeditor.node_edge import Edge
from nodeeditor.node_edge_validators import (
    edge_validator_debug,
    edge_cannot_connect_two_outputs_or_two_inputs,
    edge_cannot_connect_input_and_output_of_same_node
)
Edge.registerEdgeValidator(edge_validator_debug)
Edge.registerEdgeValidator(edge_cannot_connect_two_outputs_or_two_inputs)
Edge.registerEdgeValidator(edge_cannot_connect_input_and_output_of_same_node)

# images for the dark skin
import VibrationTracker.qss.nodeeditor_dark_resources

DEBUG = False

class VibrationTrackerWindow(NodeEditorWindow):

    def initUI(self):

        self.setStyleSheet(''' font-size: 14px; ''')

        self.name_company = 'ISAE-SUPMECA'
        self.name_product = 'Vibration Tracker'

        self.stylesheet_filename = os.path.join(os.path.dirname(__file__), "qss/nodeeditor.qss")
        
        loadStylesheets(
            os.path.join(os.path.dirname(__file__), "qss/nodeeditor-dark.qss"),
            self.stylesheet_filename
        )

        if DEBUG:
            print("Registered nodes:")
            pp(VIB_NODES)

        self.widget = VibrationTrakerSubWindow()
        self.setCentralWidget(self.widget)

        self.default_title = "Vibration Tracker"

        self.setTitle()

        self.createFileDialogDock()
        self.createmainDock()
        self.createNodeConfigDock()
        self.createNodesDock()

        self.createActions()
        self.createMenus()
        self.createToolBars()
        self.createStatusBar()

        self.updateMenus()

        self.readSettings()
        
        self.onFileOpen()

        self.widget.scene.addHasBeenModifiedListener(self.setTitle)
        self.widget.scene.addItemSelectedListener(self.onItemSelectedClicked)
    
    def createActions(self):
        super().createActions()
        self.actAbout = QAction("&About", self, statusTip="Show the application's About box", triggered=self.about)

    def setTitle(self):
        """Function responsible for setting window title"""
        title = self.default_title + " - "
        title += self.getCurrentNodeEditorWidget().getUserFriendlyFilename()

        self.setWindowTitle(title)
    
    def onFileNew(self):
        try:
            if self.maybeSave():
                self.widget.fileNew()
                self.setTitle()
            
        except Exception as e: dumpException(e)

    def createResultFolder(self, filename):
        print(os.path.dirname(filename))
        result_folder = os.path.join(os.path.dirname(filename), os.path.splitext(os.path.basename(filename))[0])
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

    def onFileSave(self):
        super().onFileSave()
        self.setTitle()

    def onFileSaveAs(self):
        super().onFileSaveAs()
        try:
            self.createResultFolder(self.getCurrentNodeEditorWidget().filename)
            self.setTitle()
        except Exception as e: dumpException(e)

    def onFileOpen(self):
        fname, filter = QFileDialog.getOpenFileName(self, 'Open graph from file', self.getFileDialogDirectory(), self.getFileDialogFilter())
        try:
            if self.widget.fileLoad(fname):
                self.statusBar().showMessage("File %s loaded" % fname, 5000)
                self.widget.show()
                self.createResultFolder(fname)
                self.setTitle()
            else:
                self.onFileSaveAs()

        except Exception as e: dumpException(e)


    def about(self):
        QMessageBox.about(self, "About Vibration Tracker",
                "The <b>Vibration Tracker</b> is a utility for measuring vibrations using image data"
                "The application provides various tools for vibration measurement with cameras "
                "It includes camera calibration, ... "
                "This application is based on the <b>NodeEditor</b> library, <b>OpenCV</b> and <b>PyQt5</b>.")
                # "<a href='https://www.blenderfreak.com/'>www.BlenderFreak.com</a>")

    def createMenus(self):
        super().createMenus()

        self.windowMenu = self.menuBar().addMenu("&Window")
        self.updateWindowMenu()
        self.windowMenu.aboutToShow.connect(self.updateWindowMenu)

        self.menuBar().addSeparator()
        self.helpMenu = self.menuBar().addMenu("&Help")
        self.helpMenu.addAction(self.actAbout)

        self.editMenu.aboutToShow.connect(self.updateEditMenu)

    def updateMenus(self):
        # print("update Menus")
        active = self.getCurrentNodeEditorWidget()
        hasMdiChild = (active is not None)

        self.actSave.setEnabled(hasMdiChild)
        self.actSaveAs.setEnabled(hasMdiChild)

        self.updateEditMenu()

    def updateEditMenu(self):
        try:
            # print("update Edit Menu")
            active = self.getCurrentNodeEditorWidget()
            hasMdiChild = (active is not None)

            self.actPaste.setEnabled(hasMdiChild)

            self.actCut.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actCopy.setEnabled(hasMdiChild and active.hasSelectedItems())
            self.actDelete.setEnabled(hasMdiChild and active.hasSelectedItems())

            self.actUndo.setEnabled(hasMdiChild and active.canUndo())
            self.actRedo.setEnabled(hasMdiChild and active.canRedo())
        except Exception as e: dumpException(e)

    def updateWindowMenu(self):
        self.windowMenu.clear()

        toolbar_nodes = self.windowMenu.addAction("Nodes List")
        toolbar_nodes.setCheckable(True)
        toolbar_nodes.triggered.connect(self.onWindowNodesToolbar)
        toolbar_nodes.setChecked(self.nodesDock.isVisible())        

        toolbar_file_dialog = self.windowMenu.addAction("File Dialog")
        toolbar_file_dialog.setCheckable(True)
        toolbar_file_dialog.triggered.connect(self.onWindowFileDialogToolbar)
        toolbar_file_dialog.setChecked(self.fileDialogDock.isVisible())

        toolbar_main = self.windowMenu.addAction("Current State")
        toolbar_main.setCheckable(True)
        toolbar_main.triggered.connect(self.onWindowMainToolbar)
        toolbar_main.setChecked(self.mainDock.isVisible())

        toolbar_node_config = self.windowMenu.addAction("Node Config")
        toolbar_node_config.setCheckable(True)
        toolbar_node_config.triggered.connect(self.onWindowNodeConfigToolbar)
        toolbar_node_config.setChecked(self.nodeConfigDock.isVisible())
    
    def onWindowFileDialogToolbar(self):
        if self.fileDialogDock.isVisible():
            self.fileDialogDock.hide()
        else:
            self.fileDialogDock.show()
    
    def onWindowMainToolbar(self):
        if self.mainDock.isVisible():
            self.mainDock.hide()
        else:
            self.mainDock.show()
    
    def onWindowNodeConfigToolbar(self):
        if self.nodeConfigDock.isVisible():
            self.nodeConfigDock.hide()
        else:
            self.nodeConfigDock.show()


    def onWindowNodesToolbar(self):
        if self.nodesDock.isVisible():
            self.nodesDock.hide()
        else:
            self.nodesDock.show()

    def createToolBars(self):
        pass

    def createNodesDock(self):
        self.nodesListWidget = QDMDragListbox()

        self.nodesDock = QDockWidget("Nodes")
        self.nodesDock.setWidget(self.nodesListWidget)
        self.nodesDock.setFloating(False)
        # self.nodesDock.setMinimumHeight(200)
        # self.nodesDock.setMinimumWidth(400)
        self.fileDialogDock.resize(200, 200)

        self.addDockWidget(Qt.RightDockWidgetArea, self.nodesDock)

    def createFileDialogDock(self):
        self.fileDialogWidget = QWidget()

        self.fileDialogDock = QDockWidget("File Dialog")
        self.fileDialogDock.setWidget(self.fileDialogWidget)
        self.fileDialogDock.setFloating(False)
        self.fileDialogDock.setMinimumHeight(200)
        self.fileDialogDock.setMinimumWidth(200)
        self.fileDialogDock.setMaximumWidth(200)


        self.addDockWidget(Qt.TopDockWidgetArea, self.fileDialogDock)

    def createmainDock(self):
        self.mainWidget = QWidget()
        self.mainDock = QDockWidget("Current State")

        self.mainDock.setWidget(self.mainWidget)
        self.mainDock.setFloating(False)

        self.mainDock.setMinimumHeight(200)
        self.mainDock.setMinimumWidth(200)
        self.addDockWidget(Qt.TopDockWidgetArea, self.mainDock)

    def createNodeConfigDock(self):
        self.nodeConfigWidget = QWidget()

        self.nodeConfigDock = QDockWidget("Node Config")
        self.nodeConfigDock.setWidget(self.nodeConfigWidget)
        self.nodeConfigDock.setFloating(False)
        self.nodeConfigDock.setMinimumHeight(200)
        self.nodeConfigDock.setMinimumWidth(250)

        self.addDockWidget(Qt.TopDockWidgetArea, self.nodeConfigDock)


    def updatemainDock(self):
        if self.mainDock.isVisible():
            currentHeight  = self.mainDock.height()
            currentWidth = self.mainDock.width()
            currentPos = self.mainDock.pos()
            dockedArea = self.dockWidgetArea(self.mainDock)
            isFloating  = self.mainDock.isFloating()
            newmainWidget  = self.widget.getSelectedItems()[0].node.mainWidget


            if isFloating:
                if self.mainWidget:
                    self.mainWidget.setParent(None)
                self.mainWidget = newmainWidget
                self.mainDock.setWidget(self.mainWidget)

            else:
                self.mainDock.close()
                self.mainWidget.close()
                self.mainDock = QDockWidget("Current State")
                self.mainWidget = newmainWidget
                self.mainDock.setWidget(self.mainWidget)


            self.mainDock.resize(currentWidth, currentHeight)

            if isFloating:
                self.mainDock.setFloating(True)
                self.mainDock.show()
                self.mainDock.move(currentPos)
            else:
                self.addDockWidget(dockedArea, self.mainDock)

            print("Node Selected and update Main Dock")
        else:
            pass        

    def updateNodeConfigDock(self):

        if self.nodeConfigDock.isVisible():

                
            currentHeight  = self.nodeConfigDock.height()
            currentWidth = self.nodeConfigDock.width()
            currentPos = self.nodeConfigDock.pos()
            dockedArea = self.dockWidgetArea(self.nodeConfigDock)
            isFloating  = self.nodeConfigDock.isFloating()

            newnodeConfigWidget = self.widget.getSelectedItems()[0].node.configWidget

            if isFloating:
                if self.nodeConfigWidget:
                    self.nodeConfigWidget.setParent(None)
                    self.nodeConfigWidget = newnodeConfigWidget
                    self.nodeConfigDock.setWidget(self.nodeConfigWidget)

            else:
                self.nodeConfigDock.close()
                self.nodeConfigWidget.close()
                self.nodeConfigDock = QDockWidget("Node Config")
                self.nodeConfigWidget = newnodeConfigWidget
                self.nodeConfigDock.setWidget(self.nodeConfigWidget)
                
           
            self.nodeConfigDock.setFloating(False)
            self.nodeConfigDock.resize(currentWidth, currentHeight)

            if isFloating:
                self.nodeConfigDock.setFloating(True)
                self.nodeConfigDock.show()
                self.nodeConfigDock.move(currentPos)
            else:
                self.addDockWidget(dockedArea, self.nodeConfigDock)
            print("Node Selected and update Node Config Dock")
        else:
            pass
    def createStatusBar(self):
        self.statusBar().showMessage("Ready")


    def onItemSelectedClicked(self):

        if len(self.widget.getSelectedItems()) > 0:
            selected_item = self.widget.getSelectedItems()[0]
            if hasattr(selected_item, 'node'):
                print(selected_item.node.id)
                self.updatemainDock()
                self.updateNodeConfigDock()
        else:
            print("No item selected")

