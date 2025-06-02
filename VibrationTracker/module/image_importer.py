import PyQt5.QtWidgets as QtGui
import os, sys
from PyQt5.QtWidgets import QApplication
import glob
import json
from natsort import natsorted

class ImageImporter:
    """
    A class to handle image import, saving image paths as JSON, and reading from JSON.
    Supports selecting directories and automatically fetching image files.
    """

    def openFolderDialog(self):
        """
        Opens a folder selection dialog for the user to select a directory.
        Returns:
            str: The selected folder path.
        """
        startingDir = './'
        dialog = QtGui.QFileDialog()
        dialog.setFileMode(QtGui.QFileDialog.FileMode())
        self.folderPath = dialog.getExistingDirectory(None, 'Open working directory', startingDir)
        return self.folderPath

    def readImagesNamesFromFolder(self, folderPath):
        """
        Reads image filenames with .bmp, .tif, and .tiff extensions from the specified folder.
        Automatically sorts filenames naturally.
        
        Args:
            folderPath (str): Path to the folder containing images.
        
        Returns:
            list: Sorted list of image file paths.
        """
        imageNames = glob.glob(folderPath + '/*.bmp') + glob.glob(folderPath + '/*.tif') + glob.glob(folderPath + '/*.tiff')
        imageNames = natsorted(imageNames)
        return imageNames 
    
    def saveAsJson(self, imagesNames, resultFolderPath):
        """
        Saves a list of image file paths into a JSON file.
        
        Args:
            imagesNames (list): List of image file paths.
            resultFolderPath (str): Path where the JSON file will be saved.
        """
        self.outputName = os.path.join(resultFolderPath, 'imagesNames.json')
        with open(self.outputName, 'w') as f:
            json.dump(imagesNames, f)

    def runImageImport(self, resultFolderPath):
        """
        Runs the full image import process:
        1. Opens folder selection dialog.
        2. Reads image filenames from the folder.
        3. Saves filenames into a JSON file.
        
        Args:
            resultFolderPath (str): Directory to save the JSON output.
        
        Returns:
            list: List of imported image filenames.
        """
        self.folderPath = self.openFolderDialog()
        imageNames = self.readImagesNamesFromFolder(self.folderPath)
        self.saveAsJson(imageNames, resultFolderPath)
        return imageNames
    
    def readFromJson(self, jsonPath):
        """
        Reads image filenames from a JSON file and prints them.
        
        Args:
            jsonPath (str): Path to the JSON file.
        
        Returns:
            list: List of image filenames read from the JSON file.
        """
        with open(jsonPath) as f:
            imagesNames = json.load(f)
            print('imagesNames: ', imagesNames)
        return imagesNames
        

if __name__ == '__main__':
    """
    Entry point for the script. Initializes the QApplication and starts the image import process.
    """
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = ImageImporter()
    folderPath = ex.runImageImport('./images.json')
    sys.exit(app.exec_())
