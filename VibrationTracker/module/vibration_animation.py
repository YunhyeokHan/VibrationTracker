import matplotlib.pyplot as plt
import os
import json
import numpy as np

class AnimateVibration:

    def createResultFolder(self, index = 0):
        currentWorkingDir = os.path.dirname(os.path.dirname(self.filePath))
        resultFolderPath = os.path.join(currentWorkingDir, "AnimateVibration_" + str(index))
        if not os.path.exists(resultFolderPath):
            os.makedirs(resultFolderPath)
        
        return resultFolderPath
    
    
    def readDisplacementdata(self, filePath):
        
        with open(filePath) as f:
            data = json.load(f)
        # check if the data is 3D or 2D
        key = list(data.keys())
        if key[1] == "Point3D":
            displacement = np.array(data["resultDisplacement"])
            
            position = np.array(data["Point3D"])

            position_all = np.array(position) + np.array(displacement)
            return position_all
        
        elif key[1] == "Point2D":
            displacement = np.array(data["resultDisplacement"])
            position = np.array(data["Point2D"])
            position_all = np.array(position) + np.array(displacement)
            return position_all
    
    

if __name__ == "__main__":
    animateVibration = AnimateVibration()
    animateVibration.filePath = r"C:\Users\hany\Desktop\DynaEyes\ToGitHub\example_assemblage\Displacement_2636658905632\displacementResults.json"
    animateVibration.createResultFolder(index = 0)
    position_all = animateVibration.readDisplacementdata(animateVibration.filePath)
    print("done")


    
