import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from VibrationTracker.vib_window import VibrationTrackerWindow


if hasattr(Qt, 'AA_EnableHighDpiScaling'):
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # enable highdpi scaling

if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    # use highdpi icons
    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    wnd = VibrationTrackerWindow()
    wnd.show()

    sys.exit(app.exec_())