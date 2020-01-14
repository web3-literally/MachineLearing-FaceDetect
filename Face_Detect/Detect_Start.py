from PyQt5.QtWidgets import QApplication
from Main.MainWnd import MainWnd
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWnd()
    window.show()
    sys.exit(app.exec())

