"""KreatifBot başlatıcı.

Kullanım:
    python main.py              -> uygulamayı açar
    python main.py --self-test  -> pencereyi açıp kapatır (kurulum/derleme kontrolü)
"""

import os
import sys
import traceback

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide6")


def main() -> int:
    self_test = "--self-test" in sys.argv

    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox

    from kreatifbot.config import setup_logging
    from kreatifbot.gui.main_window import MainWindow, apply_dark_theme

    log_file = setup_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("KreatifBot")
    apply_dark_theme(app)

    def excepthook(exc_type, exc, tb):
        import logging
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        logging.getLogger("kreatifbot").error("Yakalanmamış hata:\n%s", text)
        if not self_test:
            QMessageBox.critical(None, "Beklenmeyen hata", f"{exc}\n\nAyrıntılar: {log_file}")

    sys.excepthook = excepthook

    window = MainWindow(prompt_api=not self_test)
    window.show()
    if self_test:
        QTimer.singleShot(1500, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
