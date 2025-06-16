from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit,
    QHBoxLayout, QTextBrowser, QFileDialog, QDialog, QTextEdit
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer, QSize
from PySide6.QtGui import QFont, QPalette, QColor, QMovie
import sys
import subprocess
import os
import re
import psutil
import shutil

# ──────────────── Background Worker ────────────────
class PredictorThread(QThread):
    update_text = Signal(str)
    update_done = Signal(str)

    def __init__(self, address):
        super().__init__()
        self.address = address

    def run(self):
        try:
            process = subprocess.Popen(
                ["python3", "predict.py"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            process.stdin.write(self.address + "\n")
            process.stdin.flush()

            for line in iter(process.stdout.readline, ''):
                if line:
                    self.update_text.emit(line.rstrip())

            if process.poll() is None:
                try:
                    process.stdin.write("exit\n")
                    process.stdin.flush()
                except Exception:
                    pass

            process.terminate()
            self.update_done.emit(f"✅ Завершено: {self.address}")

        except Exception as e:
            self.update_text.emit(f"[Error] {str(e)}")
            self.update_done.emit("")

# ──────────────── Trainer Worker ────────────────
class TrainerThread(QThread):
    log_update = Signal(str)

    def run(self):
        try:
            process = subprocess.Popen(
                ["python3", "train.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            for line in process.stdout:
                self.log_update.emit(line.strip())
        except Exception as e:
            self.log_update.emit(f"[ERROR] {str(e)}")

# ──────────────── Training Dialog ────────────────
class TrainingDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" ")
        self.setMinimumSize(600, 400)
        layout = QVBoxLayout()

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background-color: #ffb6c1; color: #2a2a2a; font-family: Consolas; font-size: 12px;")
        layout.addWidget(self.log_box)

        self.cpu_gpu_label = QLabel("CPU: --%, GPU: --%")
        self.cpu_gpu_label.setStyleSheet("color: #2a2a2a")
        layout.addWidget(self.cpu_gpu_label)

        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_system_usage)
        self.timer.start(1000)

        self.thread = TrainerThread()
        self.thread.log_update.connect(self.log_box.append)
        self.thread.start()

    def update_system_usage(self):
        cpu = psutil.cpu_percent()
        gpu = "?"
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = f"{gpus[0].load * 100:.1f}%"
        except:
            gpu = "n/a"
        self.cpu_gpu_label.setText(f"CPU: {cpu:.1f}%, GPU: {gpu}")


class PredictorGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(" ")
        self.showMaximized()
        self.thread = None
        self.address_list = []
        self.current_index = 0
        self.running = False

        self.setStyleSheet("""
            QWidget {
                background-color: #ffb6c1;
            }
            QPushButton {
                background-color: #7f5af0;
                color: white;
                border-radius: 6px;
                padding: 6px;
                min-width: 120px;
            }
            QPushButton:pressed {
                background-color: #5e3fc9;
            }
            QLineEdit {
                background-color: white;
                color: #2a2a2a;
                border-radius: 6px;
                padding: 6px;
            }
            QPushButton#genButton {
                background-color: #3e005f;
                color: #ffb6c1;
            }
            QPushButton#genButton:pressed {
                background-color: #2a0047;
            }
        """)

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#ffb6c1"))
        self.setPalette(palette)

        self.main_layout = QVBoxLayout()
        self.input_layout = QHBoxLayout()
        self.console_layout = QVBoxLayout()
        self.footer_layout = QHBoxLayout()

        self.input = QLineEdit()
        self.input.setPlaceholderText("Вставьте адрес")
        self.input_layout.addWidget(self.input)

        self.button = QPushButton("Дешифровать")
        self.button.clicked.connect(self.start_single)
        self.input_layout.addWidget(self.button)

        self.load_button = QPushButton("Загрузить .txt")
        self.load_button.clicked.connect(self.load_addresses_from_file)
        self.input_layout.addWidget(self.load_button)

        self.status_line = QTextBrowser()
        self.status_line.setAlignment(Qt.AlignLeft)
        self.status_line.setReadOnly(True)
        self.status_line.setFont(QFont("Consolas", 10))
        self.status_line.setStyleSheet("background-color: white; color: #2a2a2a; border-radius: 6px; padding: 6px;")
        self.status_line.setMinimumHeight(400)

        self.spinner = QLabel()
        self.spinner.setFixedSize(QSize(24, 24))
        self.spinner.setAlignment(Qt.AlignCenter)
        self.spinner_movie = QMovie("spinner.gif")
        self.spinner.setMovie(self.spinner_movie)
        self.spinner.hide()

        self.train_button = QPushButton("Доучить модель")
        self.train_button.clicked.connect(self.launch_trainer)
        self.footer_layout.addWidget(self.train_button)

        self.hard_reparse_btn = QPushButton("Hard Reparse")
        self.hard_reparse_btn.clicked.connect(self.hard_reparse)
        self.footer_layout.addWidget(self.hard_reparse_btn)

        self.gen_button = QPushButton("Gen")
        self.gen_button.setObjectName("genButton")
        self.gen_button.clicked.connect(self.toggle_gen_mode)
        self.footer_layout.addWidget(self.gen_button)

        self.stop_button = QPushButton("СТОП")
        self.stop_button.setStyleSheet("background-color: #ff3c3c; color: white; border-radius: 6px; padding: 6px;")
        self.stop_button.clicked.connect(self.stop_prediction)
        self.footer_layout.addWidget(self.stop_button)

        self.footer_layout.addStretch()
        self.footer_layout.addWidget(self.spinner)

        self.console_layout.addWidget(self.status_line)
        self.main_layout.addLayout(self.input_layout)
        self.main_layout.addLayout(self.console_layout)
        self.main_layout.addLayout(self.footer_layout)
        self.setLayout(self.main_layout)

        self.setup_gen_timer()

    def setup_gen_timer(self):
        self.gen_timer = QTimer()
        self.gen_timer.timeout.connect(self.safe_hard_reparse)

    def toggle_gen_mode(self):
        if self.gen_timer.isActive():
            self.gen_timer.stop()
            self.update_status_text("Генерация остановлена.")
            self.gen_button.setText("Gen")
        else:
            self.gen_timer.start(10000)
            self.update_status_text("Генерация активна.")
            self.gen_button.setText("Gen")
            self.safe_hard_reparse()

    def update_status_text(self, text):
        self.status_line.append(f"<b>{text}</b>" if "КОПИЯ СДЕЛАНА" in text else text)
        self.status_line.verticalScrollBar().setValue(self.status_line.verticalScrollBar().maximum())

    def start_single(self):
        address = self.input.text().strip()
        if not address:
            return
        self.address_list = [address]
        self.current_index = 0
        self.run_next()

    def run_next(self):
        if self.thread and self.thread.isRunning():
            self.update_status_text("Ожидание завершения предыдущего потока...")
            QTimer.singleShot(500, self.run_next)
            return

        if self.current_index >= len(self.address_list):
            self.spinner.hide()
            self.spinner_movie.stop()
            self.running = False
            return

        self.running = True
        address = self.address_list[self.current_index]
        self.update_status_text(f"Проверяю: {address}")
        self.thread = PredictorThread(address)
        self.thread.update_text.connect(self.update_status_text)
        self.thread.update_done.connect(self.handle_next)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.start()
        self.spinner.show()
        self.spinner_movie.start()

    def handle_next(self, summary):
        if summary:
            self.update_status_text(summary)
        self.current_index += 1
        QTimer.singleShot(100, self.run_next)

    def cleanup_thread(self):
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None

    def load_addresses_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Выбрать .txt", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, "r") as f:
                self.address_list = [line.strip() for line in f if line.strip()]
            self.current_index = 0
            self.run_next()

    def stop_prediction(self):
        if self.thread and self.thread.isRunning():
            self.thread.terminate()
            self.thread.wait()
            self.update_status_text("Процесс остановлен.")
        self.spinner.hide()
        self.spinner_movie.stop()
        self.thread = None
        self.address_list = []
        self.current_index = 0
        self.running = False

    def safe_hard_reparse(self):
        if self.running:
            self.update_status_text("Ожидание завершения предыдущего потока...")
            return
        self.hard_reparse()

    def hard_reparse(self):
        try:
            shutil.copy("Data/attemptsnn.txt", "Data/workplacecopy.txt")
            self.update_status_text("<КОПИЯ СДЕЛАНА>")
        except Exception as e:
            self.update_status_text(f"[Ошибка копирования] {str(e)}")
            return

        self.address_list = []
        try:
            with open("Data/workplacecopy.txt", "r", encoding="utf-8") as f:
                for line in f:
                    match = re.search(r"Address:\s*([a-zA-Z0-9x]+)", line)
                    if match:
                        self.address_list.append(match.group(1))
        except Exception as e:
            self.update_status_text(f"[Ошибка чтения workplacecopy.txt] {str(e)}")
            return

        self.update_status_text(f"Найдено {len(self.address_list)} адресов для Reparse")
        self.current_index = 0
        self.run_next()

    def launch_trainer(self):
        dialog = TrainingDialog()
        dialog.exec()

    def closeEvent(self, event):
        if self.thread:
            if self.thread.isRunning():
                self.thread.quit()
                self.thread.wait()
            self.thread = None

        if hasattr(self, 'gen_timer') and self.gen_timer.isActive():
            self.gen_timer.stop()

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictorGUI()
    window.show()
    sys.exit(app.exec())