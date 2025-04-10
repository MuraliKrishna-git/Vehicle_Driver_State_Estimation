import sys
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class LivePlotWindow(QtWidgets.QWidget):
    def __init__(self, live_data):
        super().__init__()
        self.live_data = live_data
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Real-Time Heart Rate Monitor")
        layout = QtWidgets.QVBoxLayout()

        self.plot_bpm = pg.PlotWidget(title="BPM Over Time")
        self.plot_bpm.setLabel('left', 'BPM')
        self.plot_bpm.setLabel('bottom', 'Time (s)')
        self.curve_bpm = self.plot_bpm.plot(pen=pg.mkPen('g', width=2))
        layout.addWidget(self.plot_bpm)

        self.plot_angle = pg.PlotWidget(title="Face Angle Over Time")
        self.plot_angle.setLabel('left', 'Angle (°)')
        self.plot_angle.setLabel('bottom', 'Time (s)')
        self.curve_angle = self.plot_angle.plot(pen=pg.mkPen('r', width=2))
        layout.addWidget(self.plot_angle)

        self.plot_green = pg.PlotWidget(title="Green Intensity Over Time")
        self.plot_green.setLabel('left', 'Green Intensity')
        self.plot_green.setLabel('bottom', 'Time (s)')
        self.curve_green = self.plot_green.plot(pen=pg.mkPen('b', width=2))
        layout.addWidget(self.plot_green)

        self.plot_green_bpm = pg.PlotWidget(title="Green Intensity vs. BPM (Scatter)")
        self.plot_green_bpm.setLabel('left', 'BPM')
        self.plot_green_bpm.setLabel('bottom', 'Green Intensity')
        self.scatter_green_bpm = pg.ScatterPlotItem()
        self.plot_green_bpm.addItem(self.scatter_green_bpm)
        layout.addWidget(self.plot_green_bpm)

        self.setLayout(layout)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(100)

    def update_plots(self):
        times = self.live_data["time"]
        bpms = self.live_data["bpm"]
        angles = self.live_data["face_angle"]
        green_values = self.live_data["green_intensity"]

        min_length = min(len(times), len(bpms), len(angles), len(green_values))

        times = times[:min_length]
        bpms = bpms[:min_length]
        angles = angles[:min_length]
        green_values = green_values[:min_length]

        self.curve_bpm.setData(times, bpms)
        self.curve_angle.setData(times, angles)
        self.curve_green.setData(times, green_values)

        if len(green_values) == len(bpms):
            scatter_points = [{'pos': (g, b), 'size': 5, 'brush': pg.mkBrush('m')} for g, b in zip(green_values, bpms)]
            self.scatter_green_bpm.setData(scatter_points)

def start_live_plot(live_data):
    app = QtWidgets.QApplication(sys.argv)
    win = LivePlotWindow(live_data)
    win.resize(800, 800)
    win.show()
    app.exec_()
