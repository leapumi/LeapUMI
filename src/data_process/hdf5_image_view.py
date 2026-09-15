import sys
import os
import h5py
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QLineEdit, QMessageBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class HDF5Viewer(QWidget):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.h5_file_name = os.path.basename(h5_path)

        try:
            self.h5 = h5py.File(h5_path, 'r')
        except Exception as e:
            QMessageBox.critical(self, "错误", f"HDF5 文件无法打开：{e}")
            sys.exit(1)

        # 获取 demo 列表
        self.demo_names = list(self.h5['data'].keys())
        self.demo_names.sort()
        self.current_demo = self.demo_names[0]

        self.idx = 0
        self.init_ui()
        self.load_demo(self.current_demo)

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 顶部：demo 选择区域
        hbox_demo = QHBoxLayout()
        hbox_demo.addWidget(QLabel("demo:"))
        self.demo_input = QLineEdit()
        self.demo_input.setText(self.current_demo)
        self.demo_input.returnPressed.connect(self.on_demo_input)
        hbox_demo.addWidget(self.demo_input)
        layout.addLayout(hbox_demo)

        # 图像显示区域
        self.fig = Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)

        # 信息显示区域
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-family: monospace;")
        layout.addWidget(self.info_label)

        # 底部控件（滑动条 + 输入框）
        hbox = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider)
        hbox.addWidget(QLabel("frame_idx:"))
        hbox.addWidget(self.slider)

        self.idx_input = QLineEdit()
        self.idx_input.setFixedWidth(80)
        self.idx_input.returnPressed.connect(self.on_idx_input)
        hbox.addWidget(self.idx_input)
        layout.addLayout(hbox)

    def load_demo(self, demo_name):
        demo_path = f"data/{demo_name}"
        try:
            self.time = self.h5[f"{demo_path}/obs/time"][()]
            self.agentview_image = self.h5[f"{demo_path}/obs/agentview_image"][()]
            self.arm_qpos = self.h5[f"{demo_path}/obs/arm_qpos"][()]
            self.eef_pos = self.h5[f"{demo_path}/obs/eef_pos"][()]
            self.gripper_qpos = self.h5[f"{demo_path}/obs/gripper_qpos"][()]
            self.actions = self.h5[f"{demo_path}/actions"][()]
            # self.traj_pos = self.h5[f"{demo_path}/obs/traj_pos"][()]
            # self.traj_quat = self.h5[f"{demo_path}/obs/traj_quat"][()]
        except KeyError as e:
            QMessageBox.critical(self, "数据错误", f"无法加载 demo '{demo_name}': {e}")
            return

        # 加载 eef_quat 或 eef_rot
        try:
            self.eef_rot_or_quat = self.h5[f"{demo_path}/obs/eef_quat"][()]
            self.rotation_field = "eef_quat"
        except KeyError:
            try:
                self.eef_rot_or_quat = self.h5[f"{demo_path}/obs/eef_rot"][()]
                self.rotation_field = "eef_rot"
            except KeyError as e:
                QMessageBox.critical(
                    self, "缺失字段", f"未找到 'eef_quat' 或 'eef_rot': {e}")
                return

        self.current_demo = demo_name
        self.idx = 0
        self.slider.setMaximum(len(self.time) - 1)
        self.slider.setValue(0)
        self.update_view()

    def on_demo_input(self):
        demo_name = self.demo_input.text()
        if demo_name in self.demo_names:
            self.load_demo(demo_name)
        else:
            QMessageBox.warning(
                self, "无效 demo",
                f"'{demo_name}' 不在可用 demo 列表中。\n可选 demo: {self.demo_names}"
            )

    def format_array(self, arr, precision=3, width=7):
        return "[" + "  ".join(f"{v:{width}.{precision}f}" for v in arr) + "]"

    def format_multiline_array(self, arr, splits, precision=3, width=7):
        lines = ["["]
        start = 0
        for s in splits:
            segment = arr[start:start + s]
            line = "  " + \
                "  ".join(f"{v:{width}.{precision}f}" for v in segment)
            lines.append(line)
            start += s
        lines.append("]")
        return "\n".join(lines)

    def update_view(self):
        idx = self.idx

        # 图像更新
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        img = self.agentview_image[idx]
        if img.ndim == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
        self.canvas.draw()

        # 文本信息更新
        info = (
            f"file_name: {self.h5_file_name}\n"
            f"demo: {self.current_demo}\n"
            f"frame_idx: {idx}    obs/time: {self.time[idx]:.3f}\n\n"
            f" obs/arm_qpos      : {self.format_array(self.arm_qpos[idx])}\n"
            f" obs/eef_pos       : {self.format_array(self.eef_pos[idx])}\n"
            f" obs/{self.rotation_field:<14}: {self.format_array(self.eef_rot_or_quat[idx])}\n"
            f" obs/gripper_qpos  : {self.format_array(self.gripper_qpos[idx], precision=2, width=6)}\n\n"
            f"actions:\n{self.format_multiline_array(self.actions[idx], splits=[3, 4, 16])}\n"
            # f"traj_pos: {self.format_array(self.traj_pos[idx])}\n"
            # f"traj_quat: {self.format_array(self.traj_quat[idx])}"
        )

        self.info_label.setText(info)
        self.idx_input.setText(str(idx))
        self.setWindowTitle(f"HDF5 Viewer - {self.h5_file_name}")

    def on_slider(self, value):
        self.idx = value
        self.update_view()

    def on_idx_input(self):
        try:
            idx = int(self.idx_input.text())
            idx = max(0, min(idx, len(self.time) - 1))
            self.idx = idx
            self.slider.setValue(idx)
            self.update_view()
        except ValueError:
            pass


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python hdf5_viewer.py <hdf5文件路径>")
        sys.exit(1)

    app = QApplication(sys.argv)
    viewer = HDF5Viewer(sys.argv[1])
    viewer.show()
    sys.exit(app.exec_())
