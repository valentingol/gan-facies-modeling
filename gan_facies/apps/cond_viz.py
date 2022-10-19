"""Visualizer for conditional models."""

import os.path as osp
import sys
from functools import partial

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QPushButton,
                             QRadioButton, QSlider, QWidget)

from gan_facies.data.process import to_img_grid
from gan_facies.gan.cond_sagan.modules import CondSAGenerator
from gan_facies.utils.conditioning import colorize_pixel_map
from gan_facies.utils.configs import ConfigType, GlobalConfig


class ConditionalVisualizer(QWidget):
    """Conditional Visualizer PyQT6 application.

    Visualize output of conditional model interactively.

    Parameters
    ----------
    data : numpy.ndarray
        Data of type uint8 with shape (z~depth~n_samples, y, x).
        Each different number represents a different class
        (starting from 0).
    """

    def __init__(self, generator: CondSAGenerator):
        super().__init__()
        n_classes, data_size = generator.n_classes, generator.data_size
        self.generator = generator.eval()
        self.n_classes = n_classes
        self.data_size = data_size

        self.pixels = torch.zeros(n_classes, data_size, data_size)
        self.pixels_color = colorize_pixel_map(self.pixels[None, ...])
        self.proba = np.zeros((data_size, data_size, 3), dtype=np.uint8)
        self.grid = np.zeros((data_size, data_size, 3), dtype=np.uint8)
        self.current_pixel_size = 6
        self.current_pixel_class = n_classes - 1

        self.init_ui()
        self.init_widgets()

    def init_ui(self) -> None:
        """Init UI."""
        self.render_size = 512
        self.setFixedSize(1836, 642)
        # Center the UI
        frame = self.frameGeometry()
        center_point = self.screen().availableGeometry().center()
        frame.moveCenter(center_point)
        self.move(frame.topLeft())

        self.setWindowTitle('Conditional visualizer')

    def init_widgets(self) -> None:
        """Init the widgets."""
        rsize = self.render_size

        # Pixel map
        self.pixel_map = QLabel(self)
        self.pixel_map.setPixmap(self.np2pixmap(self.pixels_color[1:-1, 1:-1]))
        self.set_img_location(self.pixel_map, 100, 30, rsize, rsize)
        self.pixel_map.mousePressEvent = self.add_pixel  # type: ignore
        self.pixel_map_label = QLabel(self)
        self.pixel_map_label.move(rsize//2, 10)
        self.pixel_map_label.setText("<font color='cyan'>"
                                     "Click to add conditional pixel"
                                     "</font>")

        # Sample grid
        self.sample_grid = QLabel(self)
        self.sample_grid.setPixmap(self.np2pixmap(self.grid))
        self.set_img_location(self.sample_grid, 150 + rsize, 30, rsize, rsize)
        self.sample_grid_label = QLabel(self)
        self.sample_grid_label.move(100 + 3*rsize//2, 10)
        self.sample_grid_label.setText("Grid of samples")

        # Probability map
        self.proba_map = QLabel(self)
        self.proba_map.setPixmap(self.np2pixmap(self.proba))
        self.set_img_location(self.proba_map, 200 + 2*rsize, 30, rsize, rsize)
        self.proba_map_label = QLabel(self)
        self.proba_map_label.move(150 + 5*rsize//2, 10)
        self.proba_map_label.setText("Probability map")

        # Size slider
        self.size_slider = QSlider(Qt.Orientation.Vertical, self)
        self.size_slider.setGeometry(30, rsize // 10, 40, 4 * rsize // 5)
        self.size_slider.setRange(1, 16)
        self.size_slider.setValue(6)
        self.size_slider.setTickInterval(1)
        self.size_slider.setTickPosition(QSlider.TickPosition.TicksRight)
        self.size_slider.valueChanged.connect(self.update_size)  # type: ignore
        self.size_slider_label = QLabel(self)
        self.size_slider_label.move(10, 10)
        self.size_slider_label.setText("Pixel size")
        self.size_slider_val_label = QLabel(self)
        self.size_slider_val_label.move(40, rsize - rsize // 10)
        self.size_slider_val_label.setText("6")
        self.size_slider_val_label.setMinimumWidth(80)

        # Class buttons
        class_buttons = QWidget(self)
        class_buttons.setFixedSize(rsize, 30)
        class_buttons.move(100, 40 + rsize)
        layout = QHBoxLayout(class_buttons)

        for i in range(self.n_classes):
            button = QRadioButton(str(i))
            button.setGeometry(10, 10, 20, 20)
            button.setChecked(i == self.n_classes - 1)
            connect_fn = partial(self.update_class, new_class=i)
            button.toggled.connect(connect_fn)  # type: ignore
            layout.addWidget(button)
            setattr(self, f"class_{i}_button", button)

        self.class_slider_label = QLabel(self)
        self.class_slider_label.move(80 + rsize//2, 80 + rsize)
        self.class_slider_label.setText("Pixel class")

        # Resample button
        self.resample_button = QPushButton('Resample', self)
        self.resample_button.move(120 + rsize, 40 + rsize)
        self.resample_button.clicked.connect(self.sample_new)  # type: ignore

        # Clear button
        self.clear_button = QPushButton('Clear', self)
        self.clear_button.move(230 + rsize, 40 + rsize)
        self.clear_button.clicked.connect(self.clear)  # type: ignore

    def add_pixel(self, QMouseEvent: QEvent) -> None:
        """Add pixel to the pixel map."""
        psize = self.current_pixel_size
        x, y = QMouseEvent.position().x(), QMouseEvent.position().y()
        x_pix = int(x / self.render_size * self.data_size)
        y_pix = int(y / self.render_size * self.data_size)
        x_0 = max(0, x_pix - psize//2)
        x_1 = min(self.data_size, x_pix + psize - psize//2)
        y_0 = max(0, y_pix - psize//2)
        y_1 = min(self.data_size, y_pix + psize - psize//2)
        # Update pixels (features are [is_cond, is_class1, ..., is_classN-1]
        # for class 0, we set all the features to 0 but is_cond)
        self.pixels[0, y_0:y_1, x_0:x_1] = 1
        self.pixels[1:, y_0:y_1, x_0:x_1] = 0
        self.pixels[self.current_pixel_class, y_0:y_1, x_0:x_1] = 1

        # Update pixels color
        self.pixels_color = colorize_pixel_map(self.pixels[None, ...])
        self.pixel_map.setPixmap(self.np2pixmap(self.pixels_color[1:-1, 1:-1]))

        self.sample_new()

    def update_size(self, value: int) -> None:
        """Update current pixel size."""
        self.current_pixel_size = value
        self.size_slider_val_label.setText(str(value))

    def update_class(self, new_class: int) -> None:
        """Update current pixel class."""
        self.current_pixel_class = new_class

    def sample_new(self) -> None:
        """Sample new images and update grid/proba map."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        z_input = torch.randn(512, self.generator.z_dim, device=device)

        # Update grid of samples
        pixels_for_grid = self.pixels.repeat(64, 1, 1, 1)
        images, _ = self.generator.generate(z_input[:64],
                                            pixels_for_grid.to(device))
        self.grid = to_img_grid(images)
        self.sample_grid.setPixmap(self.np2pixmap(self.grid))
        # Update probability map
        _, _, proba_color = self.generator.proba_map(z_input,
                                                     self.pixels.to(device),
                                                     batch_size=64)
        self.proba = proba_color
        self.proba = cv2.resize(self.proba,
                                (self.render_size, self.render_size),
                                interpolation=cv2.INTER_LINEAR)
        self.proba_map.setPixmap(self.np2pixmap(self.proba))

    def clear(self) -> None:
        """Clear maps."""
        data_size, n_classes = self.data_size, self.n_classes
        self.pixels = torch.zeros(n_classes, data_size, data_size)
        self.pixels_color = colorize_pixel_map(self.pixels[None, ...])
        self.proba = np.zeros((data_size, data_size, 3), dtype=np.uint8)
        self.grid = np.zeros((data_size, data_size, 3), dtype=np.uint8)
        self.pixel_map.setPixmap(self.np2pixmap(self.pixels_color[1:-1, 1:-1]))
        self.sample_grid.setPixmap(self.np2pixmap(self.grid))
        self.proba_map.setPixmap(self.np2pixmap(self.proba))

    @staticmethod
    def np2pixmap(np_arr: np.ndarray) -> QPixmap:
        """Convert a numpy array to a QPixmap."""
        img = Image.fromarray(np_arr.astype(np.uint8)).convert('RGB')
        img = np.array(img)
        height, width, _ = img.shape
        q_image = QImage(img.data, width, height, 3 * width,
                         QImage.Format.Format_RGB888)
        return QPixmap(q_image)

    @staticmethod
    def set_img_location(img_op: QWidget, x: int, y: int, width: int,
                         height: int) -> None:
        """Set the location of an image."""
        img_op.setScaledContents(True)
        img_op.setFixedSize(width, height)
        img_op.move(x, y)


def load_generator(config: ConfigType) -> CondSAGenerator:
    """Load generator."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = np.load(config.dataset_path)
    n_classes = dataset.max() + 1
    generator = CondSAGenerator(n_classes=n_classes,
                                model_config=config.model).to(device)
    model_dir = osp.join(config.output_dir, config.run_name, 'models')
    step = config.recover_model_step
    if step <= 0:
        model_path = osp.join(model_dir, 'generator_last.pth')
    else:
        model_path = osp.join(model_dir, f'generator_step_{step}.pth')
    generator.load_state_dict(torch.load(model_path))
    return generator


if __name__ == '__main__':
    global_config = GlobalConfig.build_from_argv(
        fallback='gan_facies/configs/exp/base.yaml')
    # NOTE: The config is not saved when testing only
    generator = load_generator(global_config)
    app = QApplication(sys.argv[:0])

    vis = ConditionalVisualizer(generator)
    vis.show()
    sys.exit(app.exec())
