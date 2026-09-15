"""
Usage: python zarr_dataset_viewer.py <zarr.zip file path>
Example: python zarr_dataset_viewer.py example_demo_session/dataset.zarr.zip

Optional arguments:
  --info-only    Only print data structure, don't launch GUI
"""
import sys
import os
import zarr
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QLineEdit, QMessageBox, QComboBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Register custom codecs
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
register_codecs()


def print_zarr_tree(group, indent=0, prefix=""):
    """
    Recursively print Zarr data structure tree view (simplified version, no array attributes)
    """
    indent_str = "  " * indent
    
    # Get all child groups and arrays
    items = list(group.keys())
    
    for i, key in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        child_prefix = "    " if is_last else "│   "
        
        try:
            item = group[key]
            
            if isinstance(item, zarr.hierarchy.Group):
                # This is a group (like folder)
                print(f"{prefix}{connector}📁 {key}/")
                print_zarr_tree(item, indent + 1, prefix + child_prefix)
            elif isinstance(item, zarr.core.Array):
                # This is an array, only print basic info
                dtype_str = str(item.dtype)
                shape_str = str(item.shape)
                
                # Choose icon based on data type
                if 'uint8' in dtype_str and len(item.shape) >= 3:
                    icon = "🖼️ "
                else:
                    icon = "📊"
                
                print(f"{prefix}{connector}{icon} {key}: shape={shape_str}, dtype={dtype_str}")
            else:
                print(f"{prefix}{connector}❓ {key} (unknown type: {type(item)})")
        except Exception as e:
            print(f"{prefix}{connector}❌ {key} (error: {e})")


def print_dataset_structure(zarr_path):
    """Print complete hierarchical structure of dataset"""
    print("\n" + "=" * 70)
    print(f"📦 Zarr Dataset Structure: {zarr_path}")
    print("=" * 70 + "\n")
    
    with zarr.ZipStore(zarr_path, mode='r') as store:
        root = zarr.group(store=store)
        
        # Print root directory info
        print(f"🏠 / (root)")
        print_zarr_tree(root, indent=0, prefix="")
        
        # Print statistical summary
        print("\n" + "=" * 70)
        print("📈 Dataset Statistical Summary")
        print("=" * 70)
        
        if 'meta' in root and 'episode_ends' in root['meta']:
            episode_ends = root['meta']['episode_ends'][:]
            num_episodes = len(episode_ends)
            total_frames = episode_ends[-1] if len(episode_ends) > 0 else 0
            
            # Calculate length of each episode
            episode_starts = np.concatenate([[0], episode_ends[:-1]])
            episode_lengths = episode_ends - episode_starts
            
            print(f"  Total Episodes: {num_episodes}")
            print(f"  Total Frames: {total_frames}")
            print(f"  Episode Length: min={episode_lengths.min()}, max={episode_lengths.max()}, avg={episode_lengths.mean():.1f}")
        
        if 'data' in root:
            data_group = root['data']
            print(f"\n  Data Keys:")
            for key in data_group.array_keys():
                arr = data_group[key]
                size_mb = arr.nbytes / (1024 * 1024)
                print(f"    - {key}: {arr.shape} ({arr.dtype}) [{size_mb:.2f} MB]")
    
    print("\n" + "=" * 70 + "\n")


class ZarrDatasetViewer(QWidget):
    def __init__(self, zarr_path):
        super().__init__()
        self.zarr_path = zarr_path
        self.zarr_file_name = os.path.basename(zarr_path)
        
        try:
            self.zip_store = zarr.ZipStore(zarr_path, mode='r')
            self.root = zarr.group(store=self.zip_store)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open Zarr file: {e}")
            sys.exit(1)

        # Parse data structure
        self.data_group = self.root['data']
        self.meta_group = self.root['meta']
        self.episode_ends = self.meta_group['episode_ends'][:]
        self.num_episodes = len(self.episode_ends)
        
        # Get all available keys
        self.data_keys = list(self.data_group.array_keys())
        print(f"Dataset contains keys: {self.data_keys}")
        
        # Identify image keys and low-dim data keys
        self.rgb_keys = [k for k in self.data_keys if 'rgb' in k.lower() or 'image' in k.lower()]
        self.lowdim_keys = [k for k in self.data_keys if k not in self.rgb_keys]
        
        # Calculate episode start indices
        self.episode_starts = np.concatenate([[0], self.episode_ends[:-1]])
        
        self.current_episode = 0
        self.frame_idx = 0
        self.init_ui()
        self.load_episode(0)

    def init_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Top: Episode selection area
        hbox_episode = QHBoxLayout()
        hbox_episode.addWidget(QLabel("Episode:"))
        self.episode_combo = QComboBox()
        self.episode_combo.addItems([str(i) for i in range(self.num_episodes)])
        self.episode_combo.currentIndexChanged.connect(self.on_episode_changed)
        hbox_episode.addWidget(self.episode_combo)
        
        self.episode_info_label = QLabel()
        hbox_episode.addWidget(self.episode_info_label)
        hbox_episode.addStretch()
        layout.addLayout(hbox_episode)

        # Image display area
        num_images = len(self.rgb_keys)
        if (num_images > 0):
            self.fig = Figure(figsize=(5 * max(num_images, 1), 4))
            self.canvas = FigureCanvas(self.fig)
            layout.addWidget(self.canvas)
        else:
            self.fig = None
            self.canvas = None

        # Info display area
        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(self.info_label)

        # Bottom controls (slider + input box)
        hbox = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self.on_slider)
        hbox.addWidget(QLabel("Frame:"))
        hbox.addWidget(self.slider)

        self.idx_input = QLineEdit()
        self.idx_input.setFixedWidth(80)
        self.idx_input.returnPressed.connect(self.on_idx_input)
        hbox.addWidget(self.idx_input)
        layout.addLayout(hbox)

        self.setWindowTitle(f"Zarr Dataset Viewer - {self.zarr_file_name}")
        self.resize(800, 700)

    def load_episode(self, episode_idx):
        self.current_episode = episode_idx
        self.start_idx = self.episode_starts[episode_idx]
        self.end_idx = self.episode_ends[episode_idx]
        self.episode_length = self.end_idx - self.start_idx
        
        self.episode_info_label.setText(
            f"(Length: {self.episode_length} frames, Global index: {self.start_idx} - {self.end_idx - 1})"
        )
        
        self.frame_idx = 0
        self.slider.setMaximum(self.episode_length - 1)
        self.slider.setValue(0)
        self.update_view()

    def on_episode_changed(self, idx):
        self.load_episode(idx)

    def format_array(self, arr, precision=4, width=9):
        if arr is None:
            return "N/A"
        arr = np.atleast_1d(arr)
        return "[" + "  ".join(f"{v:{width}.{precision}f}" for v in arr) + "]"

    def update_view(self):
        frame_idx = self.frame_idx
        global_idx = self.start_idx + frame_idx

        # Update images
        if self.fig is not None:
            self.fig.clear()
            for i, rgb_key in enumerate(self.rgb_keys):
                ax = self.fig.add_subplot(1, len(self.rgb_keys), i + 1)
                try:
                    img = self.data_group[rgb_key][global_idx]
                    if img.ndim == 2:
                        ax.imshow(img, cmap='gray')
                    else:
                        ax.imshow(img)
                    ax.set_title(rgb_key)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                ax.axis('off')
            self.canvas.draw()

        # Build info text
        info_lines = [
            f"File: {self.zarr_file_name}",
            f"Episode: {self.current_episode} / {self.num_episodes - 1}",
            f"Frame Index (Local): {frame_idx} / {self.episode_length - 1}",
            f"Frame Index (Global): {global_idx}",
            "",
            "=" * 60,
            "Low-dimensional Data:",
            "=" * 60,
        ]

        for key in self.lowdim_keys:
            try:
                value = self.data_group[key][global_idx]
                shape_str = f"shape={self.data_group[key].shape[1:]}"
                
                if isinstance(value, np.ndarray) and value.size <= 16:
                    value_str = self.format_array(value.flatten())
                elif isinstance(value, (int, float, np.integer, np.floating)):
                    value_str = f"{value:.4f}"
                else:
                    value_str = f"(shape: {value.shape})"
                    
                info_lines.append(f"  {key:35s} {shape_str:20s}: {value_str}")
            except Exception as e:
                info_lines.append(f"  {key:35s}: Error - {e}")

        # Display image data shape
        if self.rgb_keys:
            info_lines.append("")
            info_lines.append("=" * 60)
            info_lines.append("Image Data:")
            info_lines.append("=" * 60)
            for key in self.rgb_keys:
                shape = self.data_group[key].shape
                info_lines.append(f"  {key:35s}: shape={shape}")

        self.info_label.setText("\n".join(info_lines))
        self.idx_input.setText(str(frame_idx))

    def on_slider(self, value):
        self.frame_idx = value
        self.update_view()

    def on_idx_input(self):
        try:
            idx = int(self.idx_input.text())
            idx = max(0, min(idx, self.episode_length - 1))
            self.frame_idx = idx
            self.slider.setValue(idx)
            self.update_view()
        except ValueError:
            pass

    def closeEvent(self, event):
        self.zip_store.close()
        event.accept()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python zarr_dataset_viewer.py <zarr.zip file path> [--info-only]")
        print("Example: python zarr_dataset_viewer.py example_demo_session/dataset.zarr.zip")
        print("         python zarr_dataset_viewer.py example_demo_session/dataset.zarr.zip --info-only")
        sys.exit(1)

    zarr_path = sys.argv[1]
    info_only = "--info-only" in sys.argv
    
    # Print complete dataset structure
    print_dataset_structure(zarr_path)
    
    # Exit if only info is needed
    if info_only:
        sys.exit(0)
    
    # Launch GUI
    app = QApplication(sys.argv)
    viewer = ZarrDatasetViewer(zarr_path)
    viewer.show()
    sys.exit(app.exec_())
