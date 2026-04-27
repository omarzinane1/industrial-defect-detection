"""Interface desktop Tkinter pour la detection de defauts industriels."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.inference import INVALID_LABEL, METHOD_RULES, METHOD_SVM, PredictionResult, predict_image
from src.motion import MotionEstimate, MotionEstimator
from src.pipeline_visualization import (
    VIEW_CONTOURS,
    VIEW_CONTRAST,
    VIEW_FEATURES,
    VIEW_ORIGINAL,
    VIEW_SEGMENTATION,
    VIEW_SUSPECT_ZONES,
    VIEW_THRESHOLD,
    VIEW_TITLES,
    PipelineVisualization,
    build_pipeline_visualization,
    format_pipeline_statistics,
)


APP_TITLE = "Detection de defauts industriels"
PREVIEW_SIZE = (720, 520)
PIPELINE_PREVIEW_SIZE = (560, 460)

COLOR_BG = "#07111f"
COLOR_PANEL = "#111c2b"
COLOR_PANEL_ALT = "#1a2b3f"
COLOR_BORDER = "#36516e"
COLOR_TEXT = "#f3f4f6"
COLOR_MUTED = "#b6c2cf"
COLOR_ACCENT = "#0ea5e9"
COLOR_OK = "#39d98a"
COLOR_DEFECT = "#ff4d4f"
COLOR_WARNING = "#fbbf24"
COLOR_CARD_NEUTRAL = "#1b2a3c"
COLOR_CARD_OK = "#06351f"
COLOR_CARD_DEFECT = "#42151a"
COLOR_CARD_INVALID = "#3a2a08"
COLOR_INPUT = "#07111f"


class DefectDetectionApp(tk.Tk):
    """Application Tkinter principale."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x800")
        self.minsize(1180, 720)

        self.method_var = tk.StringVar(value=METHOD_RULES)
        self.source_var = tk.StringVar(value="Source : aucune")
        self.status_var = tk.StringVar(value="Pret.")
        self.prediction_var = tk.StringVar(value="Aucune prediction")
        self.confidence_var = tk.StringVar(value="Score non disponible")
        self.validation_var = tk.StringVar(value="Statut validation : non analyse")
        self.summary_var = tk.StringVar(value="Resume : aucune analyse lancee.")
        self.motion_state_var = tk.StringVar(value="Etat du mouvement : hors camera")
        self.motion_score_var = tk.StringVar(value="Score de mouvement : n/a")
        self.motion_points_var = tk.StringVar(value="Points suivis : n/a")
        self.motion_displacement_var = tk.StringVar(value="Deplacement moyen : n/a")
        self.motion_quality_var = tk.StringVar(value="Qualite capture : n/a")
        self.motion_message_var = tk.StringVar(value="Controle de stabilite disponible en mode camera.")

        self.current_image: np.ndarray | None = None
        self.current_image_path: Path | None = None
        self.current_source_kind = "none"
        self.camera: cv2.VideoCapture | None = None
        self.camera_running = False
        self.predict_live = tk.BooleanVar(value=False)
        self._last_live_prediction_ms = 0
        self._last_pipeline_update_ms = 0

        self._photo_ref: ImageTk.PhotoImage | None = None
        self.pipeline_windows: dict[str, tk.Toplevel] = {}
        self.pipeline_image_labels: dict[str, tk.Label] = {}
        self.pipeline_text_widgets: dict[str, tk.Text] = {}
        self.pipeline_photo_refs: dict[str, ImageTk.PhotoImage] = {}
        self.pipeline_visualization: PipelineVisualization | None = None
        self.motion_estimator = MotionEstimator()
        self.last_motion_estimate: MotionEstimate | None = None

        self._configure_style()
        self._build_layout()
        self._set_result_card("neutral")
        self._set_motion_card("off")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background=COLOR_BG)
        style.configure("Panel.TFrame", background=COLOR_PANEL, relief="solid", borderwidth=1)
        style.configure("PanelAlt.TFrame", background=COLOR_PANEL_ALT, relief="solid", borderwidth=1)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("Muted.TLabel", background=COLOR_PANEL, foreground=COLOR_MUTED, font=("Segoe UI", 9))
        style.configure("Title.TLabel", background=COLOR_BG, foreground=COLOR_TEXT, font=("Segoe UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=COLOR_BG, foreground=COLOR_MUTED, font=("Segoe UI", 10))
        style.configure("Section.TLabel", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 12, "bold"))
        style.configure("ResultNeutral.TLabel", background=COLOR_PANEL_ALT, foreground=COLOR_TEXT, font=("Segoe UI", 22, "bold"))
        style.configure("ResultOk.TLabel", background=COLOR_PANEL_ALT, foreground=COLOR_OK, font=("Segoe UI", 22, "bold"))
        style.configure("ResultDefect.TLabel", background=COLOR_PANEL_ALT, foreground=COLOR_DEFECT, font=("Segoe UI", 22, "bold"))
        style.configure("ResultInvalid.TLabel", background=COLOR_PANEL_ALT, foreground=COLOR_WARNING, font=("Segoe UI", 22, "bold"))
        style.configure("Status.TLabel", background="#0b2536", foreground=COLOR_TEXT, padding=(10, 6))
        style.configure(
            "TButton",
            background="#26384d",
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10),
            padding=(10, 7),
        )
        style.configure(
            "Primary.TButton",
            background=COLOR_ACCENT,
            foreground="#06101c",
            font=("Segoe UI", 10, "bold"),
            padding=(10, 9),
        )
        style.configure(
            "Danger.TButton",
            background="#7f1d1d",
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10, "bold"),
            padding=(10, 7),
        )
        style.configure(
            "Pipeline.TButton",
            background="#1e3a56",
            foreground=COLOR_TEXT,
            font=("Segoe UI", 9),
            padding=(8, 7),
        )
        style.configure("TRadiobutton", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 10))
        style.configure("TCheckbutton", background=COLOR_PANEL, foreground=COLOR_TEXT, font=("Segoe UI", 10))

        style.map("TButton", background=[("active", "#314761")])
        style.map("Primary.TButton", background=[("active", "#38bdf8")])
        style.map("Danger.TButton", background=[("active", "#991b1b")])
        style.map("TRadiobutton", background=[("active", COLOR_PANEL)])
        style.map("TCheckbutton", background=[("active", COLOR_PANEL)])

    def _build_layout(self) -> None:
        self.configure(background=COLOR_BG)
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(20, 14))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Poste de controle : acquisition, prediction et visualisation du pipeline.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self._build_left_panel()
        self._build_right_panel()

        status = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel")
        status.grid(row=2, column=0, columnspan=2, sticky="ew")

    def _build_left_panel(self) -> None:
        left = ttk.Frame(self, style="Panel.TFrame", padding=16)
        left.grid(row=1, column=0, sticky="nsew", padx=(20, 10), pady=(0, 20))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Visualisation principale", style="Section.TLabel").grid(row=0, column=0, sticky="w")

        preview_frame = tk.Frame(left, background="#060b12", highlightbackground=COLOR_BORDER, highlightthickness=1)
        preview_frame.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_label = tk.Label(
            preview_frame,
            text="Aucune image chargee",
            anchor="center",
            background="#060b12",
            foreground=COLOR_MUTED,
            font=("Segoe UI", 13),
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

    def _build_right_panel(self) -> None:
        right_shell = ttk.Frame(self, style="Panel.TFrame")
        right_shell.grid(row=1, column=1, sticky="nsew", padx=(10, 20), pady=(0, 20))
        right_shell.columnconfigure(0, weight=1)
        right_shell.rowconfigure(0, weight=1)

        self.right_canvas = tk.Canvas(
            right_shell,
            background=COLOR_PANEL,
            highlightthickness=0,
            borderwidth=0,
        )
        right_scrollbar = ttk.Scrollbar(right_shell, orient="vertical", command=self.right_canvas.yview)
        self.right_canvas.configure(yscrollcommand=right_scrollbar.set)
        self.right_canvas.grid(row=0, column=0, sticky="nsew")
        right_scrollbar.grid(row=0, column=1, sticky="ns")

        right = ttk.Frame(self.right_canvas, style="Panel.TFrame", padding=16)
        self.right_canvas_window = self.right_canvas.create_window((0, 0), window=right, anchor="nw")
        self.right_canvas.bind("<Configure>", self._resize_right_canvas_content)
        right.bind("<Configure>", self._update_right_scroll_region)
        right_shell.bind("<Enter>", self._enable_right_mousewheel)
        right_shell.bind("<Leave>", self._disable_right_mousewheel)
        right.bind("<Enter>", self._enable_right_mousewheel)
        right.bind("<Leave>", self._disable_right_mousewheel)

        right.columnconfigure(0, weight=1)

        ttk.Label(right, text="Methode de prediction", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        method_box = ttk.Frame(right, style="Panel.TFrame")
        method_box.grid(row=1, column=0, sticky="ew", pady=(8, 16))
        ttk.Radiobutton(method_box, text=METHOD_RULES, value=METHOD_RULES, variable=self.method_var).grid(
            row=0, column=0, sticky="w", padx=8, pady=(8, 2)
        )
        ttk.Radiobutton(method_box, text=METHOD_SVM, value=METHOD_SVM, variable=self.method_var).grid(
            row=1, column=0, sticky="w", padx=8, pady=2
        )
        ttk.Checkbutton(
            method_box,
            text="Analyse semi temps reel avec la camera",
            variable=self.predict_live,
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(8, 8))

        ttk.Label(right, text="Actions", style="Section.TLabel").grid(row=2, column=0, sticky="w")
        action_box = ttk.Frame(right, style="Panel.TFrame")
        action_box.grid(row=3, column=0, sticky="ew", pady=(8, 18))
        for column in range(2):
            action_box.columnconfigure(column, weight=1)

        ttk.Button(action_box, text="Importer image", command=self.import_image).grid(
            row=0, column=0, sticky="ew", padx=(8, 5), pady=(8, 6)
        )
        ttk.Button(action_box, text="Ouvrir camera", command=self.start_camera).grid(
            row=0, column=1, sticky="ew", padx=(5, 8), pady=(8, 6)
        )
        ttk.Button(action_box, text="Capturer", command=self.capture_camera_frame).grid(
            row=1, column=0, sticky="ew", padx=(8, 5), pady=(0, 6)
        )
        ttk.Button(action_box, text="Arreter camera", style="Danger.TButton", command=self.stop_camera).grid(
            row=1, column=1, sticky="ew", padx=(5, 8), pady=(0, 6)
        )
        ttk.Button(action_box, text="Lancer la prediction", style="Primary.TButton", command=self.run_prediction).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(2, 6)
        )
        ttk.Button(action_box, text="Reinitialiser", command=self.reset).grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8)
        )

        ttk.Label(right, text="Controle de stabilite camera", style="Section.TLabel").grid(row=4, column=0, sticky="w")
        motion_box = tk.Frame(
            right,
            background=COLOR_PANEL_ALT,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
            padx=14,
            pady=12,
        )
        motion_box.grid(row=5, column=0, sticky="ew", pady=(8, 18))
        motion_box.columnconfigure(0, weight=1)

        self.motion_box = motion_box
        self.motion_badge = tk.Label(
            motion_box,
            text="HORS CAMERA",
            background="#27364a",
            foreground=COLOR_TEXT,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3,
        )
        self.motion_badge.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.motion_state_label = tk.Label(
            motion_box,
            textvariable=self.motion_state_var,
            background=COLOR_PANEL_ALT,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        )
        self.motion_state_label.grid(row=1, column=0, sticky="ew")

        self.motion_message_label = tk.Label(
            motion_box,
            textvariable=self.motion_message_var,
            background=COLOR_PANEL_ALT,
            foreground=COLOR_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
            justify="left",
            wraplength=420,
        )
        self.motion_message_label.grid(row=2, column=0, sticky="ew", pady=(6, 10))

        self.motion_score_label = tk.Label(
            motion_box,
            textvariable=self.motion_score_var,
            background=COLOR_PANEL_ALT,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.motion_score_label.grid(row=3, column=0, sticky="ew")

        self.motion_points_label = tk.Label(
            motion_box,
            textvariable=self.motion_points_var,
            background=COLOR_PANEL_ALT,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.motion_points_label.grid(row=4, column=0, sticky="ew", pady=(2, 0))

        self.motion_displacement_label = tk.Label(
            motion_box,
            textvariable=self.motion_displacement_var,
            background=COLOR_PANEL_ALT,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.motion_displacement_label.grid(row=5, column=0, sticky="ew", pady=(2, 0))

        self.motion_quality_label = tk.Label(
            motion_box,
            textvariable=self.motion_quality_var,
            background=COLOR_PANEL_ALT,
            foreground=COLOR_MUTED,
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        )
        self.motion_quality_label.grid(row=6, column=0, sticky="ew", pady=(8, 0))

        ttk.Label(right, text="Decision de controle", style="Section.TLabel").grid(row=6, column=0, sticky="w")
        result_box = tk.Frame(
            right,
            background=COLOR_CARD_NEUTRAL,
            highlightbackground=COLOR_BORDER,
            highlightthickness=2,
            padx=18,
            pady=16,
        )
        result_box.grid(row=7, column=0, sticky="ew", pady=(8, 18))
        result_box.columnconfigure(0, weight=1)

        self.result_box = result_box
        self.result_badge = tk.Label(
            result_box,
            text="EN ATTENTE",
            background="#27364a",
            foreground=COLOR_TEXT,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=4,
        )
        self.result_badge.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.prediction_label = tk.Label(
            result_box,
            textvariable=self.prediction_var,
            background=COLOR_CARD_NEUTRAL,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 28, "bold"),
            anchor="w",
        )
        self.prediction_label.grid(row=1, column=0, sticky="ew")

        self.confidence_label = tk.Label(
            result_box,
            textvariable=self.confidence_var,
            background=COLOR_CARD_NEUTRAL,
            foreground=COLOR_MUTED,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        self.confidence_label.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.validation_label = tk.Label(
            result_box,
            textvariable=self.validation_var,
            background=COLOR_CARD_NEUTRAL,
            foreground=COLOR_MUTED,
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        self.validation_label.grid(row=3, column=0, sticky="ew", pady=(10, 0))

        self.source_label = tk.Label(
            result_box,
            textvariable=self.source_var,
            background=COLOR_CARD_NEUTRAL,
            foreground=COLOR_MUTED,
            font=("Segoe UI", 10),
            anchor="w",
        )
        self.source_label.grid(row=4, column=0, sticky="ew", pady=(2, 0))

        self.summary_label = tk.Label(
            result_box,
            textvariable=self.summary_var,
            background=COLOR_CARD_NEUTRAL,
            foreground=COLOR_TEXT,
            font=("Segoe UI", 11),
            anchor="w",
            justify="left",
            wraplength=420,
        )
        self.summary_label.grid(row=5, column=0, sticky="ew", pady=(12, 0))

        ttk.Label(right, text="Indicateurs / statistiques", style="Section.TLabel").grid(row=8, column=0, sticky="nw")
        stats_frame = ttk.Frame(right, style="Panel.TFrame")
        stats_frame.grid(row=9, column=0, sticky="ew", pady=(8, 18))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)

        self.stats_text = tk.Text(
            stats_frame,
            height=10,
            wrap="word",
            relief="flat",
            borderwidth=0,
            background=COLOR_INPUT,
            foreground=COLOR_TEXT,
            insertbackground=COLOR_TEXT,
            font=("Consolas", 10),
        )
        self.stats_text.grid(row=0, column=0, sticky="ew", padx=(8, 0), pady=8)
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=8, padx=(0, 8))
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        self._write_stats("Aucune analyse lancee.")

        ttk.Label(right, text="Vues du pipeline", style="Section.TLabel").grid(row=10, column=0, sticky="w")
        pipeline_box = ttk.Frame(right, style="Panel.TFrame")
        pipeline_box.grid(row=11, column=0, sticky="ew", pady=(8, 12))
        for column in range(2):
            pipeline_box.columnconfigure(column, weight=1)

        pipeline_buttons = [
            (VIEW_ORIGINAL, 0, 0),
            (VIEW_CONTRAST, 0, 1),
            (VIEW_THRESHOLD, 1, 0),
            (VIEW_SEGMENTATION, 1, 1),
            (VIEW_CONTOURS, 2, 0),
            (VIEW_SUSPECT_ZONES, 2, 1),
            (VIEW_FEATURES, 3, 0),
        ]
        for view_key, row, column in pipeline_buttons:
            colspan = 2 if view_key == VIEW_FEATURES else 1
            ttk.Button(
                pipeline_box,
                text=VIEW_TITLES[view_key],
                style="Pipeline.TButton",
                command=lambda key=view_key: self.open_pipeline_view(key),
            ).grid(row=row, column=column, columnspan=colspan, sticky="ew", padx=8, pady=5)

    def _resize_right_canvas_content(self, event: tk.Event) -> None:
        self.right_canvas.itemconfigure(self.right_canvas_window, width=event.width)

    def _update_right_scroll_region(self, _event: tk.Event) -> None:
        self.right_canvas.configure(scrollregion=self.right_canvas.bbox("all"))

    def _enable_right_mousewheel(self, _event: tk.Event) -> None:
        self.bind_all("<MouseWheel>", self._on_right_mousewheel)

    def _disable_right_mousewheel(self, _event: tk.Event) -> None:
        self.unbind_all("<MouseWheel>")

    def _on_right_mousewheel(self, event: tk.Event) -> None:
        delta = int(-1 * (event.delta / 120))
        self.right_canvas.yview_scroll(delta, "units")

    def _reset_motion_panel(self, message: str = "Controle de stabilite disponible en mode camera.") -> None:
        self.last_motion_estimate = None
        self.motion_state_var.set("Etat du mouvement : hors camera")
        self.motion_score_var.set("Score de mouvement : n/a")
        self.motion_points_var.set("Points suivis : n/a")
        self.motion_displacement_var.set("Deplacement moyen : n/a")
        self.motion_quality_var.set("Qualite capture : n/a")
        self.motion_message_var.set(message)
        self._set_motion_card("off")

    def _display_motion_estimate(self, estimate: MotionEstimate | None) -> None:
        if estimate is None:
            self._reset_motion_panel()
            return

        self.last_motion_estimate = estimate
        self.motion_state_var.set(f"Etat du mouvement : {estimate.status}")
        self.motion_score_var.set(f"Score de mouvement : {estimate.motion_score:.3f}")
        self.motion_points_var.set(
            f"Points suivis : {estimate.tracked_points} / {estimate.detected_points}"
        )
        self.motion_displacement_var.set(f"Deplacement moyen : {estimate.mean_displacement:.3f} px")
        self.motion_quality_var.set(f"Qualite capture : {estimate.capture_quality}")
        self.motion_message_var.set(estimate.message)

        status_to_style = {
            "Initialisation": "init",
            "Stable": "stable",
            "En mouvement": "moving",
            "Instable": "unstable",
            "Indisponible": "warning",
        }
        self._set_motion_card(status_to_style.get(estimate.status, "warning"))

    def _set_motion_card(self, state: str) -> None:
        card_styles = {
            "off": (COLOR_PANEL_ALT, COLOR_TEXT, "#27364a", "HORS CAMERA"),
            "init": ("#11273a", COLOR_ACCENT, "#0b4f6c", "INITIALISATION"),
            "stable": ("#0b3021", COLOR_OK, "#047857", "STABLE"),
            "moving": ("#2c240d", COLOR_WARNING, "#b45309", "EN MOUVEMENT"),
            "unstable": ("#401519", COLOR_DEFECT, "#b91c1c", "INSTABLE"),
            "warning": ("#2a2530", COLOR_WARNING, "#92400e", "A VERIFIER"),
        }
        background, accent, badge_bg, badge_text = card_styles.get(state, card_styles["warning"])

        self.motion_box.configure(background=background, highlightbackground=accent)
        self.motion_badge.configure(text=badge_text, background=badge_bg, foreground=COLOR_TEXT)
        self.motion_state_label.configure(background=background, foreground=accent)
        self.motion_message_label.configure(background=background, foreground=COLOR_MUTED)
        self.motion_score_label.configure(background=background, foreground=COLOR_TEXT)
        self.motion_points_label.configure(background=background, foreground=COLOR_TEXT)
        self.motion_displacement_label.configure(background=background, foreground=COLOR_TEXT)
        self.motion_quality_label.configure(background=background, foreground=accent)

    def import_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Selectionner une image",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if not path:
            return

        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            messagebox.showerror("Image illisible", "Impossible de charger cette image.")
            return

        self.stop_camera()
        self.current_image = image
        self.current_image_path = Path(path)
        self.current_source_kind = "image"
        self.motion_estimator.reset()
        self.source_var.set(f"Source : image importee - {self.current_image_path.name}")
        self.status_var.set("Image importee. Lancez la prediction ou ouvrez une vue du pipeline.")
        self._reset_motion_panel("Controle de stabilite indisponible sur une image importee.")
        self._show_image(image)
        self._refresh_pipeline_windows(include_stats=VIEW_FEATURES in self.pipeline_windows)

    def start_camera(self) -> None:
        if self.camera_running:
            return

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("Camera indisponible", "Impossible d'ouvrir la camera du PC.")
            camera.release()
            return

        self.camera = camera
        self.camera_running = True
        self.current_image_path = None
        self.current_source_kind = "camera"
        self.motion_estimator.reset()
        self.source_var.set("Source : camera PC")
        self.status_var.set("Camera ouverte.")
        self._reset_motion_panel("Initialisation du suivi Lucas-Kanade en attente.")
        self._update_camera_frame()

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.status_var.set("Camera arretee.")

    def capture_camera_frame(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("Aucune image", "Aucune frame camera disponible a capturer.")
            return
        self.stop_camera()
        self.current_source_kind = "capture"
        self.source_var.set("Source : capture camera")
        self.status_var.set("Frame capturee. Lancez la prediction.")
        self._show_image(self.current_image)
        self._refresh_pipeline_windows(include_stats=True)

    def run_prediction(self, show_warning: bool = True) -> None:
        if self.current_image is None:
            messagebox.showwarning("Aucune image", "Importez une image ou ouvrez la camera avant de lancer l'analyse.")
            return

        if self.current_source_kind in {"camera", "capture"} and self.last_motion_estimate is not None:
            if self.last_motion_estimate.should_block_prediction:
                self.status_var.set(self.last_motion_estimate.message)
                if show_warning:
                    messagebox.showwarning("Capture instable", self.last_motion_estimate.message)
                return

        method = self.method_var.get()
        try:
            result = predict_image(self.current_image, method)
        except Exception as exc:  # message utilisateur propre pour dependances/modeles manquants
            messagebox.showerror("Erreur de prediction", str(exc))
            self.status_var.set("Erreur pendant la prediction.")
            return

        self._display_result(result)
        self._refresh_pipeline_windows(include_stats=True)
        self.status_var.set("Prediction terminee.")

    def reset(self) -> None:
        self.stop_camera()
        self.current_image = None
        self.current_image_path = None
        self.current_source_kind = "none"
        self.pipeline_visualization = None
        self.motion_estimator.reset()
        self.preview_label.configure(image="", text="Aucune image chargee")
        self._photo_ref = None
        self.prediction_var.set("Aucune prediction")
        self.confidence_var.set("Score non disponible")
        self.validation_var.set("Statut validation : non analyse")
        self.summary_var.set("Resume : aucune analyse lancee.")
        self.source_var.set("Source : aucune")
        self._set_result_card("neutral")
        self._reset_motion_panel()
        self._write_stats("Aucune analyse lancee.")
        self._close_pipeline_windows()
        self.status_var.set("Interface reinitialisee.")

    def open_pipeline_view(self, view_key: str) -> None:
        if self.current_image is None:
            messagebox.showwarning("Aucune image", "Chargez une image ou ouvrez la camera avant d'afficher le pipeline.")
            return

        self._update_pipeline_cache(include_stats=view_key == VIEW_FEATURES)

        existing = self.pipeline_windows.get(view_key)
        if existing is not None and existing.winfo_exists():
            self._update_pipeline_window(view_key)
            existing.lift()
            existing.focus_force()
            return

        window = tk.Toplevel(self)
        window.title(VIEW_TITLES[view_key])
        window.configure(background=COLOR_BG)
        window.minsize(520, 420)
        window.protocol("WM_DELETE_WINDOW", lambda key=view_key: self._close_pipeline_window(key))
        self.pipeline_windows[view_key] = window

        ttk.Label(window, text=VIEW_TITLES[view_key], style="Title.TLabel").pack(anchor="w", padx=16, pady=(14, 8))

        if view_key == VIEW_FEATURES:
            text = tk.Text(
                window,
                wrap="word",
                relief="flat",
                borderwidth=0,
                background="#0b1220",
                foreground=COLOR_TEXT,
                insertbackground=COLOR_TEXT,
                font=("Consolas", 10),
            )
            text.pack(fill="both", expand=True, padx=16, pady=(0, 16))
            self.pipeline_text_widgets[view_key] = text
        else:
            label = tk.Label(window, background="#060b12", foreground=COLOR_MUTED)
            label.pack(fill="both", expand=True, padx=16, pady=(0, 16))
            self.pipeline_image_labels[view_key] = label

        self._update_pipeline_window(view_key)

    def _update_camera_frame(self) -> None:
        if not self.camera_running or self.camera is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Camera", "Impossible de lire le flux camera.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_image = gray
        motion_estimate = self.motion_estimator.update(gray)
        self._display_motion_estimate(motion_estimate)
        self._show_image(gray)

        now = int(self.tk.call("clock", "milliseconds"))
        if self.pipeline_windows and now - self._last_pipeline_update_ms > 180:
            self._last_pipeline_update_ms = now
            self._refresh_pipeline_windows(include_stats=VIEW_FEATURES in self.pipeline_windows)

        if self.predict_live.get() and now - self._last_live_prediction_ms > 1200:
            self._last_live_prediction_ms = now
            if motion_estimate.status == "Stable":
                self.run_prediction(show_warning=False)
            else:
                self.status_var.set(motion_estimate.message)

        self.after(30, self._update_camera_frame)

    def _update_pipeline_cache(self, include_stats: bool = False) -> bool:
        if self.current_image is None:
            return False
        try:
            self.pipeline_visualization = build_pipeline_visualization(self.current_image, include_stats=include_stats)
        except Exception as exc:
            self.status_var.set(f"Erreur pipeline : {exc}")
            return False
        return True

    def _refresh_pipeline_windows(self, include_stats: bool = False) -> None:
        if not self.pipeline_windows or self.current_image is None:
            return
        if not self._update_pipeline_cache(include_stats=include_stats):
            return
        for view_key in list(self.pipeline_windows):
            self._update_pipeline_window(view_key)

    def _update_pipeline_window(self, view_key: str) -> None:
        if self.pipeline_visualization is None:
            return

        window = self.pipeline_windows.get(view_key)
        if window is None or not window.winfo_exists():
            self._close_pipeline_window(view_key)
            return

        if view_key == VIEW_FEATURES:
            if not self.pipeline_visualization.stats and self.current_image is not None:
                self._update_pipeline_cache(include_stats=True)
            text = self.pipeline_text_widgets.get(view_key)
            if text is None:
                return
            content = format_pipeline_statistics(self.pipeline_visualization.stats)
            text.configure(state="normal")
            text.delete("1.0", "end")
            text.insert("1.0", content)
            text.configure(state="disabled")
            return

        image = self.pipeline_visualization.images.get(view_key)
        label = self.pipeline_image_labels.get(view_key)
        if image is None or label is None:
            return

        photo = self._image_to_photo(image, PIPELINE_PREVIEW_SIZE, background=(6, 11, 18))
        self.pipeline_photo_refs[view_key] = photo
        label.configure(image=photo, text="")

    def _close_pipeline_window(self, view_key: str) -> None:
        window = self.pipeline_windows.pop(view_key, None)
        self.pipeline_image_labels.pop(view_key, None)
        self.pipeline_text_widgets.pop(view_key, None)
        self.pipeline_photo_refs.pop(view_key, None)
        if window is not None and window.winfo_exists():
            window.destroy()

    def _close_pipeline_windows(self) -> None:
        for view_key in list(self.pipeline_windows):
            self._close_pipeline_window(view_key)

    def _show_image(self, image: np.ndarray) -> None:
        self._photo_ref = self._image_to_photo(image, PREVIEW_SIZE, background=(6, 11, 18))
        self.preview_label.configure(image=self._photo_ref, text="")

    def _image_to_photo(
        self,
        image: np.ndarray,
        size: tuple[int, int],
        background: tuple[int, int, int],
    ) -> ImageTk.PhotoImage:
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        pil_image.thumbnail(size, resample_filter)

        canvas = Image.new("RGB", size, color=background)
        offset = ((size[0] - pil_image.width) // 2, (size[1] - pil_image.height) // 2)
        canvas.paste(pil_image, offset)
        return ImageTk.PhotoImage(canvas)

    def _display_result(self, result: PredictionResult) -> None:
        self.confidence_var.set(result.confidence_label or "Score non disponible")
        self.summary_var.set(f"Resume : {result.summary}")

        if result.predicted_label == INVALID_LABEL:
            self.prediction_var.set("Image non valide")
            self.validation_var.set("Statut validation : Image non valide")
            self._set_result_card("invalid")
        else:
            self.validation_var.set("Statut validation : Piece detectee")
            if result.predicted_name == "Defective":
                self.prediction_var.set("Piece defectueuse")
                self._set_result_card("defective")
            else:
                self.prediction_var.set("Piece OK")
                self._set_result_card("ok")

        self._write_stats(self._format_stats(result))

    def _set_result_card(self, state: str) -> None:
        card_styles = {
            "neutral": (COLOR_CARD_NEUTRAL, COLOR_TEXT, "#27364a", "SYSTEME PRET"),
            "ok": (COLOR_CARD_OK, COLOR_OK, "#047857", "CONTROLE VALIDE"),
            "defective": (COLOR_CARD_DEFECT, COLOR_DEFECT, "#b91c1c", "ALERTE QUALITE"),
            "invalid": (COLOR_CARD_INVALID, COLOR_WARNING, "#b45309", "! IMAGE NON VALIDE"),
        }
        background, accent, badge_bg, badge_text = card_styles.get(state, card_styles["neutral"])

        self.result_box.configure(background=background, highlightbackground=accent)
        self.result_badge.configure(text=badge_text, background=badge_bg, foreground=COLOR_TEXT)
        self.prediction_label.configure(background=background, foreground=accent)
        self.confidence_label.configure(background=background, foreground=COLOR_MUTED)
        self.validation_label.configure(background=background, foreground=accent)
        self.source_label.configure(background=background, foreground=COLOR_MUTED)
        self.summary_label.configure(background=background, foreground=COLOR_TEXT)

    def _format_stats(self, result: PredictionResult) -> str:
        lines = [
            f"Methode : {result.method}",
            f"Classe predite : {result.predicted_name}",
            f"Resume : {result.summary}",
            "",
            "Statistiques :",
        ]

        for key, value in result.stats.items():
            if value is None:
                continue
            if isinstance(value, float):
                lines.append(f"- {key}: {value:.6f}")
            else:
                lines.append(f"- {key}: {value}")

        if self.current_source_kind in {"camera", "capture"} and self.last_motion_estimate is not None:
            motion = self.last_motion_estimate
            lines.extend(
                [
                    "",
                    "Controle de stabilite camera :",
                    f"- etat mouvement: {motion.status}",
                    f"- score mouvement: {motion.motion_score:.6f}",
                    f"- points suivis: {motion.tracked_points}",
                    f"- deplacement moyen: {motion.mean_displacement:.6f}",
                    f"- qualite capture: {motion.capture_quality}",
                    f"- message mouvement: {motion.message}",
                ]
            )

        return "\n".join(lines)

    def _write_stats(self, text: str) -> None:
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.configure(state="disabled")

    def on_close(self) -> None:
        self.stop_camera()
        self._close_pipeline_windows()
        self.destroy()


def main() -> None:
    app = DefectDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
