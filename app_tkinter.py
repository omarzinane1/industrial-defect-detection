"""Interface desktop Tkinter pour la detection de defauts industriels."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.inference import METHOD_RULES, METHOD_SVM, PredictionResult, predict_image


APP_TITLE = "Détection de défauts industriels"
PREVIEW_SIZE = (640, 480)


class DefectDetectionApp(tk.Tk):
    """Application Tkinter principale."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x720")
        self.minsize(1050, 650)

        self.method_var = tk.StringVar(value=METHOD_RULES)
        self.source_var = tk.StringVar(value="Aucune source")
        self.status_var = tk.StringVar(value="Prêt.")
        self.prediction_var = tk.StringVar(value="Aucune prédiction")
        self.confidence_var = tk.StringVar(value="Score non disponible")

        self.current_image: np.ndarray | None = None
        self.current_image_path: Path | None = None
        self.camera: cv2.VideoCapture | None = None
        self.camera_running = False
        self.predict_live = tk.BooleanVar(value=False)
        self._last_live_prediction_ms = 0
        self._photo_ref: ImageTk.PhotoImage | None = None

        self._configure_style()
        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("TFrame", background="#f5f6f8")
        style.configure("Panel.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f5f6f8", foreground="#1f2933", font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background="#ffffff", foreground="#1f2933", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#f5f6f8", font=("Segoe UI", 18, "bold"))
        style.configure("Section.TLabel", background="#ffffff", font=("Segoe UI", 12, "bold"))
        style.configure("Result.TLabel", background="#ffffff", font=("Segoe UI", 20, "bold"))
        style.configure("Status.TLabel", background="#e9edf2", foreground="#344054", padding=(8, 5))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 6))
        style.configure("TRadiobutton", background="#ffffff", font=("Segoe UI", 10))
        style.configure("TCheckbutton", background="#ffffff", font=("Segoe UI", 10))

    def _build_layout(self) -> None:
        self.configure(background="#f5f6f8")
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(18, 14))
        header.grid(row=0, column=0, columnspan=2, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Choisir une méthode, charger une image ou utiliser la caméra, puis lancer l’analyse.",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        self._build_left_panel()
        self._build_right_panel()

        status = ttk.Label(self, textvariable=self.status_var, style="Status.TLabel")
        status.grid(row=2, column=0, columnspan=2, sticky="ew")

    def _build_left_panel(self) -> None:
        left = ttk.Frame(self, style="Panel.TFrame", padding=14)
        left.grid(row=1, column=0, sticky="nsew", padx=(18, 9), pady=(0, 18))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)

        ttk.Label(left, text="Aperçu image / caméra", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.preview_label = ttk.Label(left, text="Aucune image chargée", anchor="center", style="Panel.TLabel")
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=12)
        self.preview_label.configure(borderwidth=1, relief="solid")

        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        for index in range(5):
            controls.columnconfigure(index, weight=1)

        ttk.Button(controls, text="Importer une image", command=self.import_image).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        ttk.Button(controls, text="Ouvrir la caméra", command=self.start_camera).grid(
            row=0, column=1, sticky="ew", padx=6
        )
        ttk.Button(controls, text="Capturer", command=self.capture_camera_frame).grid(
            row=0, column=2, sticky="ew", padx=6
        )
        ttk.Button(controls, text="Arrêter la caméra", command=self.stop_camera).grid(
            row=0, column=3, sticky="ew", padx=6
        )
        ttk.Button(controls, text="Réinitialiser", command=self.reset).grid(row=0, column=4, sticky="ew", padx=(6, 0))

    def _build_right_panel(self) -> None:
        right = ttk.Frame(self, style="Panel.TFrame", padding=16)
        right.grid(row=1, column=1, sticky="nsew", padx=(9, 18), pady=(0, 18))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(5, weight=1)

        ttk.Label(right, text="Méthode de prédiction", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        method_box = ttk.Frame(right, style="Panel.TFrame")
        method_box.grid(row=1, column=0, sticky="ew", pady=(8, 16))
        ttk.Radiobutton(method_box, text=METHOD_RULES, value=METHOD_RULES, variable=self.method_var).grid(
            row=0, column=0, sticky="w", pady=2
        )
        ttk.Radiobutton(method_box, text=METHOD_SVM, value=METHOD_SVM, variable=self.method_var).grid(
            row=1, column=0, sticky="w", pady=2
        )
        ttk.Checkbutton(
            method_box,
            text="Analyse semi temps réel avec la caméra",
            variable=self.predict_live,
        ).grid(row=2, column=0, sticky="w", pady=(8, 0))

        action_box = ttk.Frame(right, style="Panel.TFrame")
        action_box.grid(row=2, column=0, sticky="ew", pady=(0, 16))
        action_box.columnconfigure(0, weight=1)
        ttk.Button(action_box, text="Lancer la prédiction", command=self.run_prediction).grid(
            row=0, column=0, sticky="ew"
        )

        ttk.Label(right, text="Résultat", style="Section.TLabel").grid(row=3, column=0, sticky="w")
        result_box = ttk.Frame(right, style="Panel.TFrame", padding=12)
        result_box.grid(row=4, column=0, sticky="ew", pady=(8, 16))
        result_box.columnconfigure(0, weight=1)
        ttk.Label(result_box, textvariable=self.prediction_var, style="Result.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(result_box, textvariable=self.confidence_var, style="Panel.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Label(result_box, textvariable=self.source_var, style="Panel.TLabel").grid(row=2, column=0, sticky="w")

        ttk.Label(right, text="Statistiques utiles", style="Section.TLabel").grid(row=5, column=0, sticky="nw")
        stats_frame = ttk.Frame(right, style="Panel.TFrame")
        stats_frame.grid(row=6, column=0, sticky="nsew", pady=(8, 0))
        stats_frame.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)

        self.stats_text = tk.Text(
            stats_frame,
            height=18,
            wrap="word",
            relief="flat",
            borderwidth=0,
            background="#ffffff",
            foreground="#1f2933",
            font=("Consolas", 10),
        )
        self.stats_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=self.stats_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.stats_text.configure(yscrollcommand=scrollbar.set)
        self._write_stats("Aucune analyse lancée.")

    def import_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Sélectionner une image",
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
        self.source_var.set(f"Source : image importée - {self.current_image_path.name}")
        self.status_var.set("Image importée. Lancez la prédiction.")
        self._show_image(image)

    def start_camera(self) -> None:
        if self.camera_running:
            return

        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            messagebox.showerror("Caméra indisponible", "Impossible d'ouvrir la caméra du PC.")
            camera.release()
            return

        self.camera = camera
        self.camera_running = True
        self.current_image_path = None
        self.source_var.set("Source : caméra PC")
        self.status_var.set("Caméra ouverte.")
        self._update_camera_frame()

    def stop_camera(self) -> None:
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.status_var.set("Caméra arrêtée.")

    def capture_camera_frame(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("Aucune image", "Aucune frame caméra disponible à capturer.")
            return
        self.stop_camera()
        self.source_var.set("Source : capture caméra")
        self.status_var.set("Frame capturée. Lancez la prédiction.")
        self._show_image(self.current_image)

    def run_prediction(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("Aucune image", "Importez une image ou ouvrez la caméra avant de lancer l'analyse.")
            return

        method = self.method_var.get()
        try:
            result = predict_image(self.current_image, method)
        except Exception as exc:  # message utilisateur propre pour dependances/modeles manquants
            messagebox.showerror("Erreur de prédiction", str(exc))
            self.status_var.set("Erreur pendant la prédiction.")
            return

        self._display_result(result)
        self.status_var.set("Prédiction terminée.")

    def reset(self) -> None:
        self.stop_camera()
        self.current_image = None
        self.current_image_path = None
        self.preview_label.configure(image="", text="Aucune image chargée")
        self._photo_ref = None
        self.prediction_var.set("Aucune prédiction")
        self.confidence_var.set("Score non disponible")
        self.source_var.set("Aucune source")
        self._write_stats("Aucune analyse lancée.")
        self.status_var.set("Interface réinitialisée.")

    def _update_camera_frame(self) -> None:
        if not self.camera_running or self.camera is None:
            return

        ok, frame = self.camera.read()
        if not ok:
            self.stop_camera()
            messagebox.showerror("Caméra", "Impossible de lire le flux caméra.")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_image = gray
        self._show_image(gray)

        now = self.tk.call("clock", "milliseconds")
        if self.predict_live.get() and int(now) - self._last_live_prediction_ms > 1200:
            self._last_live_prediction_ms = int(now)
            self.run_prediction()

        self.after(30, self._update_camera_frame)

    def _show_image(self, image: np.ndarray) -> None:
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(rgb)
        resample_filter = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
        pil_image.thumbnail(PREVIEW_SIZE, resample_filter)

        canvas = Image.new("RGB", PREVIEW_SIZE, color=(245, 246, 248))
        offset = ((PREVIEW_SIZE[0] - pil_image.width) // 2, (PREVIEW_SIZE[1] - pil_image.height) // 2)
        canvas.paste(pil_image, offset)

        self._photo_ref = ImageTk.PhotoImage(canvas)
        self.preview_label.configure(image=self._photo_ref, text="")

    def _display_result(self, result: PredictionResult) -> None:
        self.prediction_var.set(f"Classe prédite : {result.predicted_name}")
        self.confidence_var.set(result.confidence_label or "Score non disponible")
        self._write_stats(self._format_stats(result))

    def _format_stats(self, result: PredictionResult) -> str:
        lines = [
            f"Méthode : {result.method}",
            f"Classe prédite : {result.predicted_name}",
            f"Résumé : {result.summary}",
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

        return "\n".join(lines)

    def _write_stats(self, text: str) -> None:
        self.stats_text.configure(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.configure(state="disabled")

    def on_close(self) -> None:
        self.stop_camera()
        self.destroy()


def main() -> None:
    app = DefectDetectionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
