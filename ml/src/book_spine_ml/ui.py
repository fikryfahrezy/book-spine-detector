from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

from .inference import InferenceEngine


def _uploaded_content(value) -> bytes | None:
    if not value:
        return None
    if isinstance(value, dict):
        item = next(iter(value.values()))
    else:
        item = value[0]
    content = item["content"]
    return bytes(content)


def launch_manual_tester(checkpoint_path: str | Path = "runs/separator-baseline/best.pt"):
    import ipywidgets as widgets
    from IPython.display import clear_output, display

    checkpoint = widgets.Text(
        value=str(checkpoint_path), description="Checkpoint:", layout=widgets.Layout(width="90%")
    )
    upload = widgets.FileUpload(accept="image/*", multiple=False, description="Choose image")
    orientation = widgets.Dropdown(
        options=["auto", "vertical", "horizontal"], value="auto", description="Lines:"
    )
    pixel_threshold = widgets.FloatSlider(
        value=0.45, min=0.05, max=0.95, step=0.05, description="Pixel threshold:"
    )
    peak_height = widgets.FloatSlider(
        value=0.30, min=0.05, max=0.95, step=0.05, description="Peak height:"
    )
    prominence = widgets.FloatSlider(
        value=0.08, min=0.01, max=0.50, step=0.01, description="Prominence:"
    )
    min_distance = widgets.IntSlider(
        value=8, min=2, max=50, step=1, description="Min distance:"
    )
    run_button = widgets.Button(description="Run model", button_style="primary")
    output = widgets.Output()
    engine_cache: dict[str, InferenceEngine] = {}

    def run_model(_):
        with output:
            clear_output(wait=True)
            content = _uploaded_content(upload.value)
            if content is None:
                print("Choose an image first.")
                return
            model_path = Path(checkpoint.value).expanduser()
            if not model_path.exists():
                print(f"Checkpoint not found: {model_path}")
                return
            try:
                key = str(model_path.resolve())
                engine = engine_cache.get(key)
                if engine is None:
                    print("Loading model...")
                    engine = InferenceEngine(model_path)
                    engine_cache.clear()
                    engine_cache[key] = engine
                image = Image.open(io.BytesIO(content)).convert("RGB")
                prediction = engine.predict(
                    image,
                    orientation=orientation.value,
                    pixel_threshold=pixel_threshold.value,
                    peak_height=peak_height.value,
                    peak_prominence=prominence.value,
                    minimum_peak_distance=min_distance.value,
                )
                clear_output(wait=True)
                display(prediction.overlay)
                result = prediction.result
                print(
                    f"Count: {result.count} | orientation: {result.orientation} | "
                    f"provisional confidence: {result.confidence:.3f} | "
                    f"boundaries: {len(result.boundary_positions)}"
                )
            except Exception as error:
                print(f"Inference failed: {type(error).__name__}: {error}")

    run_button.on_click(run_model)
    controls = widgets.VBox(
        [
            checkpoint,
            widgets.HBox([upload, orientation, run_button]),
            pixel_threshold,
            peak_height,
            prominence,
            min_distance,
        ]
    )
    display(controls, output)
    return controls

