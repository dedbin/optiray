import importlib
viewer = importlib.import_module("optics_viewer_plotly")

# забираем типы/структуры
OpticalElement = viewer.OpticalElement
Plate = viewer.Plate  # (на самом деле из ядра)
Ray = viewer.Ray

mirror = OpticalElement(
    surface=Plate(0.0, 0.0, 1.0, -5.0),
    interaction="reflect",
    name="mirror",
)

rays = [
    Ray(0, 1, 0, 0, 0, 1.5),
    Ray(0, 1, 0, 0, 0.1, 1.5),
]

fig = viewer.plot_system(
    rays=rays,
    elements=[mirror],
    max_events=5,
    escape_distance=50,
    title="Multi-ray reflection",
)

viewer.show_figure(fig)
