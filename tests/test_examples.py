from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_example(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_examples_can_be_called_from_python(tmp_path, deproj_samples):
    repository = Path(__file__).resolve().parents[1]
    run_sample = _load_example(repository / "examples" / "01_run_sample.py")
    gallery = _load_example(repository / "examples" / "02_plot_gallery.py")
    labeled_run = _load_example(repository / "examples" / "03_run_labeled_sample.py")
    labeled_plots = _load_example(repository / "examples" / "04_label_plots.py")
    surface_distances = _load_example(repository / "examples" / "05_surface_distances.py")
    intensities = _load_example(repository / "examples" / "06_surface_normal_intensity.py")

    _, frame, csv_path = run_sample.run_sample(
        deproj_samples / "Segmentation-2.tif",
        deproj_samples / "HeightMap-2.tif",
        tmp_path / "run_sample",
    )
    assert csv_path.is_file()
    assert len(frame) == 426

    paths = gallery.make_gallery(
        deproj_samples / "Segmentation-2.tif",
        deproj_samples / "HeightMap-2.tif",
        tmp_path / "gallery",
        dpi=80,
    )
    assert {path.name for path in paths} >= {
        "mask_objects.png",
        "area_map.png",
        "boundaries_3d.png",
        "neighbor_map.png",
        "error_maps_2x2.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)

    _, labeled_frame, labeled_csv = labeled_run.run_labeled_sample(tmp_path / "labeled_run")
    assert labeled_csv.is_file()
    assert "source_label" in labeled_frame
    labeled_paths = labeled_plots.make_labeled_plots(tmp_path / "labeled_gallery")
    assert {path.name for path in labeled_paths} >= {
        "label_objects.png",
        "area_map.png",
        "boundaries_3d.png",
        "label_objects_custom.png",
    }

    surface_output = surface_distances.run_surface_distance_example(
        deproj_samples / "Labels-2.tif",
        deproj_samples / "HeightMap-2.tif",
        tmp_path / "surface_distances",
        dpi=80,
    )
    assert surface_output["path_plot"].is_file()
    assert surface_output["pairwise"].shape[0] <= 100
    assert surface_output["distances"]["heightmap_straight_surface"] > 0
    assert surface_output["distances"]["boundary_straight_surface_pixel_input"] > 0
    assert surface_output["distances"]["boundary_graph_geodesic"] > 0

    intensity_result, intensity_regions, intensity_measurements = intensities.main(
        tmp_path / "intensity"
    )
    assert len(intensity_result.epicells) == len(intensity_regions.cells) == 4
    assert len(intensity_measurements.table) == 8
    assert (tmp_path / "intensity" / "measurements.npz").is_file()
    assert (tmp_path / "intensity" / "prism_preview.png").stat().st_size > 0
    assert (tmp_path / "intensity" / "mean_feature_map.png").stat().st_size > 0
