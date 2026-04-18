import cv2
import numpy as np

from src.preprocessing import prepare_image
from src.segmentation import segment_main_object_circle, apply_mask

def unwrap_annulus(gray_img, center, r_inner, r_outer, n_angles=360, n_radii=24):
    cx, cy = float(center[0]), float(center[1])

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)
    radii = np.linspace(r_inner, r_outer, n_radii).astype(np.float32)

    map_x = cx + np.outer(radii, np.cos(angles))
    map_y = cy + np.outer(radii, np.sin(angles))

    strip = cv2.remap(
        gray_img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return strip

def normalize_strip(strip):
    strip = strip.astype(np.float32)
    return (strip - strip.mean()) / (strip.std() + 1e-6)

def filter_small_components(binary_mask, min_pixels=10):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    cleaned = np.zeros_like(binary_mask)

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_pixels:
            cleaned[labels == label] = 255

    return cleaned

def longest_true_run_circular(bool_array):
    n = len(bool_array)
    if n == 0:
        return 0
    if np.all(bool_array):
        return n

    doubled = np.concatenate([bool_array, bool_array])

    best = 0
    current = 0

    for val in doubled:
        if val:
            current += 1
            best = max(best, current)
        else:
            current = 0

    return min(best, n)

def rewrap_binary_strip_to_mask(binary_strip, image_shape, center, r_inner, r_outer):
    H, W = image_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    n_radii, n_angles = binary_strip.shape
    cx, cy = float(center[0]), float(center[1])

    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    radii = np.linspace(r_inner, r_outer, n_radii)

    rr, aa = np.meshgrid(radii, angles, indexing="ij")

    xs = cx + rr * np.cos(aa)
    ys = cy + rr * np.sin(aa)

    xs = np.round(xs).astype(int)
    ys = np.round(ys).astype(int)

    valid = (
        (xs >= 0) & (xs < W) &
        (ys >= 0) & (ys < H) &
        (binary_strip > 0)
    )

    mask[ys[valid], xs[valid]] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

    return mask

def analyze_strip_against_reference(strip_norm, ref, z_thresh=2.2, min_pixels=10, col_threshold=0.05):
    zmap = np.abs(strip_norm - ref["mean"]) / (ref["std"] + 0.35)
    zmap_blur = cv2.GaussianBlur(zmap.astype(np.float32), (3, 3), 0)

    binary = np.zeros_like(zmap_blur, dtype=np.uint8)
    binary[zmap_blur >= z_thresh] = 255

    binary = filter_small_components(binary, min_pixels=min_pixels)
    binary_bool = binary > 0

    abnormal_ratio = float(binary_bool.mean())

    col_profile = binary_bool.mean(axis=0)
    abnormal_cols = col_profile >= col_threshold

    abnormal_cols_ratio = float(abnormal_cols.mean())
    longest_run_ratio = float(longest_true_run_circular(abnormal_cols) / len(abnormal_cols))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    areas = []
    for label in range(1, num_labels):
        areas.append(stats[label, cv2.CC_STAT_AREA])

    num_components = len(areas)
    max_component_area = int(max(areas)) if len(areas) > 0 else 0

    return {
        "zmap": zmap_blur,
        "binary": binary,
        "abnormal_ratio": abnormal_ratio,
        "abnormal_cols_ratio": abnormal_cols_ratio,
        "longest_run_ratio": longest_run_ratio,
        "num_components": num_components,
        "max_component_area": max_component_area
    }

def build_reference(image_paths, cfg, load_grayscale_image, n_reference=80, seed=42):
    import random

    random.seed(seed)
    selected = random.sample(image_paths, min(n_reference, len(image_paths)))

    strips = []
    used_paths = []

    for path in selected:
        try:
            img = load_grayscale_image(path)
            prepared = prepare_image(img)
            main = segment_main_object_circle(prepared)

            center = main["center"]
            radius = main["radius"]

            r_inner = cfg["r_inner_frac"] * radius
            r_outer = cfg["r_outer_frac"] * radius

            strip = unwrap_annulus(
                prepared,
                center=center,
                r_inner=r_inner,
                r_outer=r_outer,
                n_angles=cfg["n_angles"],
                n_radii=cfg["n_radii"]
            )

            strip_norm = normalize_strip(strip)
            strips.append(strip_norm)
            used_paths.append(path)

        except Exception:
            continue

    strips = np.stack(strips, axis=0)

    return {
        "mean": strips.mean(axis=0),
        "std": strips.std(axis=0),
        "used_paths": used_paths,
        "stack": strips
    }

def analyze_image_with_references(img, outer_ref, inner_ref, outer_cfg, inner_cfg):
    prepared = prepare_image(img)
    main = segment_main_object_circle(prepared)

    center = main["center"]
    radius = main["radius"]
    mask = main["mask"]

    piece_only = apply_mask(prepared, mask)

    outer_r_inner = outer_cfg["r_inner_frac"] * radius
    outer_r_outer = outer_cfg["r_outer_frac"] * radius

    outer_strip = unwrap_annulus(
        prepared,
        center=center,
        r_inner=outer_r_inner,
        r_outer=outer_r_outer,
        n_angles=outer_cfg["n_angles"],
        n_radii=outer_cfg["n_radii"]
    )
    outer_strip_norm = normalize_strip(outer_strip)

    outer_res = analyze_strip_against_reference(
        outer_strip_norm,
        outer_ref,
        z_thresh=outer_cfg["z_thresh"],
        min_pixels=outer_cfg["min_pixels"],
        col_threshold=outer_cfg["col_threshold"]
    )

    outer_mask_img = rewrap_binary_strip_to_mask(
        outer_res["binary"],
        image_shape=prepared.shape,
        center=center,
        r_inner=outer_r_inner,
        r_outer=outer_r_outer
    )

    inner_r_inner = inner_cfg["r_inner_frac"] * radius
    inner_r_outer = inner_cfg["r_outer_frac"] * radius

    inner_strip = unwrap_annulus(
        prepared,
        center=center,
        r_inner=inner_r_inner,
        r_outer=inner_r_outer,
        n_angles=inner_cfg["n_angles"],
        n_radii=inner_cfg["n_radii"]
    )
    inner_strip_norm = normalize_strip(inner_strip)

    inner_res = analyze_strip_against_reference(
        inner_strip_norm,
        inner_ref,
        z_thresh=inner_cfg["z_thresh"],
        min_pixels=inner_cfg["min_pixels"],
        col_threshold=inner_cfg["col_threshold"]
    )

    inner_mask_img = rewrap_binary_strip_to_mask(
        inner_res["binary"],
        image_shape=prepared.shape,
        center=center,
        r_inner=inner_r_inner,
        r_outer=inner_r_outer
    )

    return {
        "prepared": prepared,
        "piece_only": piece_only,
        "mask": mask,
        "center": center,
        "radius": radius,
        "outer_strip": outer_strip,
        "outer_res": outer_res,
        "outer_mask_img": outer_mask_img,
        "inner_strip": inner_strip,
        "inner_res": inner_res,
        "inner_mask_img": inner_mask_img
    }