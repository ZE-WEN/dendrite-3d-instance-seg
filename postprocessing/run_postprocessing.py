import multiprocessing as mp
from pathlib import Path

from pipeline_utils import (
    downsample_nearest,
    read_tiff_stack,
    run_after_mg,
    run_mg,
    run_repair,
    save_tiff_stack,
)

# change these to your own data
INPUT_PATH = "/path/to/refined.tif"   # 3D instance labels (Z, Y, X)
OUT_DIR = "/path/to/output"
NAME = "dataset"                      # prefix for the output files

USE_MEMMAP = False   # read the input with memmap, useful for very large tifs
TARGET_XY = 1024     # slices are downsampled to this size before the repair

# repair in the x view
REPAIR = {
    "solidity_min": 0.85,
    "eccentricity_max": 0.55,
    "min_area": 50,
    "coverage_min": 0.60,
    "centroid_dist_max": 20.0,
    "bbox_iou_min": 0.30,
    "min_overlap_pixels": 3,
    "closing_radius": 2,
    "bbox_margin": 3,
    "fill_holes": True,
    "run_backward_pass": True,
    "num_workers": 20,
    "chunk_size": 64,
}

MIN_COMPONENT_AREA = 500   # merge step, smaller pieces get reassigned
OVERLAP_THRESH = 0.95      # id harmonization over a 5-slice window


def main():
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    downsampled = out / f"{NAME}_downsampled.tif"
    repaired = out / f"{NAME}_repaired.tif"
    repair_log = out / f"{NAME}_repair_log.csv"
    merged = out / f"{NAME}_merged.tif"
    final = out / f"{NAME}_final.tif"

    print(f"1/4 downsampling to {TARGET_XY} x {TARGET_XY}")
    vol = read_tiff_stack(INPUT_PATH, memmap=USE_MEMMAP)
    save_tiff_stack(str(downsampled), downsample_nearest(vol, target_xy=TARGET_XY))
    del vol

    print("2/4 repair in the x view")
    run_repair(input_tif=str(downsampled), output_tif=str(repaired),
               log_csv=str(repair_log), cfg=REPAIR, memmap=USE_MEMMAP)

    print("3/4 merging and removing small pieces")
    run_mg(vol1_path=downsampled, vol2_path=repaired, output_path=merged,
           min_component_area=MIN_COMPONENT_AREA)

    print("4/4 harmonizing ids across slices")
    run_after_mg(input_path=merged, output_path=final, overlap_thresh=OVERLAP_THRESH)

    print("done, final result:", final)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # the repair runs in parallel
    main()
