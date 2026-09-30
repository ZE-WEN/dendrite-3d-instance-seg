from pathlib import Path

from infer.infer import run_full_inference, run_iterative_inference

# change these to your own data
PROPOSAL_TIF = "/path/to/postprocessed_2d.tif"   # 3D tif (ZYX) with instance labels
RF_MODEL = "/path/to/rf_edge_classifier.pkl"
OUT_DIR = "/path/to/output"

# "full" links neighbouring slices in one pass
# "iterative" does that first, then bridges single missing slices
MODE = "full"

MIN_OVERLAP_PX = 8
P_THRESH = 0.6          # full mode
P_THRESH_DZ1 = 0.6      # iterative mode
P_THRESH_DZ2 = 0.7      # stricter, since these links skip a slice
D_CENTROID_MAX = 3.0
AREA_RATIO_MIN = 0.5

out = Path(OUT_DIR)

if MODE == "full":
    run_full_inference(
        proposal_tif_path=PROPOSAL_TIF,
        rf_model_path=RF_MODEL,
        out_edges_csv=out / "edges.csv",
        out_tracks_tif=out / "linked.tif",
        min_overlap_px=MIN_OVERLAP_PX,
        connectivity=1,
        use_skip_links=False,
        p_thresh=P_THRESH,
    )
else:
    run_iterative_inference(
        proposal_tif_path=PROPOSAL_TIF,
        rf_model_path=RF_MODEL,
        out_edges_csv_A=out / "edges_dz1.csv",
        out_tracks_tif_A=out / "linked_dz1.tif",
        out_edges_csv_B=out / "edges_all.csv",
        out_tracks_tif_final=out / "linked_final.tif",
        min_overlap_px_A=MIN_OVERLAP_PX,
        p_thresh_dz1=P_THRESH_DZ1,
        p_thresh_dz2=P_THRESH_DZ2,
        d_centroid_max=D_CENTROID_MAX,
        area_ratio_min=AREA_RATIO_MIN,
    )
