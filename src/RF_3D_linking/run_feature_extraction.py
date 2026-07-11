from derive import derive_edges_and_features

df_edges, comp_vol = derive_edges_and_features(
    gt_path_or_array=r'',
    min_overlap_px=8, 
    connectivity=1,         
    use_skip_links=False,    
    max_skip_dz=1,
    save_components_tif=r'',   
    save_edges_csv=r''       
)
