import pandas as pd
import glob
import os

# folder-to-folder
def filter_loc(df: pd.DataFrame, loc_filter_args: dict) -> pd.DataFrame:
    # unpacking args for legibility of the code below
    #source_path = loc_filter_args["source_path"]
    min_lon     = loc_filter_args["min_lon"]
    max_lon     = loc_filter_args["max_lon"]
    min_lat     = loc_filter_args["min_lat"]
    max_lat     = loc_filter_args["max_lat"]
    #dest_path   = loc_filter_args["dest_path"]
    
    #df         = pd.read_csv(source_path)                        # open it
    df_reduced = df[(df.LON >= min_lon) & (df.LON <= max_lon) &  # filter longitude
                    (df.LAT >= min_lat) & (df.LAT <= max_lat)]   # filter latitude
    #df_reduced.to_csv(dest_path, index=False)
    return df_reduced

# end is not inclusive
# save name should not include the extension
def open_filter_combine_save(filelist: list, start_ndx: int, end_ndx: int, loc_filter_args: dict, save_name: str):
    print("working on", save_name, "with files", start_ndx, "(inclusive) to", end_ndx, "(not inclusive)")
    dest_dir = loc_filter_args["dest_dir"]
    df_list  = []
    for ndx in range(start_ndx, end_ndx):
        name = filelist[ndx]
        print("opening", name)
        df   = pd.read_csv(name)
        print("processing", name)
        df   = filter_loc(df, loc_filter_args=loc_filter_args)
        df_list.append(df)
        
    all_df    = pd.concat(df_list, axis=1)
    all_df.drop_duplicates()
    save_as   = save_name + ".csv"
    save_path = os.path.join(dest_dir, save_as)
    all_df.to_csv(save_path)
    
def filter_folder(loc_filter_args: dict):
    source_dir = loc_filter_args["source_dir"]
    train_frac = loc_filter_args["train_frac"]
    val_frac   = loc_filter_args["val_frac"]
    test_frac  = loc_filter_args["test_frac"]
    filelist   = sorted(glob.glob(os.path.join(source_dir, "*.csv")))
    num_files  = len(filelist)
    print("found", num_files, "files in", source_dir)
    
    train_start = 0                                    # inclusive
    train_end   = train_start + int(num_files * train_frac) # not inclusive
    val_start   = train_end                            # inclusive
    val_end     = val_start   + int(num_files * val_frac)   # not inclusive
    test_start  = val_end                              # inclusive
    test_end    = test_start  + int(num_files * test_frac) # not inclusive, obv.
    
    open_filter_combine_save(filelist=filelist, start_ndx=train_start, end_ndx=train_end, loc_filter_args=loc_filter_args, save_name="train")
    open_filter_combine_save(filelist=filelist, start_ndx=val_start,   end_ndx=val_end,   loc_filter_args=loc_filter_args, save_name="val"  )
    open_filter_combine_save(filelist=filelist, start_ndx=test_start,  end_ndx=test_end,  loc_filter_args=loc_filter_args, save_name="test" )
    
if __name__ == "__main__":
    loc_filter_args = {}
    # off the east coast
    loc_filter_args["min_lon"] = -70 
    loc_filter_args["max_lon"] = -60
    loc_filter_args["min_lat"] =  30
    loc_filter_args["max_lat"] =  40
    loc_filter_args["source_dir"] = "source_data"
    loc_filter_args["dest_dir"]   = "reduced_data"
    loc_filter_args["train_frac"] = 3./5.
    loc_filter_args["val_frac"]   = 1./5.
    loc_filter_args["test_frac"]  = 1./5.
    
    filter_folder(loc_filter_args=loc_filter_args)
