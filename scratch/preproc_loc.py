import pandas as pd

def preprocess_file(filename: str, min_lat: float, max_lat: float, min_lon: float, max_lon: float, save_as: str):
    df_all    = pd.read_csv(filename)
    print(df_all.shape)
    df_latlon = df_all[["BaseDateTime", "LAT", "LON"]]
    print(df_latlon.shape)
    df        = df_latlon.loc[(df_latlon.LAT > min_lat) & (df_latlon.LAT < max_lat) & 
                              (df_latlon.LON > min_lon) & (df_latlon.LON < max_lon)]
    print(df.shape)
    
    df[["Year", "Month", "DayTime"]] = df.BaseDateTime.str.split("-", expand=True)#.copy()
    del df["BaseDateTime"]
    
    df[["Day", "Time"]]              = df.DayTime.str.split("T", expand=True)
    del df["DayTime"]
    
    df[["Hour", "Minute", "Second"]] = df.Time.str.split(":", expand=True)
    del df["Time"]
    
    print(df)
    
    df[["Year", "Month", "Day", "Hour", "Minute", "Second"]].astype(dtype="Int64")
    df.to_csv(save_as, index=False)

    
if __name__ == "__main__":
    filename = "AIS_2023_01_01.csv"
    save_as  = "jus_vabeach.csv"
    # around va beach (lat, long)
    # 36.176017,-77.133719
    # 37.599435,-74.957005
    min_lat  = 36.176017
    max_lat  = 37.599435
    min_lon  = -77.133719
    max_lon  = -74.957005
    
    preprocess_file(filename=filename, min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, save_as=save_as)
    