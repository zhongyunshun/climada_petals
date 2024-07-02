import os
import pandas as pd
import geopandas as gpd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point
from scipy.spatial import cKDTree
from typing import Tuple, List
import time
import warnings
warnings.filterwarnings("ignore")

# If you don't have enough memory to load all FWI and fire intensity data at once, you can load and merge them for
# shorter period like each year, and save the merged data to a file. Then, you can load the saved merged data and
# merge them for the entire period. This way, you can avoid memory issues.
class FireWeatherMerger:
    file_extensions = ['.grib', '.csv', '.nc']
    def __init__(self, geo_bound: Tuple[float, float, float, float], fwi_data_folder: str, fire_intensity_folder: str,
                 plot_on_map: bool = True, how: str = 'left', save_path: str = './'):
        """
        Initializes the FireWeatherMerger class with given parameters.

        Args:
            geo_bound (tuple): Geographic boundaries in the form (min_lon, min_lat, max_lon, max_lat).
            fwi_data_folder (str): Path to the folder containing FWI data files.
            fire_intensity_folder (str): Path to the folder containing fire intensity data files.
            plot_on_map (bool): Whether to plot the data on a map. Defaults to True.
            save_path (str): Path to save the merged data and plots. Defaults to './'.
            how (str): Method to use for the spatial join. Defaults to 'left'. Can be 'left', 'right', or 'inner'.
        """
        self.geo_bound = geo_bound
        self.fwi_data_folder = fwi_data_folder
        self.fire_intensity_folder = fire_intensity_folder
        self.plot_on_map = plot_on_map
        self.save_path = save_path
        self.merged_gdf = None
        self.how = how

    @staticmethod
    def ensure_geodataframe(df, crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
        """
        Ensures the input DataFrame is converted to a GeoDataFrame with the specified CRS.

        Args:
            df (pd.DataFrame or gpd.GeoDataFrame): Input DataFrame.
            crs (str): Coordinate reference system. Defaults to 'EPSG:4326'.

        Returns:
            gpd.GeoDataFrame: Converted GeoDataFrame.
        """
        if isinstance(df, pd.Series):
            df = pd.DataFrame([df])
        if not isinstance(df, gpd.GeoDataFrame):
            df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
        if df.crs is None:
            df.set_crs(crs, inplace=True)
        return df

    def load_data(self, folder: str) -> gpd.GeoDataFrame:
        """
        Loads data from the specified folder and filters it based on the geographic boundaries.
        One folder must only contain one type of data (e.g., '.grib', '.csv', or '.nc').
        Args:
            folder (str): Path to the folder containing data files.

        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame.
        """

        dfs = []
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            file_extension = os.path.splitext(file)[1]

            if file_extension in self.file_extensions and not file.startswith('._'):
                if file_extension in ['.grib', '.nc']:
                    df = xr.open_dataset(file_path)
                    df = df.to_dataframe().reset_index()
                    # FWI longitude ranges from [0, 360],should convert to [-180, 180]:
                    df['longitude'] = (df['longitude'] - 180) % 360 - 180
                elif file_extension in ['.csv']:
                    df = pd.read_csv(file_path)
                else:
                    raise ValueError(f"Unsupported file extension: {file_extension}")

                df = df[(df['longitude'] >= self.geo_bound[0]) & (df['longitude'] <= self.geo_bound[2]) &
                        (df['latitude'] >= self.geo_bound[1]) & (df['latitude'] <= self.geo_bound[3])]
                dfs.append(df)

        df = pd.concat(dfs)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')
        return gdf

    @staticmethod
    def find_nearest_sjoin(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame, date: pd.Timestamp, how: str) -> gpd.GeoDataFrame:
        """
        Finds the nearest neighbors between two GeoDataFrames using spatial join based on coordinates.

        Args:
            df1 (gpd.GeoDataFrame): First GeoDataFrame.
            df2 (gpd.GeoDataFrame): Second GeoDataFrame.
            date (pd.Timestamp): Date for the data being merged.
            how (str): Method to use for the spatial join. Can be 'left', 'right', or 'inner'.

        Returns:
            gpd.GeoDataFrame: Merged GeoDataFrame with nearest neighbors.
        """
        # Uncomment the following three lines to eliminate warnings about CRS
        # df1 = df1.to_crs("EPSG:4326")
        # df2 = df2.to_crs("EPSG:4326")

        merged_subdf = gpd.sjoin_nearest(df1, df2, how=how, distance_col='distance')
        merged_subdf['date'] = date
        # return merged_subdf.to_crs("EPSG:4326")
        return merged_subdf

    @staticmethod
    def find_nearest_kdtree(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame, date: pd.Timestamp) -> gpd.GeoDataFrame:
        """
        Finds the nearest neighbors between two GeoDataFrames based on spatial coordinates.

        Args:
            df1 (gpd.GeoDataFrame): First GeoDataFrame.
            df2 (gpd.GeoDataFrame): Second GeoDataFrame.
            date (datetime.date): Date for the data being merged.

        Returns:
            pd.DataFrame: Merged DataFrame with nearest neighbors.
        """
        tree = cKDTree(np.c_[df2.geometry.x, df2.geometry.y])
        distances, indices = tree.query(np.c_[df1.geometry.x, df1.geometry.y], k=1)
        nearest_df2 = df2.iloc[indices].reset_index(drop=True)
        nearest_df2.columns = [f"{col}_nearest" for col in nearest_df2.columns]
        merged_subdf = pd.concat([df1.reset_index(drop=True), nearest_df2, pd.Series(distances, name='distance')], axis=1)
        merged_subdf['date'] = date
        # pd.concate returns a DataFrame rather than a GeoDataFrame.
        # After concatenating the DataFrames, should convert the result back to a GeoDataFrame.
        merged_subdf = gpd.GeoDataFrame(merged_subdf, geometry=merged_subdf.geometry, crs=df1.crs)
        return merged_subdf

    def merge_data(self) -> gpd.GeoDataFrame:
        """
        Merges the FWI data and fire intensity data based on the nearest coordinates and same date.

        Returns:
            gpd.GeoDataFrame: Merged GeoDataFrame.
        """
        fwi_gdf = self.load_data(self.fwi_data_folder)
        fire_intensity_gdf = self.load_data(self.fire_intensity_folder)

        # 'Time' in fwi_gdf is index so reset it to a column
        fwi_gdf.reset_index(inplace=True)

        fwi_gdf['time'] = pd.to_datetime(fwi_gdf['time']).dt.date
        fire_intensity_gdf['acq_date'] = pd.to_datetime(fire_intensity_gdf['acq_date']).dt.date

        fire_intensity_gdf.rename(columns={'acq_date': 'date'}, inplace=True)
        fwi_gdf.rename(columns={'time': 'date'}, inplace=True)

        fire_intensity_gdf = fire_intensity_gdf.set_index('date')
        fwi_gdf = fwi_gdf.set_index('date')

        merged_dfs = []
        fwi_dates_set = set(fwi_gdf.index.unique()) # unique dates in fwi_gdf, convert to set to reduce runtime
        # Iterate over unique dates in fire_intensity_gdf and find the nearest neighbors in fwi_gdf
        # Iterate fire_intensity_gdf first because it has fewer unique dates whereas fwi_gdf has daily data
        for date in fire_intensity_gdf.index.unique():
            if date in fwi_dates_set:
                fire_subset = FireWeatherMerger.ensure_geodataframe(fire_intensity_gdf.loc[[date]])
                fwi_subset = FireWeatherMerger.ensure_geodataframe(fwi_gdf.loc[[date]])
                # average fire intensity value for same fwi_lat, fwi_lon, Distance in the merged data

                # Choose one of the following methods to find the nearest neighbors
                # merged_dfs.append(self.find_nearest_kdtree(fire_subset, fwi_subset, date))
                merged_dfs.append(self.find_nearest_sjoin(fire_subset, fwi_subset, date, self.how))

        if len(merged_dfs) == 0:
            raise ValueError("No common dates found between the two datasets.")
        elif len(merged_dfs) == 1:
            self.merged_gdf = merged_dfs[0]
        else:
            self.merged_gdf = pd.concat(merged_dfs, ignore_index=True)


        # fwi names are different for two fine_nearest methods
        if 'fwinx_nearest' in self.merged_gdf.columns:
            self.merged_gdf.rename(columns={'fwinx_nearest': 'fwi'}, inplace=True)
        elif 'fwinx' in self.merged_gdf.columns:
            self.merged_gdf.rename(columns={'fwinx': 'fwi'}, inplace=True)

        # pd.concate returns a DataFrame rather than a GeoDataFrame.
        # After concatenating the DataFrames, should convert the result back to a GeoDataFrame.
        self.merged_gdf = gpd.GeoDataFrame(self.merged_gdf, geometry=self.merged_gdf.geometry, crs=self.merged_gdf.crs)
        return self.merged_gdf

    def plot_data(self, plot_type: str) -> None:
        """
        Plots the merged data on a map with the specified geographic boundaries and saves the plot.

        Args:
            plot_type (str): Type of data to plot ('fwi' or 'fire_intensity').
        """
        if self.merged_gdf is not None:
            fig, ax = plt.subplots(figsize=(10, 10))
            if plot_type == 'fwi':
                self.merged_gdf.plot(ax=ax, column='fwi', legend=True, cmap='hot', markersize=5)
            elif plot_type == 'fire_intensity':
                self.merged_gdf.plot(ax=ax, column='brightness', legend=True, cmap='hot', markersize=5)

            ax.set_xlim(self.geo_bound[0], self.geo_bound[2])
            ax.set_ylim(self.geo_bound[1], self.geo_bound[3])
            ctx.add_basemap(ax, crs="EPSG:4326", source=ctx.providers.OpenStreetMap.Mapnik)
            plt.savefig(os.path.join(self.save_path, f'{plot_type}_data_plot.png'))
        else:
            raise ValueError("Merged GeoDataFrame is empty. Run merge_data() before plotting.")

    def run(self) -> gpd.GeoDataFrame:
        """
        Runs the entire process of loading, merging, and optionally plotting the data.

        Returns:
            gpd.GeoDataFrame: Merged GeoDataFrame.
        """
        merged_gdf = self.merge_data()
        if self.plot_on_map:
            self.plot_data(plot_type='fwi')
            self.plot_data(plot_type='fire_intensity')
        return merged_gdf


# Example usage:
# min_lon, min_lat, max_lon, max_lat
geo_bound_uk = (-10, 49, 2, 61)
geo_bound_canada = (-141, 49, -52, 83) # actually 42 instead of 49
geo_bound_usa = (-125, 24, -66, 49)
geo_bound_eu = (-31, 34, 40, 72)
year = 2013
areas = ['uk', 'canada', 'usa', 'eu']
merge_methods = ['left', 'right']
geo_bounds = zip(areas, [geo_bound_uk, geo_bound_canada, geo_bound_usa, geo_bound_eu])

fwi_folder = f'../../climada_petals/data/wildfire/copernicus_fwi/interpolated25_netcdf/{year}/'
fire_intensity_folder = f'../../climada_petals/data/wildfire/nasa_fire_intensity/{year}/'
# fwi_folder = '../../climada_petals/data/wildfire/2001_fwi_fire_intensity_expriment/'
# fire_intensity_folder = '../../climada_petals/data/wildfire/2001_fwi_fire_intensity_expriment/fire_intensity_csv/'
save_path = f'../../climada_petals/data/wildfire/output/{year}/'

for merge_method in merge_methods:
    for area, geo_bound in geo_bounds:
        start_time = time.time()
        print(f"Processing data for {area} area, year: {year}, merge method: {merge_method}")
        merger = FireWeatherMerger(geo_bound, fwi_folder, fire_intensity_folder, plot_on_map=False,  how=merge_method, save_path=save_path)
        merged_gdf = merger.run()

        # convert date to string because GeoPackage driver does not support the datetime.date type directly
        merged_gdf['date'] = merged_gdf['date'].astype(str)
        # drop unnecessary columns or columns containing datetime object because GeoPackage driver does not support them
        columns_to_drop = ['index_left', 'index_right', 'valid_time', 'step', 'scan', 'track', 'acq_time', 'version', 'type']
        for column in columns_to_drop:
            if column in merged_gdf.columns:
                merged_gdf.drop(columns=[column], inplace=True)

        merged_gdf.to_file(os.path.join(save_path, f'merged_{area}_{year}_{merge_method}_gdf'), driver='GPKG')
        print(f"Saved merged GeoDataFrame to {save_path}merged_{area}_{year}_{merge_method}_gdf")
        print(f"Execution time: {time.time() - start_time} seconds")