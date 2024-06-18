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

# If you don't have enough memory to load all FWI and fire intensity data at once, you can load and merge them for
# shorter period like each year, and save the merged data to a file. Then, you can load the saved merged data and
# merge them for the entire period. This way, you can avoid memory issues.
class FireWeatherMerger:
    def __init__(self, geo_bound: Tuple[float, float, float, float], fwi_data_folder: str, fire_intensity_folder: str,
                 plot_on_map: bool = True, save_path: str = './'):
        """
        Initializes the FireWeatherMerger class with given parameters.

        Args:
            geo_bound (tuple): Geographic boundaries in the form (min_lon, min_lat, max_lon, max_lat).
            fwi_data_folder (str): Path to the folder containing FWI data files.
            fire_intensity_folder (str): Path to the folder containing fire intensity data files.
            plot_on_map (bool): Whether to plot the data on a map. Defaults to True.
            save_path (str): Path to save the merged data and plots. Defaults to './'.
        """
        self.geo_bound = geo_bound
        self.fwi_data_folder = fwi_data_folder
        self.fire_intensity_folder = fire_intensity_folder
        self.plot_on_map = plot_on_map
        self.save_path = save_path
        self.merged_gdf = None

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

    def load_data(self, folder: str, file_extension: str, datatype: str) -> gpd.GeoDataFrame:
        """
        Loads data from the specified folder and filters it based on the geographic boundaries.
        One folder must only contain one type of data (FWI or fire intensity).
        Args:
            folder (str): Path to the folder containing data files.
            file_extension (str): File extension to look for (e.g., '.grib', '.csv').
            datatype (str): Whether the data being loaded is FWI data 'fwi' or fire intensity data 'fire_intensity'.

        Returns:
            gpd.GeoDataFrame: Filtered GeoDataFrame.
        """

        dfs = []
        for file in os.listdir(folder):
            if file.endswith(file_extension) and not file.startswith('._'):
                if datatype == 'fwi':
                    ds = xr.open_dataset(os.path.join(folder, file), engine='cfgrib')
                    df = ds.to_dataframe()
                elif datatype == 'fire_intensity':
                    df = pd.read_csv(os.path.join(folder, file))
                else:
                    raise ValueError("Invalid datatype. Choose 'fwi' or 'fire_intensity'.")

                df = df[(df['longitude'] >= self.geo_bound[0]) & (df['longitude'] <= self.geo_bound[2]) &
                        (df['latitude'] >= self.geo_bound[1]) & (df['latitude'] <= self.geo_bound[3])]
                dfs.append(df)

        df = pd.concat(dfs)
        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['longitude'], df['latitude']), crs='EPSG:4326')
        return gdf

    @staticmethod
    def find_nearest_sjoin(df1: gpd.GeoDataFrame, df2: gpd.GeoDataFrame, date: pd.Timestamp) -> gpd.GeoDataFrame:
        """
        Finds the nearest neighbors between two GeoDataFrames using spatial join based on coordinates.

        Args:
            df1 (gpd.GeoDataFrame): First GeoDataFrame.
            df2 (gpd.GeoDataFrame): Second GeoDataFrame.
            date (pd.Timestamp): Date for the data being merged.

        Returns:
            gpd.GeoDataFrame: Merged GeoDataFrame with nearest neighbors.
        """
        # Uncomment the following three lines to eliminate warnings about CRS
        # df1 = df1.to_crs("EPSG:4326")
        # df2 = df2.to_crs("EPSG:4326")

        merged_subdf = gpd.sjoin_nearest(df1, df2, how='left', distance_col='distance')
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
        fwi_gdf = self.load_data(self.fwi_data_folder, '.grib', datatype='fwi')
        fire_intensity_gdf = self.load_data(self.fire_intensity_folder, '.csv', datatype='fire_intensity')

        # 'Time' in fwi_gdf is index so reset it to a column
        fwi_gdf.reset_index(inplace=True)
        fwi_gdf['time'] = pd.to_datetime(fwi_gdf['time']).dt.date
        fire_intensity_gdf['acq_date'] = pd.to_datetime(fire_intensity_gdf['acq_date']).dt.date

        fire_intensity_gdf.rename(columns={'acq_date': 'date'}, inplace=True)
        fwi_gdf.rename(columns={'time': 'date'}, inplace=True)

        fire_intensity_gdf = fire_intensity_gdf.set_index('date')
        fwi_gdf = fwi_gdf.set_index('date')

        merged_dfs = []

        # Iterate over unique dates in fire_intensity_gdf and find the nearest neighbors in fwi_gdf
        # Iterate fire_intensity_gdf first because it has fewer unique dates whereas fwi_gdf has daily data
        for date in fire_intensity_gdf.index.unique():
            if date in fwi_gdf.index.unique():
                fire_subset = FireWeatherMerger.ensure_geodataframe(fire_intensity_gdf.loc[[date]])
                fwi_subset = FireWeatherMerger.ensure_geodataframe(fwi_gdf.loc[[date]])
                # average fire intensity value for same fwi_lat, fwi_lon, Distance in the merged data

                # Choose one of the following methods to find the nearest neighbors
                # merged_dfs.append(self.find_nearest_kdtree(fire_subset, fwi_subset, date))
                merged_dfs.append(self.find_nearest_sjoin(fire_subset, fwi_subset, date))

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
geo_bound_uk = (-9, 34, 32, 72)
fwi_folder = '../../climada_petals/data/wildfire/copernicus_fwi/'
fire_intensity_folder = '../../climada_petals/data/wildfire/nasa_fire_intensity/'
# fwi_folder = '../../climada_petals/data/wildfire/2001_fwi_fire_intensity_expriment/'
# fire_intensity_folder = '../../climada_petals/data/wildfire/2001_fwi_fire_intensity_expriment/fire_intensity_csv/'
save_path = '../../climada_petals/data/wildfire/output/'

merger = FireWeatherMerger(geo_bound_uk, fwi_folder, fire_intensity_folder, plot_on_map=True, save_path=save_path)
merged_gdf = merger.run()

# convert date to string because GeoPackage driver does not support the datetime.date type directly
merged_gdf['date'] = merged_gdf['date'].astype(str)
merged_gdf.drop(columns=['index_right', 'scan', 'track', 'acq_time', 'version', 'type'], inplace=True)
merged_gdf.to_file(os.path.join(save_path, 'merged_eu_2001_gdf'), driver='GPKG')