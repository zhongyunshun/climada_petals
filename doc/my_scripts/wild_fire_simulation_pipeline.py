import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from climada_petals.hazard import WildFire
from climada.hazard import Centroids
from climada.util.constants import ONE_LAT_KM
from climada_petals.hazard.wildfire import WildFireFuture


class FireRiskModel:
    """
    A model to simulate wildfire risks using historical fire data and ML algorithms predicted outputs.
    """
    def __init__(self, wf_file_path, random_seed=42, wf_resolution_km=1.0,
                 wf_resolution_arcsec=30, geo_bound=(-10, 49, 2, 61), prop_proba=0.21):
        """
        Initializes the FireRiskModel with specified attributes.

        Args:
            wf_file_path (str): Path to the wildfire historical FIRMS data file.
            random_seed (int): Seed for the random number generator to ensure reproducibility.
            wf_resolution_km (float): Spatial resolution of the wildfire data in kilometers.
            wf_resolution_arcsec (float): Spatial resolution in arcseconds.
            geo_bound (tuple): Geographic boundary (longitude and latitude) as (lon_min, lat_min, lon_max, lat_max).
            prop_proba (float): overall propagation probability for wildfire occurrences.
        """
        self.wf_file_path = wf_file_path
        self.random_seed = random_seed
        self.wf_resolution_km = wf_resolution_km
        self.wf_resolution_arcsec = wf_resolution_arcsec
        self.geo_bound = geo_bound
        self.prop_proba = prop_proba
        np.random.seed(self.random_seed)  # Ensure reproducible results

    def load_data(self):
        """
        Loads wildfire FIRMS data from a file and filters it based on geographic boundaries.

        Returns:
            pandas.DataFrame: Dataframe containing wildfire data filtered by geographic region.
        """
        df_firms = pd.read_csv(os.path.join('../climada_petals/data', self.wf_file_path))
        return self.select_region(df_firms, self.geo_bound)

    @staticmethod
    def select_region(df, bound):
        """
        Select the region of interest from the dataframe.
        df: hazard dataframe
        bound: tuple in format (lon_min, lat_min, lon_max, lat_max)
        """
        return df[(df['longitude'] >= bound[0]) &
                  (df['latitude'] >= bound[1]) &
                  (df['longitude'] <= bound[2]) &
                  (df['latitude'] <= bound[3])]

    def simulate_fire_probability_array_from_ML_algorithms(self):
        """
        Simulates the outputs from ML algorithms based on resolution settings.

        Returns:
            tuple: A tuple containing two lists, one for the coordinates of the centroids
            and one for the predicted probabilities of the centroids.
        """
        res_deg = self.wf_resolution_arcsec / ONE_LAT_KM
        centroids = Centroids.from_pnt_bounds(self.geo_bound, res_deg)
        centroids.set_meta_to_lat_lon()
        centr_lonlats = list(zip(centroids.lon, centroids.lat))
        fire_probs = np.random.rand(len(centr_lonlats))
        return centr_lonlats, fire_probs

    @staticmethod
    def generate_coordinates_of_ignited_points(centr_lonlats, fire_probs, num_fires):
        """
        Draw coordinates for ignited points based on multinomial probabilities.

        Args:
            centr_lonlats (list of tuple): List of tuples with longitude and latitude coordinates of the centroids.
            fire_probs (list of float): List of probabilities corresponding to each coordinate.
            num_fires (int): Number of fires to simulate.

        Returns:
            list: List of tuples representing the coordinates of ignited points.
        """
        normalized_probs = fire_probs / np.sum(fire_probs)
        draws = np.random.choice(len(normalized_probs), size=num_fires, p=normalized_probs)
        ignited_centr_lonlats = [centr_lonlats[idx] for idx in draws]
        return ignited_centr_lonlats

    def run_simulation(self, centr_lonlats=None, fire_probs=None):
        """
        Runs the wildfire simulation using either provided or simulated data.

        Args:
            centr_lonlats (list of tuple, optional): Predefined list of longitude and latitude coordinates
            of ignited pixels.
            fire_probs (list of float, optional): Predefined list of probabilities for ignited pixels.

        Returns:
            WildFireFuture: An WildFire object representing the simulated wildfire data.
        """
        df_firms = self.load_data()

        wf = WildFireFuture()
        wf.set_hist_fire_seasons_FIRMS(df_firms, centr_res_factor=1. / self.wf_resolution_km)

        num_fires = int(wf.n_fires.item())
        ign_range = [num_fires, num_fires + 1]

        if centr_lonlats is None or fire_probs is None:
            centr_lonlats, fire_probs = self.simulate_fire_probability_array_from_ML_algorithms()

        ignited_centr_lonlats = self.generate_coordinates_of_ignited_points(centr_lonlats, fire_probs, num_fires)

        centr_id_list = [wf.centroids.get_closest_point(lon, lat)[2] for lon, lat in ignited_centr_lonlats]
        wf.ProbaParams.prop_proba = self.prop_proba

        wf.set_proba_fire_seasons(n_fire_seasons=1, n_ignitions=ign_range, fired_id_list=centr_id_list,
                                  keep_all_fires=True)

        print('The probabilistic season is appended to the historic season:',
              "event_names", wf.event_name,
              "event_ids", wf.event_id)

        return wf


# Example of using the class
model = FireRiskModel('modis_2018_United_Kingdom.csv')
wf = model.run_simulation()
wf.plot_intensity(event=wf.event_id[2])