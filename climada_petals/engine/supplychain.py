"""
This file is part of CLIMADA.

Copyright (C) 2017 ETH Zurich, CLIMADA contributors listed in AUTHORS.

CLIMADA is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free
Software Foundation, version 3.

CLIMADA is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with CLIMADA. If not, see <https://www.gnu.org/licenses/>.

---

Define the SupplyChain class.
"""

__all__ = ["SupplyChain"]

import logging
from pathlib import Path
import zipfile
import warnings
import pandas as pd
import numpy as np

import pymrio

from boario.extended_models import ARIOPsiModel
from boario.event import EventKapitalRecover
from boario.simulation import Simulation

from climada import CONFIG
from climada.util import files_handler as u_fh
import climada.util.coordinates as u_coord

LOGGER = logging.getLogger(__name__)
WIOD_FILE_LINK = CONFIG.engine.supplychain.resources.wiod16.str()
"""Link to the 2016 release of the WIOD tables."""

MRIOT_DIRECTORY = CONFIG.engine.supplychain.local_data.mriot.dir()
"""Directory where Multi-Regional Input-Output Tables (MRIOT) are downloaded."""

calc_G = pymrio.calc_L

def parse_mriot_from_df(
        mriot_df=None, col_iso3=None, col_sectors=None, 
        rows_data=None, cols_data=None, row_fd_cats=None
        ):
    """Build multi-index dataframes of the transaction matrix, final demand and total
       production from a Multi-Regional Input-Output Table dataframe.

    Parameters
    ----------
    v : pandas.DataFrame or numpy.array
        a row vector of the total final added-value
    G : pandas.DataFrame or numpy.array
        Symmetric input output Ghosh table
    """

    start_row, end_row = rows_data
    start_col, end_col = cols_data

    sectors = mriot_df.iloc[start_row:end_row, col_sectors].unique()
    regions = mriot_df.iloc[start_row:end_row, col_iso3].unique()
    fd_cats = mriot_df.iloc[row_fd_cats, end_col:-1].unique()
    multiindex = pd.MultiIndex.from_product(
        [regions, sectors], names=["region", "sector"]
    )

    multiindex_final_demand = pd.MultiIndex.from_product(
                [regions, fd_cats], names = ['region', 'category'])

    Z = mriot_df.iloc[start_row:end_row, start_col:end_col].values.astype(float)
    Z = pd.DataFrame(data=Z, index=multiindex, columns=multiindex)

    Y = mriot_df.iloc[start_row:end_row, end_col:-1].values.astype(float)
    Y = pd.DataFrame(
                    data = Y,
                    index = multiindex,
                    columns = multiindex_final_demand
                    )

    x = mriot_df.iloc[start_row:end_row, -1].values.astype(float)
    x = pd.DataFrame(data=x, index=multiindex, columns=["total production"])

    return Z, Y, x

def calc_v(Z, x):
    """Calculate value added (v) from Z and x

    value added = industry output (x) - inter-industry inputs (sum_rows(Z))

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric input output table (flows)
    x : pandas.DataFrame or numpy.array
        industry output

    Returns
    -------
    pandas.DataFrame or numpy.array
        Value added v as row vector
    """

    value_added = np.diff(np.vstack((Z.sum(0), x.T)), axis=0)
    if isinstance(Z, pd.DataFrame):
        value_added = pd.DataFrame(value_added, columns=Z.index, index=["indout"])
    if isinstance(value_added, pd.Series):
        value_added = pd.DataFrame(value_added)
    if isinstance(value_added, pd.DataFrame):
        value_added.index = ["indout"]
    return value_added


def calc_B(Z, x):
    """Calculate the B matrix (allocation coefficients matrix)
    from Z matrix and x vector

    Parameters
    ----------
    Z : pandas.DataFrame or numpy.array
        Symmetric input output table (flows)
    x : pandas.DataFrame or numpy.array
        Industry output column vector

    Returns
    -------
    pandas.DataFrame or numpy.array
        Symmetric input output table (allocation matrix) B
        The type is determined by the type of Z.
        If DataFrame index/columns as Z

    Notes
    -----
    This function adapts pymrio.tools.iomath.calc_A to compute
    the allocation coefficients matrix B.
    """

    if isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_numpy()
    if (type(x) is not np.ndarray) and (x == 0):
        recix = 0
    else:
        with warnings.catch_warnings():
            # Ignore devide by zero warning, we set to 0 afterwards
            warnings.filterwarnings("ignore", message="divide by zero")
            recix = 1 / x
        recix[np.isinf(recix)] = 0

    if isinstance(Z, pd.DataFrame):
        return pd.DataFrame(Z.to_numpy() * recix, index=Z.index, columns=Z.columns)
    else:
        return Z * recix


def calc_x_from_G(G, v):
    """Calculate the industry output x from a v vector and G matrix

    x = vG

    The industry output x is computed from a value-added vector v

    Parameters
    ----------
    v : pandas.DataFrame or numpy.array
        a row vector of the total final added-value
    G : pandas.DataFrame or numpy.array
        Symmetric input output Ghosh table

    Returns
    -------
    pandas.DataFrame or numpy.array
        Industry output x as column vector
        The type is determined by the type of G. If DataFrame index as G

    Notes
    -----
    This function adapts the function pymrio.tools.iomath.calc_x_from_L to
    compute total output (x) from the Ghosh inverse.
    """

    x = v.dot(G)
    if isinstance(x, pd.Series):
        x = pd.DataFrame(x)
    if isinstance(x, pd.DataFrame):
        x.columns = ["indout"]
    return x


def mriot_file_name(mriot_type, mriot_year):
    """Retrieve the original EXIOBASE3, WIOD16 or OECD21 MRIOT file name

    Parameters
    ----------
    mriot_type : string
    mriot_year : int
    """

    if mriot_type == "EXIOBASE3":
        return f"IOT_{mriot_year}_ixi.zip"

    elif mriot_type == "WIOD16":
        return f"WIOT{mriot_year}_Nov16_ROW.xlsb"

    elif mriot_type == "OECD21":
        return f"ICIO2021_{mriot_year}.csv"

    else:
        raise ValueError("Unknown MRIOT type")


def download_mriot(mriot_type, mriot_year, download_dir):
    """Download EXIOBASE3, WIOD16 or OECD21 MRIOT for specific years

    Parameters
    ----------
    mriot_type : string
    mriot_year : int
    download_dir : pathlib.PosixPath
    """

    if mriot_type == "EXIOBASE3":
        # EXIOBASE3 gets a system argument. This can be ixi (ind x ind matrix)
        # or pxp (prod x prod matrix). By default both are downloaded, we here
        # use only ixi for the time being.
        pymrio.download_exiobase3(
            storage_folder=download_dir, system="ixi", years=[mriot_year]
        )

    elif mriot_type == "WIOD16":
        download_dir.mkdir(parents=True, exist_ok=True)
        downloaded_file_name = u_fh.download_file(
            WIOD_FILE_LINK,
            download_dir=download_dir,
        )
        downloaded_file_zip_path = Path(downloaded_file_name + ".zip")
        Path(downloaded_file_name).rename(downloaded_file_zip_path)

        with zipfile.ZipFile(downloaded_file_zip_path, "r") as zip_ref:
            zip_ref.extractall(download_dir)

    elif mriot_type == "OECD21":
        years_groups = ["1995-1999", "2000-2004", "2005-2009", "2010-2014", "2015-2018"]
        year_group = years_groups[int(np.floor((mriot_year - 1995) / 5))]

        pymrio.download_oecd(storage_folder=download_dir, years=year_group)


def parse_mriot(mriot_type, downloaded_file):
    """Parse EXIOBASE3, WIOD16 or OECD21 MRIOT for specific years

    Parameters
    ----------
    mriot_type : str
    downloaded_file : pathlib.PosixPath
    """

    if mriot_type == "EXIOBASE3":
        mriot = pymrio.parse_exiobase3(path=downloaded_file)
        # no need to store A
        mriot.A = None

    elif mriot_type == "WIOD16":
        mriot_df = pd.read_excel(downloaded_file, engine="pyxlsb")

        Z, Y, x = parse_mriot_from_df(
            mriot_df,
            col_iso3=2,
            col_sectors=1,
            row_fd_cats=2,
            rows_data=(5,2469),
            cols_data=(4,2468),
        )

        mriot = pymrio.IOSystem(Z=Z, Y=Y, x=x)
        multiindex_unit = pd.MultiIndex.from_product(
                [mriot.get_regions(), mriot.get_sectors()],
                names = ['region', 'sector']
                )
        mriot.unit = pd.DataFrame(
                    data = np.repeat(["M.EUR"], len(multiindex_unit)),
                    index = multiindex_unit,
                    columns = ["unit"]
                    )

    elif mriot_type == "OECD21":
        mriot = pymrio.parse_oecd(path=downloaded_file)
        mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)

    else:
        raise RuntimeError(f"Unknown mriot_type: {mriot_type}")
    return mriot

class SupplyChain:
    """SupplyChain class.

    The SupplyChain class provides methods for loading Multi-Regional Input-Output
    Tables (MRIOT) and computing direct, indirect and total impacts.

    Attributes
    ----------
    mriot : pymrio.IOSystem
            An object containing all MRIOT related info (see also pymrio package):
                mriot.Z : transaction matrix, or interindustry flows matrix
                mriot.Y : final demand
                mriot.x : industry or total output
                mriot.meta : metadata
    coeffs : pd.DataFrame
            Technical (if Leontief, A) or allocations (if Ghosh, B) coefficients matrix
    inverse : pd.DataFrame
            Leontief (L) or Ghosh (G) inverse matrix
    dir_prod_impt_mat : pd.DataFrame
            Direct production impact for each country, sector and event
    dir_prod_impt_eai : pd.DataFrame
            Expected direct production impact for each country and sector
    indir_prod_impt_mat : pd.DataFrame
            Indirect production impact for each country, sector and event
    indir_prod_impt_mat : pd.DataFrame
            Expected indirect production impact for each country and sector
    tot_prod_impt_mat : pd.DataFrame
            Total production impact for each country, sector and event
    tot_prod_impt_eai : pd.DataFrame
            Expected total production impact for each country and sector
    sim: boario simulation instance
    events_w_imp_id : ids of events leading to impacts
    events_w_imp_date : dates of events leading to impacts
    """

    def __init__(
        self,
        mriot=None,
        inverse=None,
        coeffs=None,
        dir_prod_impt_mat=None,
        dir_prod_impt_eai=None,
        indir_prod_impt_mat=None,
        indir_prod_impt_eai=None,
        total_imp_mat=None,
        total_eai=None,
    ):

        """Initialize SupplyChain."""
        self.mriot = pymrio.IOSystem() if mriot is None else mriot
        self.inverse = pd.DataFrame([]) if inverse is None else inverse
        self.coeffs = pd.DataFrame([]) if coeffs is None else coeffs
        self.dir_prod_impt_mat = (
            pd.DataFrame([]) if dir_prod_impt_mat is None else dir_prod_impt_mat
        )
        self.dir_prod_impt_eai = pd.DataFrame([]) if dir_prod_impt_eai is None else dir_prod_impt_eai
        self.indir_prod_impt_mat = (
            pd.DataFrame([]) if indir_prod_impt_mat is None else indir_prod_impt_mat
        )
        self.indir_prod_impt_eai = pd.DataFrame([]) if indir_prod_impt_eai is None else indir_prod_impt_eai
        self.total_imp_mat = (
            pd.DataFrame([]) if total_imp_mat is None else total_imp_mat
        )
        self.total_eai = pd.DataFrame([]) if total_eai is None else total_eai

    @classmethod
    def from_mriot(
        cls, mriot_type, mriot_year, mriot_dir=MRIOT_DIRECTORY, del_downloads=True
    ):
        """Download and read Multi-Regional Input-Output Tables using pymrio.

        Parameters
        ----------
        mriot_type : str
            Type of mriot table to read.
            The three possible types are: 'EXIOBASE3', 'WIOD16', 'OECD21'
        mriot_year : int
            Year of MRIOT
        mriot_dir : pathlib.PosixPath
            Path to the MRIOT folder
        del_downloads : bool
            If the downloaded files are deleted after saving the parsed data. Default is
            True. WIOD16 data and OECD21 data are downloaded as group of years

        Returns
        -------
        mriot : pymrio.IOSystem
            An object containing all MRIOT related info (see also pymrio package):
                mriot.Z : transaction matrix, or interindustry flows matrix
                mriot.Y : final demand
                mriot.x : total output
                mriot.meta : metadata
        """

        # download directory and file of interest
        downloads_dir = mriot_dir / mriot_type / "downloads"
        downloaded_file = downloads_dir / mriot_file_name(mriot_type, mriot_year)

        # parsed data directory
        parsed_data_dir = mriot_dir / mriot_type / str(mriot_year)

        # if data were not downloaded nor parsed: download, parse and save parsed
        if not downloaded_file.exists() and not parsed_data_dir.exists():
            download_mriot(mriot_type, mriot_year, downloads_dir)

            mriot = parse_mriot(mriot_type, downloaded_file)
            mriot.save(parsed_data_dir)

            if del_downloads:
                for dwn in downloads_dir.iterdir():
                    dwn.unlink()
                downloads_dir.rmdir()

        # if data were downloaded but not parsed: parse and save parsed
        elif downloaded_file.exists() and not parsed_data_dir.exists():
            mriot = parse_mriot(mriot_type, downloaded_file)
            mriot.save(parsed_data_dir)

        # if data were parsed and saved: load them
        else:
            mriot = pymrio.load(path=parsed_data_dir)

        # aggregate ROWs for EXIOBASE:
        if mriot_type == 'EXIOBASE3':
            agg_regions = mriot.get_regions().tolist()[:-5] + ['ROW']*5
            mriot = mriot.aggregate(region_agg = agg_regions)

        # Check if negative demand - this happens when the
        # "Changes in Inventory (CII)" demand category is
        # larger than the sum of all other categories
        if (mriot.Y.sum(axis=1) < 0).any():
            LOGGER.debug("Found negatives values in total final demand,"
                        "setting them to 0 and recomputing production vector")
            mriot.Y.loc[mriot.Y.sum(axis=1) < 0] = mriot.Y.loc[mriot.Y.sum(axis=1) < 0].clip(lower=0)
            mriot.x = pymrio.calc_x(mriot.Z, mriot.Y)

        mriot.meta.change_meta(
            "description", "Metadata for pymrio Multi Regional Input-Output Table"
        )
        mriot.meta.change_meta("name", f"{mriot_type}-{mriot_year}")

        return cls(mriot=mriot)

    def calc_secs_exp_imp_shock(self, exposure, impact, impacted_secs, shock_factor=None):
        """TODO: better docstring
        This function needs to return an object equivalent to self.direct_imp_mat
        starting from a standard CLIMADA impact calculation. Will call this object
        self.impacts_to_sectors. This object will also compute a sector exposure.

        shock_factor: np.array
            By default, shocks to production are equivalent to the ratio between stock impact 
            and exposure. For all sectors, shock_factor regulates whether this needs to 
            attenuate (factors<1) or amplify (factors>1). Default value is None which implies 
            factors for all sectors are = 1.
        """

        if impacted_secs is None:
            warnings.warn(
                "No impacted sectors were specified. It is assumed that the exposure is "
                "representative of all sectors in the IO table"
            )
            impacted_secs = self.mriot.get_sectors().tolist()

        elif isinstance(impacted_secs, (range, np.ndarray)):
            impacted_secs = self.mriot.get_sectors()[impacted_secs].tolist()

        self.secs_exp = pd.DataFrame(
            0, index=["total_value"], columns=self.mriot.Z.columns
        )
        self.secs_imp = pd.DataFrame(
            0, index=impact.event_id, columns=self.mriot.Z.columns
        )

        mriot_type = self.mriot.meta.name.split("-")[0]

        for exp_regid in exposure.gdf.region_id.unique():
            exp_bool = exposure.gdf.region_id == exp_regid
            tot_value_reg_id = exposure.gdf[exp_bool].value.sum()
            # consider using impact.impact_reg_agg when merged - anyway check for the presence of imp_mat
            tot_imp_reg_id = impact.imp_mat[:, np.where(exp_bool)[0]].sum(1)

            mriot_reg_name = self.map_exp_to_mriot(exp_regid, mriot_type)
            secs_prod = self.mriot.x.loc[(mriot_reg_name, impacted_secs), :]

            secs_prod_ratio = (secs_prod / secs_prod.sum()).values.flatten()

            # Overall sectorial stock exposure and impact are distributed among
            # subsectors proportionally to their their own contribution to overall
            # sectorial production: Sum needed below in case of many ROWs, which are
            # aggregated into one country as per WIOD table.
            self.secs_exp.loc[:, (mriot_reg_name, impacted_secs)] += (
                tot_value_reg_id * secs_prod_ratio
            )
            self.secs_imp.loc[:, (mriot_reg_name, impacted_secs)] += (
                tot_imp_reg_id * secs_prod_ratio
            )

        events_w_imp_bool = self.secs_imp.sum(1)!=0
        self.secs_imp = self.secs_imp[events_w_imp_bool]
        self.events_w_imp_id = impact.event_id[events_w_imp_bool]
        self.events_w_imp_date = impact.date[events_w_imp_bool]

        if shock_factor is None:
            shock_factor = np.repeat(1, self.mriot.x.shape[0])

        self.secs_shock = self.secs_imp.divide(
            self.secs_exp.values
        ).fillna(0) * shock_factor

        if not np.all(self.secs_shock <= 1):
            warnings.warn(
                "Consider changing the provided provided stock-to-production losses "
                "ratios, as some of them lead to production losses in some sectors to "
                "exceed the maximum sectorial production. For these sectors, total "
                "production loss is assumed."
            )
            self.secs_shock[self.secs_shock > 1] = 1

    def calc_direct_production_impacts(self):
        """Calculate direct production impacts."""

        self.dir_prod_impt_mat = (
            self.mriot.x.values.flatten() * self.secs_shock * self.conversion_factor()
        )

    # TODO: Consider saving results in a dict {io_approach: results} so one can run and
    # save various model without reloading the IOT
    def calc_indirect_production_impacts(self, io_approach):
        """Calculate indirect production impacts according to the specified input-output
        appraoch.

        Parameters
        ----------
        io_approach : str
            The adopted input-output modeling approach.
            Possible choices are 'leontief', 'ghosh' and 'eeioa'.

        References
        ----------
        [1] W. W. Leontief, Output, employment, consumption, and investment,
        The Quarterly Journal of Economics 58, 1944.
        [2] Ghosh, A., Input-Output Approach in an Allocation System,
        Economica, New Series, 25, no. 97: 58-64. doi:10.2307/2550694, 1958.
        [3] Kitzes, J., An Introduction to Environmentally-Extended Input-Output
        Analysis, Resources, 2, 489-503; doi:10.3390/resources2040489, 2013.
        """

        n_events = self.events_w_imp_id.shape[0]
        self.calc_matrices(io_approach=io_approach)

        if io_approach == "leontief":
            degr_demand = (
                self.secs_shock * self.mriot.Y.values.flatten() * self.conversion_factor()
            )

            self.indir_prod_impt_mat = pd.concat(
                [
                    pymrio.calc_x_from_L(self.inverse, degr_demand.iloc[i])
                    for i in range(n_events)
                ],
                axis=1,
            ).T.set_index(self.events_w_imp_id)

        elif io_approach == "ghosh":
            value_added = calc_v(self.mriot.Z, self.mriot.x)
            degr_value_added = (
                self.secs_shock * value_added.values * self.conversion_factor()
            )

            self.indir_prod_impt_mat = pd.concat(
                [
                    calc_x_from_G(self.inverse, degr_value_added.iloc[i])
                    for i in range(n_events)
                ],
                axis=1,
            ).T.set_index(self.events_w_imp_id)

        elif io_approach == "eeioa":
            self.indir_prod_impt_mat = (
                pd.DataFrame(
                    self.secs_shock.dot(self.inverse)
                    * self.mriot.x.values.flatten()
                )
                * self.conversion_factor()
            )

        elif io_approach in ['boario_aggregated', 'boario_separated']:

            self.mriot.A = self.coeffs
            self.mriot.L = self.inverse

            # call ARIOPsiModel with default params
            model = ARIOPsiModel(self.mriot,
                                 order_type = "alt",
                                 alpha_base = 1.0,
                                 alpha_max = 1.25,
                                 alpha_tau = 365,
                                 rebuild_tau = 60,
                                 main_inv_dur = 90,
                                 monetary_factor = self.conversion_factor(),
                                 temporal_units_by_step = 1,
                                 iotable_year_to_temporal_unit_factor = 365,
                                 infinite_inventories_sect = None,
                                 inventory_dict = None,
                                 productive_capital_vector = self.secs_exp / self.conversion_factor(),
                                 productive_capital_to_VA_dict = None,
                                 psi_param=0.80,
                                 inventory_restoration_tau=60
                                 )

            # run simulation up to one year after the last event
            self.sim = Simulation(
                        model,
                        register_stocks=False,
                        n_temporal_units_to_sim = self.events_w_imp_date[-1]-self.events_w_imp_date[0]+365,
                        separate_sims = False,
                             )

            events_list = [EventKapitalRecover.from_series(
                                    impact=self.secs_imp.iloc[i],
                                    recovery_time = 30,
                                    recovery_function="linear",
                                    households_impact=[],
                                    occurrence = (self.events_w_imp_date[i]-self.events_w_imp_date[0]+1),
                                    duration = 1,
                                    # TODO: it is assumed that exposure/impacts are expressed with full 
                                    # values, deal with cases where CLIMADA exposure is expressed in K, 
                                    # M or Bn
                                    event_monetary_factor = 1,
                                    ) for i in range(n_events)
                            ]

            if io_approach == 'boario_aggregated':
                self.sim.add_events(events_list)

                self.sim.loop()
                self.indir_prod_impt_mat = self.sim.production_realised.copy()[
                                                    self.secs_imp.columns
                                                    ]*self.conversion_factor()

            else: #'boario_separated'
                indir_prod_impt_df_list = []
                for ev in events_list:
                    self.sim.add_event(ev)
                    self.sim.loop()
                    indir_prod_impt_df_list.append(
                        self.sim.production_realised.copy()[
                            self.secs_imp.columns
                            ]*self.conversion_factor()
                        )
                    self.sim.reset_sim_full()

                self.indir_prod_impt_mat = indir_prod_impt_df_list

        else:
            raise RuntimeError(f"Unknown io_approach: {io_approach}")

    def calc_total_production_impacts(self):
        """Calculate total production impacts."""
        self.tot_prod_impt_mat = self.dir_prod_impt_mat.add(self.indir_prod_impt_mat)

    def calc_production_impacts(
        self,
        impact,
        exposure,
        impacted_secs=None,
        io_approach=None,
        stock_to_prod_shock=None,
    ):
        """Calculate direct, indirect and total production impacts.

        Parameters
        ----------
        impact : Impact
            Impact object with stocks impacts.
        exposure : Exposures
            Exposures object for impact calculation.
        impacted_secs : range, np.ndarray or list, optional
            The directly affected sectors. If range or np.ndarray,
            it contains the affected sectors' positions in the MRIOT.
            If list, it contains the affected sectors' names in the MRIOT.
        """

        self.calc_secs_exp_imp_shock(exposure, impact, impacted_secs)

        self.calc_direct_production_impacts(stock_to_prod_shock)
        self.calc_indirect_production_impacts(io_approach)
        self.calc_total_production_impacts()

    def calc_production_eai(self, frequencies):

        if not self.dir_prod_impt_mat is None:
            self.dir_prod_impt_eai = self.dir_prod_impt_mat.T.dot(frequencies)

        if io_approach not in ["boario_aggregated", "boario_separated"]:
        #TODO: assess how to calc ind and tot eai for boario cases
            if not self.indir_prod_impt_mat is None:
                self.indir_prod_impt_eai = self.indir_prod_impt_mat.T.dot(frequencies)

            if not self.tot_prod_impt_mat is None:
                self.tot_prod_impt_eai = self.tot_prod_impt_mat.T.dot(frequencies)

    def calc_matrices(self, io_approach):
        """
        Build technical coefficient and Leontief inverse matrixes (if Leontief approach) or
        allocation coefficients and Ghosh matrixes (if Ghosh approach).
        """

        io_model = {
            "leontief": (pymrio.calc_A, pymrio.calc_L),
            "eeioa": (pymrio.calc_A, pymrio.calc_L),
            "ghosh": (calc_B, calc_G),
            "boario_aggregated": (pymrio.calc_A, pymrio.calc_L),
            "boario_separated": (pymrio.calc_A, pymrio.calc_L)
        }

        coeff_func, inv_func = io_model[io_approach]

        self.coeffs = coeff_func(self.mriot.Z, self.mriot.x)
        self.inverse = inv_func(self.coeffs)

    def conversion_factor(self):
        """
        Convert values in MRIOT.
        """

        unit = None
        if isinstance(self.mriot.unit, pd.DataFrame):
            unit = self.mriot.unit.values[0][0]
        elif isinstance(self.mriot.unit, str):
            unit = self.mriot.unit
        if unit in ["M.EUR", "Million USD"]:
            conversion_factor = 1e6
        else:
            conversion_factor = 1
            warnings.warn(
                "No known unit was provided. It is assumed that values do not need to "
                "be converted."
            )
        return conversion_factor

    def map_exp_to_mriot(self, exp_regid, mriot_type):
        """
        Map regions names in exposure into Input-output regions names.
        exp_regid must be according to ISO 3166 numeric country codes.
        """

        if mriot_type == "EXIOBASE3":
            mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha2")

        elif mriot_type in ["WIOD16", "OECD21"]:
            mriot_reg_name = u_coord.country_to_iso(exp_regid, "alpha3")

        else:
            warnings.warn(
                "For a correct calculation the format of regions' names in exposure and "
                "the IO table must match."
            )
            return exp_regid

        idx_country = np.where(self.mriot.get_regions() == mriot_reg_name)[0]

        if (not idx_country.size > 0.0) and (mriot_type == "OECD21"):
            raise ValueError(
                f"OECD21 does not contain info on {mriot_reg_name} neither has a ROW region"
                )
        elif not idx_country.size > 0.0:
            mriot_reg_name = "ROW"

        return mriot_reg_name
