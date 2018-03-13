"""
Test Exposures from MATLAB file.
"""

import unittest
import numpy as np

import climada.util.hdf5_handler as hdf5
from climada.entity.exposures import source as source
from climada.entity.exposures.base import Exposures
from climada.util.constants import ENT_DEMO_MAT
from climada.util.config import CONFIG

class TestReader(unittest.TestCase):
    """Test reader functionality of the ExposuresMat class"""

    def test_read_demo_pass(self):
        """ Read one single excel file"""
        # Read demo excel file
        expo = Exposures()
        description = 'One single file.'
        expo.read(ENT_DEMO_MAT, description)

        # Check results
        n_expos = 50

        self.assertEqual(type(expo.id[0]), np.int64)
        self.assertEqual(expo.id.shape, (n_expos,))
        self.assertEqual(expo.id[0], 0)
        self.assertEqual(expo.id[n_expos-1], n_expos-1)

        self.assertEqual(expo.value.shape, (n_expos,))
        self.assertEqual(expo.value[0], 13927504367.680632)
        self.assertEqual(expo.value[n_expos-1], 12624818493.687229)

        self.assertEqual(expo.deductible.shape, (n_expos,))
        self.assertEqual(expo.deductible[0], 0)
        self.assertEqual(expo.deductible[n_expos-1], 0)

        self.assertEqual(expo.cover.shape, (n_expos,))
        self.assertEqual(expo.cover[0], 13927504367.680632)
        self.assertEqual(expo.cover[n_expos-1], 12624818493.687229)

        self.assertEqual(type(expo.impact_id[0]), np.int64)
        self.assertEqual(expo.impact_id.shape, (n_expos,))
        self.assertEqual(expo.impact_id[0], 1)
        self.assertEqual(expo.impact_id[n_expos-1], 1)

        self.assertEqual(type(expo.category_id[0]), np.int64)
        self.assertEqual(expo.category_id.shape, (n_expos,))
        self.assertEqual(expo.category_id[0], 1)
        self.assertEqual(expo.category_id[n_expos-1], 1)

        self.assertEqual(type(expo.assigned['NA'][0]), np.int64)
        self.assertEqual(expo.assigned['NA'].shape, (n_expos,))
        self.assertEqual(expo.assigned['NA'][0], 47)
        self.assertEqual(expo.assigned['NA'][n_expos-1], 46)

        self.assertEqual(expo.region_id.shape, (0,))

        self.assertEqual(expo.coord.shape, (n_expos, 2))
        self.assertEqual(expo.coord[0][0], 26.93389900000)
        self.assertEqual(expo.coord[n_expos-1][0], 26.34795700000)
        self.assertEqual(expo.coord[0][1], -80.12879900000)
        self.assertEqual(expo.coord[n_expos-1][1], -80.15885500000)

        self.assertEqual(expo.ref_year, CONFIG["present_ref_year"])
        self.assertEqual(expo.value_unit, 'USD')
        self.assertEqual(expo.tag.file_name, ENT_DEMO_MAT)
        self.assertEqual(expo.tag.description, description)

    def test_check_demo_warning(self):
        """Check warning centroids when demo read."""
        with self.assertLogs('climada.util.checker', level='WARNING') as cm:
            Exposures(ENT_DEMO_MAT)
        self.assertIn("Exposures.deductible not set. Default values set.", cm.output[0])
        self.assertIn("Exposures.cover not set. Default values set.", cm.output[1])
        self.assertIn("Exposures.region_id not set.", cm.output[2])

class TestObligatories(unittest.TestCase):
    """Test reading exposures obligatory values."""

    def tearDown(self):
        source.DEF_VAR_MAT = {'sup_field_name': 'entity',
                              'field_name': 'assets',
                              'var_name': {'lat' : 'lat',
                                           'lon' : 'lon',
                                           'val' : 'Value',
                                           'ded' : 'Deductible',
                                           'cov' : 'Cover',
                                           'imp' : 'DamageFunID',
                                           'cat' : 'Category_ID',
                                           'reg' : 'Region_ID',
                                           'uni' : 'Value_unit',
                                           'ass' : 'centroid_index',
                                           'ref' : 'reference_year'
                                          }
                             }

    def test_no_value_fail(self):
        """Error if no values."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['val'] = 'no valid value'
        expo = Exposures()
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT, var_names=new_var_names)

    def test_no_impact_fail(self):
        """Error if no impact ids."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['imp'] = 'no valid value'
        expo = Exposures()
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT, var_names=new_var_names)

    def test_no_coord_fail(self):
        """Error if no coordinates."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['lat'] = 'no valid Latitude'
        expo = Exposures()
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        new_var_names['var_name']['lat'] = 'nLatitude'
        new_var_names['var_name']['lon'] = 'no valid Longitude'
        with self.assertRaises(KeyError):
            expo.read(ENT_DEMO_MAT, var_names=new_var_names)

class TestOptionals(unittest.TestCase):
    """Test reading exposures optional values."""

    def tearDown(self):
        source.DEF_VAR_MAT = {'sup_field_name': 'entity',
                               'field_name': 'assets',
                               'var_name': {'lat' : 'lat',
                                            'lon' : 'lon',
                                            'val' : 'Value',
                                            'ded' : 'Deductible',
                                            'cov' : 'Cover',
                                            'imp' : 'DamageFunID',
                                            'cat' : 'Category_ID',
                                            'reg' : 'Region_ID',
                                            'uni' : 'Value_unit',
                                            'ass' : 'centroid_index',
                                            'ref' : 'reference_year'
                                           }
                              }

    def test_no_category_pass(self):
        """Not error if no category id."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['cat'] = 'no valid category'
        expo = Exposures()
        expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        # Check results
        self.assertEqual(0, expo.category_id.size)

    def test_no_region_pass(self):
        """Not error if no region id."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['reg'] = 'no valid region'
        expo = Exposures()
        expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        # Check results
        self.assertEqual(0, expo.region_id.size)

    def test_no_unit_pass(self):
        """Not error if no value unit."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['uni'] = 'no valid unit'
        expo = Exposures()
        expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        # Check results
        self.assertEqual('NA', expo.value_unit)

    def test_no_assigned_pass(self):
        """Not error if no value unit."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['ass'] = 'no valid assign'
        expo = Exposures()
        expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        # Check results
        self.assertEqual(0, len(expo.assigned))
        self.assertTrue(isinstance(expo.assigned, dict))

    def test_no_refyear_pass(self):
        """Not error if no value unit."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['ref'] = 'no valid ref'
        expo = Exposures()
        expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        # Check results
        self.assertEqual(CONFIG["present_ref_year"], expo.ref_year)

class TestDefaults(unittest.TestCase):
    """Test reading exposures default values."""

    def tearDown(self):
        source.DEF_VAR_MAT = {'sup_field_name': 'entity',
                            'field_name': 'assets',
                            'var_name': {'lat' : 'lat',
                                         'lon' : 'lon',
                                         'val' : 'Value',
                                         'ded' : 'Deductible',
                                         'cov' : 'Cover',
                                         'imp' : 'DamageFunID',
                                         'cat' : 'Category_ID',
                                         'reg' : 'Region_ID',
                                         'uni' : 'Value_unit',
                                         'ass' : 'centroid_index',
                                         'ref' : 'reference_year'
                                        }
                            }

    def test_no_cover_pass(self):
        """Check default values for excel file with no cover."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['cov'] = 'Dummy'
        expo = Exposures()
        expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        # Check results
        self.assertTrue(np.array_equal(expo.value, expo.cover))

    def test_no_deductible_pass(self):
        """Check default values for excel file with no deductible."""
        new_var_names = source.DEF_VAR_MAT
        new_var_names['var_name']['ded'] = 'Dummy'
        expo = Exposures()
        expo.read(ENT_DEMO_MAT, var_names=new_var_names)

        # Check results
        self.assertTrue(np.array_equal(np.zeros(len(expo.value)), \
                                              expo.deductible))

class TestParsers(unittest.TestCase):
    """Test parser auxiliary functions"""

    def setUp(self):
        self.expo = hdf5.read(ENT_DEMO_MAT)
        self.expo = self.expo['entity']
        self.expo = self.expo['assets']

    def test_parse_optional_exist_pass(self):
        """Check variable read if present."""
        var_ini = 0
        var = source._parse_mat_optional(self.expo, var_ini, 'lat')
        self.assertEqual(50, len(var))

    def test_parse_optional_not_exist_pass(self):
        """Check pass if variable not present and initial value kept."""
        var_ini = 0
        var = source._parse_mat_optional(self.expo, var_ini, 'Not Present')
        self.assertEqual(var_ini, var)

    def test_parse_default_exist_pass(self):
        """Check variable read if present."""
        def_val = 5
        var = source._parse_mat_default(self.expo, 'lat', def_val)
        self.assertEqual(50, len(var))

    def test_parse_default_not_exist_pass(self):
        """Check pass if variable not present and default value is set."""
        def_val = 5
        var = source._parse_mat_default(self.expo, 'Not Present', def_val)
        self.assertEqual(def_val, var)

# Execute Tests
TESTS = unittest.TestLoader().loadTestsFromTestCase(TestReader)
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOptionals))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestObligatories))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDefaults))
TESTS.addTests(unittest.TestLoader().loadTestsFromTestCase(TestParsers))
unittest.TextTestRunner(verbosity=2).run(TESTS)
