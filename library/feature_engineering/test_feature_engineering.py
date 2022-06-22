import unittest
from library.feature_engineering.feature_engineering import FeatureEngineer


class MyTestCase(unittest.TestCase):
    def test_feature_generation(self):
        feature_engineer: FeatureEngineer = FeatureEngineer(radius=30, file_path="../../data/9_3_2_BEMS342281.csv")
        feature_engineer.create_features()

        self.assertEqual(len(feature_engineer.feature_engineered_data), 1)
        self.assertGreaterEqual(len(feature_engineer.feature_engineered_data[0].columns), 50)
        self.assertIs("# of Immune Cells", feature_engineer.feature_engineered_data[0].columns)


if __name__ == '__main__':
    unittest.main()
