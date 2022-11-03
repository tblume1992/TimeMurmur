# -*- coding: utf-8 -*-
from TimeMurmur.utils.FeatureExtraction import get_features


class ExtractFeatures:

    def __init__(self,
                 run_dict):
        self.run_dict = run_dict
        self.id_column = self.run_dict['global']['ID Column']
        self.date_column = self.run_dict['global']['Date Column']
        self.target_column = self.run_dict['global']['Target Column']
        self.seasonal_period = self.run_dict['global']['main_seasonal_period']

    def build_axis(self, dataset):
        dataset = dataset[['Murmur ID',
                           self.date_column,
                           self.target_column]]
        extracted_features = get_features(dataset,
                                          'Murmur ID',
                                          self.target_column,
                                          self.seasonal_period)
        self.run_dict['global']['ts_features'] = extracted_features
