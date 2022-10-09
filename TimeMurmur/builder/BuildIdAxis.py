# -*- coding: utf-8 -*-

import pandas as pd


class IdAxis:
    def __init__(self,
                 id_column,
                 feature_columns,
                 ):
        self.id_column = id_column
        if isinstance(feature_columns, str):
            feature_columns = [feature_columns]
        if feature_columns is None:
            feature_columns = []
        self.feature_columns = feature_columns

    def build_axis(self, dataset, id_exogenous=None):
        if not self.feature_columns and id_exogenous is None:
            return None
        dataset = dataset[[self.id_column] + self.feature_columns].drop_duplicates()
        if id_exogenous is not None:
            try:
                ids = id_exogenous[self.id_column].drop_duplicates()
            except:
                raise ValueError('Id Exogenous dataset MUST have Id column that matches the Id column in full dataset')
            assert len(ids) == len(id_exogenous), 'Id Exogenous dataset contains duplicate Ids!'
            dataset = dataset.merge(id_exogenous, 
                                    on=self.id_column,
                                    how='left')
        return dataset
        


