# -*- coding: utf-8 -*-

def handle_hierarchy(self, df, id_column, hierarchy):
    hier_df = df[set([id_column] + hierarchy)].drop_duplicates()
    le = LabelEncoder()
    hier_columns = []
    for column in hierarchy:
        column_name = f'hierarchy_{column}'
        hier_df[column_name] = le.fit_transform(hier_df[column])
        hier_columns.append(column_name)
    return hier_df[[id_column] + hier_columns]