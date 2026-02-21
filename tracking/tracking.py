from typing import Dict
import pandas as pd
from .dataframe_converter import convert_to_dataframe

class ProvenanceTracker:
    def __init__(self, save_on_neo4j=False):
        self.save_on_neo4j = save_on_neo4j
        self.tracking_enabled = False
        self.changes: dict[int, dict[str, pd.DataFrame]] = {}
        self.operation_counter = 0

    def subscribe(self, df):
        self.df_before = df.copy()
        self.tracking_enabled = True
        return df

    '''
    function that analyze changes before and after the operation
    the result will be a dictionary of dictionaries {operation_number:{"before":df_input, "after":df_output}}
    where operation number is the number of the operation/activity in chronological order of execution
    '''
    def analyze_changes(self, df_after):
        if not self.tracking_enabled:
            return
        self.df_after = df_after.copy()
        self.changes[self.operation_counter] = {
            "before": self.df_before,
            "after": self.df_after
        }
        self.df_before = self.df_after.copy()
        self.operation_counter += 1

    def get_changes(self):
        return self.changes