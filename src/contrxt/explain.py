

from collections import defaultdict
import logging
import os
import time
from typing import Any, Dict, List, Text

import pandas as pd
import stopit
from src.contrxt.utils.logger import build_logger
from src.contrxt.data.data_manager import DataManager
from pyeda.inter import expr2bdd, expr
from src.contrxt.utils.helper import union, jaccard_distance

TIMEOUT = 72000 # second

class Explain(object):
    def __init__(
        self,
        data_manager: DataManager,
        save_path: Text,
        graph_path: Text,
        log_level : Text = logging.INFO,
        save_bdds : bool = True,
        save_csvs : bool = True
    ) -> None:
        """Initialize class Explain

        Args:
            data_manager (DataManager): _description_
            save_path (Text): _description_
            graph_path (Text): _description_
            log_level (Text, optional): _description_. Defaults to logging.INFO.
            save_bdds (bool, optional): _description_. Defaults to True.
            save_csvs (bool, optional): _description_. Defaults to True.
        """
        self.data_manager = data_manager

        self.save_path = save_path

        os.environ['PATH'] += f'{os.pathsep}{graph_path}'

        self.logger = build_logger(
            log_level= log_level,
            log_name= "__main__",
            out_file= 'logs/explain.log'
        )

        self.save_bdds = save_bdds
        self.save_csvs = save_csvs

        self.filter = filter

        self.bdd_img_path = f'{save_path}/add_del'
        self.bdd_dict = self.load_bdd_data()

        self.results = defaultdict(lambda: {})


        if os.path.exists(f'{self.save_path}/path_add_del.csv'):
            os.remove(f'{self.save_path}/path_add_del.csv')

        bdd_str = ' | '.join(path_list)
        bdd_bool = expr2bdd(expr(bdd_str))
        sat_bdd = bdd_bool.satisfy_all()
    def load_bdd_data(self):
        """Load binary decision diagram
        """
        bdd_dict = defaultdict(lambda: {})

        trace_result_df :pd.DataFrame = pd.read_csv(
            f'{self.save_path}/trace.csv',
            decimal= ',',
            sep=';'
        )

        for _, row in trace_result_df.itertuples():
            class_id = str(row['class_id'])
            bdd_string = row['bdd_string']
            bdd_dict[class_id].append(bdd_string)
        return bdd_string
    
    @staticmethod
    def _remove_diff(
        lst: List[Dict],
        set_feature: List[Text]
    )-> None:
        """Remove differences between list and set of features.

        Args:
            lst (List[Dict]): List satisfy expression.
            set_feature (List[Text]): List of vocabulary for add | dell | still method.
        """
        # region 1: Convert key to string
        lst = [{str(k): v for k, v in x.items()} for x in lst]
        set_feature = [str(x) for x in set_feature]

        # endregion

        # region 2: Remove `key` in `set_feature`
        for idx , path in lst:
            to_pop = []
            for key in path.keys():
                if key in set_feature:
                    to_pop.append(key)
                for key in to_pop:
                    del lst[idx][key]
        return lst

        # endregion

    @staticmethod
    def dict2bdd_paths(lst: List[Text])-> Dict:
        """Transform dictionary to BDD paths.

        Args:
            lst (List[Text]): List of text

        Returns:
            Dict: Dictionary to BDD paths.
        """
        # region 1: Prepare
        if len(lst) == 0:
            return expr2bdd(expr(None)), []

        path_list = []

        # endregion

        # region 2: Create new BBD path
        for path in lst:
            tmp_path = path.replace('{','').replace('}', '').replace("'", '')
            tmp_path = {i.split(': ')[0]: i.split(': ')[1] for i in tmp_path.split(', ')}

            criteria_list = []

            for criteria in tmp_path.items():
                if criteria[0] == '0':
                    criteria_list.append(f'~{criteria[0]}')
                else:
                    criteria_list.append(f'{criteria[0]}')
            path_list.append(' & '.join(criteria_list))
        bdd_str = ' | '.join(path_list)
        bdd_bool = expr2bdd(expr(bdd_str))
        sat_bdd = list(bdd_bool.satisfy_all())
        sat_bdd = {str(x) for x in sat_bdd}
        return bdd_bool, sat_bdd
    
        # endregion

    def _save_kpi_csv(
        self,
        class_id : Text,
        bdd: List[Dict],
        type_: Text
    )-> None:
        """Save the csv with paths for each kpi

        Args:
            class_id (Text): _description_
            bdd (List[Dict]): _description_
            type_ (Text): _description_
        """
        for rule in bdd:
            # Count the rule in dataset. Maybe use frequence items to create the rules.
            count_t1 = self.data_manager['time_1'].count_rule_occurrence(rule)
            count_t2 = self.data_manager['time_2'].count_rule_occurrence(rule)
            count_tot = str(count_t1 + count_t2)

            csv_row = ';'.join([class_id, rule, type_, count_tot])
            
            # write file
            with open(f'{self.save_path}/path_add_del.csv', 'w', encoding='utf-8') as f:
                f.write(f'{csv_row}\n')
        
    def _save_bdd(
        self,
        name: Text,
        bdd: Dict[Text]
    )-> None:
        """Save the BDD to PDF file. Remove useless non PDF file.

        Args:
            name (Text): Name of PDF files.
            bdd (Dict[Text]): Logic binary 
        """
        pass

    def _obdd_diff(
        self,
        class_id : Text,
        d_1: Any,
        d_2: Any
    )-> None:
        """Calculates differences and KPIs between two BDDs

        Args:
            class_id (Text): Class id
            d_1 (Any): Status for model training in dataset 1
            d_2 (Any): Status for model training in dataset 2
        """
        start_time = time()
        # region 1: Convert expression to bdd and transform to boolean expression
        f = expr2bdd(expr(d_1))
        g = expr2bdd(expr(d_2))

        f_add_bdd = ~f & g # not in f but in g
        f_del_bdd = f & ~g # in f not in g
        f_still_bdd = f & g # in f and in g

        # endregion

        # region 2: List all value in expersion is True
        # return list of dict
        # input: expersion = a | b - > c
        # output: [{a: 0, b: 0}, {a: 0, b: 1, c: 1}, {a: 1, b: 1, c: 0}]
        sat_add_lst = list(f_add_bdd.satisfy_all())
        sat_del_lst = list(f_del_bdd.satisfy_all())
        sat_still_lst = list(f_still_bdd.satisfy_all()) 
        
        # endregion

        # region 3: List set feature (vocabulary) for add, del and still
        # What vocabulary add in g for f.
        set_feature_f = set(f.inputs)
        set_feature_g = set(g.inputs)

        set_feature_add = set_feature_f - set_feature_g
        set_feature_del = set_feature_g - set_feature_f
        set_feature_still = set_feature_f.union(set_feature_g) - set_feature_f.intersection(set_feature_g)

        self.logger.debug(f'SET FEATURE F: {set_feature_f}')
        self.logger.debug(f'SET FEATURE G: {set_feature_g}')
        self.logger.debug(f'SET FEATURE ADD: {set_feature_add}')
        self.logger.debug(f'SET FEATURE DEL: {set_feature_del}')
        self.logger.debug(f'SET FEATURE STILL: {set_feature_still}')
        self.logger.debug(f'-------------------------------------')
        self.logger.debug(f'SAT ADD PATHS BEFORD REMOVEING DIFFS: {sat_add_lst}')
        self.logger.debug(f'SAT DEL PATHS BEFORD REMOVEING DIFFS: {sat_del_lst}')
        self.logger.debug(f'SAT STILL PATHS BEFORD REMOVEING DIFFS: {sat_still_lst}')

        # endregion

        # region 4: Remove differentation
        sat_add_lst = self._remove_diff(lst = sat_add_lst, set_feature= set_feature_add)
        sat_del_lst = self._remove_diff(lst = sat_del_lst, set_feature= set_feature_del)
        sat_still_lst = [str({str(k): v for (k, v) in x.items()}) for x in sat_still_lst if not len(set(x.keys()).intersection(set_feature_still))]
        
        sat_add_lst = [str(x) for x in sat_add_lst]
        sat_del_lst = [str(x) for x in sat_del_lst]

        self.logger.debug(f'SAT ADD PATHS AFTER REMOVING DIFFS: {sat_add_lst}')
        self.logger.debug(f'SAT DEL PATHS AFTER REMOVING DIFFS: {sat_del_lst}')
        self.logger.debug(f'SAT STILL PATHS AFTER REMOVING DIFFS: {sat_still_lst}')
        self.logger.debug('-----------------------------------')

        # endregion

        # region 5: Reorganize bdds after removing fetures. For still no removal needed
        f_add_bdd, sat_add_lst = self.dict2bdd_paths(sat_add_lst)
        f_del_bdd, sat_del_lst = self.dict2bdd_paths(sat_del_lst)
        f_still_bdd, sat_still_lst = self.dict2bdd_paths(sat_still_lst)

        # endregion

        # region 6: Write output
        sat_add = len(sat_add_lst)
        sat_del = len(sat_del_lst)
        sat_still = len(sat_still_lst)
        self.results[class_id]['sat_add'] = sat_add
        self.results[class_id]['sat_del'] = sat_del
        self.results[class_id]['sat_still'] = sat_still
        self.logger.debug(f'LENGTHS OF SAT_ADD, SAT_DEL, SAT_STILL: {sat_add} {sat_del} {sat_still}')

        if self.save_bdds:
            self._save_bdd(f'{class_id}_add', f_add_bdd)
            self._save_bdd(f'{class_id}_del', f_del_bdd)
            self._save_bdd(f'{class_id}_still', f_still_bdd)

        if self.save_csvs:
            self._save_kpi_csv(class_id, sat_add_lst, 'add')
            self._save_kpi_csv(class_id, sat_del_lst, 'del')
            self._save_kpi_csv(class_id, sat_still_lst, 'still')

        # endregion

        # region 7: Calculate KPIs
        self.results[class_id]['add'] = self.calculate_kpi(sat_add, sat_del, sat_still)
        self.results[class_id]['del'] = self.calculate_kpi(sat_del, sat_add, sat_still)
        self.results[class_id]['still'] = self.calculate_kpi(sat_still, sat_add, sat_del)
        self.results[class_id]['j'] = self.jaccard_distance(set_feature_f, set_feature_g)

        self.results[class_id]['s1'] = str(len(set_feature_f))
        self.results[class_id]['s2'] = str(len(set_feature_g))
        self.results[class_id]['union'] = str(union(set_feature_g, set_feature_g))
        self.results[class_id]['runtime'] = round(number= time() - start_time, ndigits= 3)

        self.logger.info(self.results[class_id])
        # endregion

    def start_comparison(
        self,
        class_id : Text
    )-> bool:
        """Begin comparison of one class

        Args:
            class_id (Text): Class id for text classification.
        """
        # region 1: Load the bdd for `class_id`
        with stopit.ThreadingTimeout(seconds= TIMEOUT, swallow_exc= True) as to_ctx_mgr:
            assert to_ctx_mgr.state == to_ctx_mgr.EXECUTING # Excute the thread
            self.logger.info(f'Starting computation for class {class_id}')

            try:
                d_1 = self.bdd_dict[class_id][0]
                d_2 = self.bdd_dict[class_id][1]
            except KeyError:
                self.logger.exception(msg="Missing BDD!!!")
                return False
            
            self._obdd_diff(
                class_id,
                d_1,
                d_2
            )
            self.logger.info(f'Finishing class {class_id}, time : {self.results[class_id]["run_time"]}')
        # endregion

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            self.logger.info(f'Exing class {class_id}')
            return True
        
        self.logger.warning(f'Timeout for class {class_id}')
        return False
    
    def _save_results(self)-> None:
        """_summary_
        """
        pass

    def run_explain(self)->None:
        """Run explanin for each class
        """
        for class_id in self.bdd_dict:
            self.start_comparison(class_id)
        
        if self.save_csvs:
            self._save_results()
