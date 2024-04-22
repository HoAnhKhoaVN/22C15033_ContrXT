

import ast
import re
from typing import Dict, List, Text

import pandas as pd


class BDD2Text(object):
    def __init__(
        self,
        file_path: Text,
        cls,
        wrapp_size: int = 100,
    ) -> None:
        self.threshold = 0.8
        df = self.prepare_data(file_path)
        self.label = cls
        ########################################
        ################# ADD PATHS ############
        ########################################

        add_dict :pd.DataFrame = df[(df['class']==cls) & (df['mode'] == 'add')]
        N_add = add_dict['N'].tolist()[0]

        self.path_dict_add : List = add_dict['path'].tolist()[0]
        self.N_add : List[bool] = self.rule_trimmer(N_add, self.threshold)
        self.used_paths_add = [x for i, x in enumerate(self.path_dict_add) if self.N_add[i]]

        ########################################
        ################# DEL PATHS ############
        ########################################
        del_dict :pd.DataFrame = df[(df['class']==cls) & (df['mode'] == 'del')]
        N_del = del_dict['N'].tolist()[0]

        self.path_dict_del : List = del_dict['path'].tolist()[0]
        self.N_del : List[bool] = self.rule_trimmer(N_del, self.threshold)
        self.used_paths_del = [x for i, x in enumerate(self.path_dict_del) if self.N_del[i]]

        #########################################
        ################# STILL PATHS ###########
        #########################################
        still_dict :pd.DataFrame = df[(df['class']==cls) & (df['mode'] == 'still')]
        N_still : List = still_dict['N'].tolist()[0]

        self.path_dict_still : List = still_dict['path'].tolist()[0]
        self.N_still : List[bool] = self.rule_trimmer(N_still, self.threshold)
        self.used_paths_still = [x for i, x in enumerate(self.path_dict_still) if self.N_del[i]]


    def prepare_data(
        self,
        file_path: Text,
    )-> pd.DataFrame:
        # region 1: Read CSV file
        temp_df = pd.read_csv(
            filepath_or_buffer= file_path,
            sep= ';',
            header= None
        )

        temp_df.columns = ['class', 'path', 'bdd', 'N']
        temp_df = temp_df[['class', 'path', 'bdd', 'N']]
        temp_df['class'] = temp_df['class'].astype('str')

        # endregion

        # region 2: Format data

        # region 2.1: String to dictionary
        def string_to_dict(s: Text)-> Text:
            """Process the string to add in the quotes before trying to evaluate
        
            Args:
                s (Text): Text to processs

            Returns:
                Text: Text after processing

            Reference:
                https:/stackoverflow.com/a/58561819/5842939

            """
            # region 2.1.1: Add quotes to dict keys
            # Remove : in text
            s = re.sub(
                pattern=r'(\w+):',
                repl=r'"\1"' # \1 is backreference in regrex to define the first capturing group.
            )

            # endregion

            # region 2.1.2: Add quotes to lists
            def add_quotes_to_lists(
                match: re.Match
            )-> Text:
                """_summary_

                Args:
                    match (re.Match): _description_

                Returns:
                    Text: _description_
                """
                return re.sub(
                    pattern=r'(\s\[])([^\],]+)',# pattern for list
                    repl=r'\1"\2"',
                    string=match.group(0)
                )

            s = re.sub(
                pattern=r'\[[^\]]+',
                repl= add_quotes_to_lists,
                string= s
            )
            # endregion

            # region 2.1.3: Evaluate the dictionary
            final = ast.literal_eval(node_or_string=s)
            return final
            # endregion
        
        temp_df['path'] = temp_df['path'].apply(string_to_dict)
        # endregion

        # region 2.2: Delete ampamp
        def remove_ampamp(
            dict : Dict
        )-> Dict:
            """_summary_

            Args:
                dict (Dict): _description_

            Returns:
                Dict: _description_
            """
            try:
                del dict['ampamp']
            except KeyError:
                pass
            return dict
        
        temp_df['path'] = temp_df['path'].apply(func= remove_ampamp)
        # endregion
        

        # endregion

        # region 3: Putting together paths for each class
        res = []
        for cls in temp_df['class'].unique():
            for mode in ['add', 'del', 'still']:
                tempak :pd.DataFrame = temp_df[(temp_df['class']== cls) & (temp_df['bdd']==mode)]
                paths = tempak['path'].tolist()
                Ns = tempak['N'].tolist()
                res.append([
                    cls,
                    mode,
                    paths,
                    Ns
                ])

        df = pd.DataFrame(
            data = res,
            columns=['class', 'mode', 'path', 'N'],
        )

        # endregion
        return df

    def rule_trimmer(
        self,
        data: List,
        threshold = 0.8,
    )->List[bool]:
        """

        Args:
            data (List): cummulative frequency for add, del, still rules.
            threshold (float, optional): The threshold for defining the frequency itemset in Apriori algorithm. Defaults to 0.8.

        Returns:
            List[bool]: Return list of T/F.
        """

        summ = sum(data)
        frac = [0 if summ ==0 else x/summ for x in data]

        data2 = list(enumerate(frac))
        data2.sort(key = lambda x: x[1], reverse= True)

        curr = 0
        res = []

        for i in data2:
            if curr <= threshold:
                pp = True
                curr+= i[1]
            else:
                pp = False
            
            res.append((i[0],  pp))

        res.sort(key=lambda x: x[0])

        return [x[1] for x in res]
    
    def text_formatter(
        self,
        text : Text,
        bc : int = None,
        tc : int = None,
        bold : bool = False,
        underline : bool = False,
        _reversed : bool = False,
    )-> Text:
        """Add requested style to the given string.

        Add ANSI Escape codes to add text color, background color and other
        decorations like bold, underline and reversed.

        Args:
            text (Text): target string
            bc (int, optional): Background color code. Defaults to None.
            tc (int, optional): Text color code. Defaults to None.
            bold (bool, optional): if True makes the given string bold. Defaults to False.
            underline (bool, optional): if True makes the given string undelined. Defaults to False.
            _reversed (bool, optional): if True reverses the background and text colors. Defaults to False.

        Returns:
            Text: target text after formatting.
        """
        # region 1: Check the type of variables.
        assert isinstance(text, str), f'Text should be strong not {type(text)}'
        assert isinstance(bc, (int, type(None))), f'Background color code should be integer not {type(bc)}'
        assert isinstance(tc, (int, type(None))), f'Text color code should be integer not {type(tc)}'
        assert isinstance(bold, bool), f'Bold should be integer not {type(bold)}'
        assert isinstance(bold, bool), f'Bold should be integer not {type(bold)}'
                assert isinstance(bold, bool), f'Bold should be integer not {type(bold)}'
        # endregion


    def simple_text(
        self,
        fn: Text
    )-> None:
        """Write rule in natural 

        Args:
            fn (Text): _description_
        """
        with open(fn, 'w', encoding='utf-8') as f:
            # region 1: Creating class name
            f.write('='*70)
            f.write(self.label)
            f.write('='*70)
            f.write("\n")

            # endregion

            # region 2: Creating colored bar
            for kind in ['add', 'del', 'still']:

                if kind == 'add':
                    color = 155 # green
                    title_thing = 'added'
                    title = 'The model now uses the following classification rules for this class: '
                    tot_num = len(self.path_dict_add)
                    Ns = self.N_add
                elif kind == 'del':
                    color = 1 # red
                    title_thing = 'deteted'
                    title = 'The model now uses the following classification rules for this anymore: '
                    tot_num = len(self.path_dict_del)
                    Ns = self.N_del
                elif kind == 'still':
                    color = 220 #yellow ---> 4: Blue
                    title_thing = 'unchanged'
                    title = 'The following classification rules are unchanged throughout time:'
                    tot_num = len(self.path_dict_still)
                    Ns = self.N_still

                title = self.text_formatter(title, bc = color)

                if tot_num > 0:
                    f.write(title)

            # endregion

            # region 3: Starting total and used paths
                if tot_num == 0:
                    colored_title_thing = self.text_formatter(title_thing, bc = color)
                    f.write(f'There are no "{colored_title_thing}" classification rules.\n')
                    continue

                rule_rules = 'rule'
                if tot_num > 1:
                    rule_rules = 'rules'

                to_print = f'This class has {tot_num} {title_thing.lower()} classification {rule_rules}.'


                # if not all of them are used for the classifcation:
                if sum(Ns) < tot_num:
                    to_print = to_print[:-1] # Remove .
                    is_are = 'are'
                    if sum(Ns) == 1:
                        is_are = 'is'
                    to_print += f', but only {sum(Ns)} {is_are} used to classify the {int(self.threshold* 100)}% of the items.'
                
                f.write(self.wrapper.fill(to_print))
                f.write(f'\n')

            # endregion
