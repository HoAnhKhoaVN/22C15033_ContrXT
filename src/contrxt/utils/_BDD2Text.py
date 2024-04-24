

import ast
import re
from typing import Dict, List, Text, Tuple

import pandas as pd
import textwrap
from apyori import apriori


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
        self.used_paths_still = [x for i, x in enumerate(self.path_dict_still) if self.N_still[i]]

        self.wrapper = textwrap.TextWrapper(width = wrapp_size)


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
    
    def remove_key(self, dic, key):
        try:
            del dic[key]
        except KeyError:
            pass

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
        assert isinstance(underline, bool), f'Underline should be integer not {type(underline)}'
        assert isinstance(_reversed, bool), f'Reversed should be integer not {type(_reversed)}'
        # endregion

        # region 2: Default color
        if bc is not None:
            bc = f'\u001b[48;5;{bc}m'
        else:
            bc = ''

        if tc is not None:
            tc = f'\u001b[38;5;{tc}m'
        else:
            tc = ""

        if bold:
            b = '\u001b[1m'
        else:
            b = ''


        if underline:
            u = '\u001b[4m'
        else:
            u = ''


        if _reversed:
            r = '\u001b[7m'
        else:
            r = ''

        # endregion

        # region 3: Format code
        text = f'{b}{u}{r}{bc}{tc}{text}\u001b[0m'

        # endregion

        return (text)
    def dict_to_list(self, d: Dict)->List:
        return [x[0] + str(x[1]) for x in d.items()]
    
    def simple_text(
        self,
        fn: Text
    )-> None:
        """Write rule in natural 

        Args:
            fn (Text): _description_
        """
        def agg_0_1(dictak: Dict)-> Tuple[List, List]:
            """Split tokens which ended as 0 or 1.

            Args:
                dictak (Dict): Dictionary for token.

            Returns:
                Tuple[List, List]: List for zero and one.
            """
            zeros, ones = [], []
            for k, v in dictak.items():
                if v==0:
                    zeros.append(k)
                else:
                    ones.append(k)
            return zeros, ones

        def list_to_string(
            data: List[Text],
            add_feature = True
        )->Text:
            """Concat list feature to string.

            Args:
                data (List[Text]): List of features.
                add_feature (bool, optional): _description_. Defaults to True.

            Returns:
                Text: A string which lists the features.
            """
            if len(data) > 1:
                t = ""
                data_1 = data[:-1]
                last = data[-1]

                for i in data_1:
                    i = self.text_formatter(i, bold= True)
                    t += f'{i}, '
                last = self.text_formatter(last, bold= True)
                t += f'and {last}'

                if add_feature:
                    t += ' features'
            
            elif len(data) == 1:
                t = self.text_formatter(data[0], bold= True)
                if add_feature :
                    t += ' feature'
            
            else:
                t = 'XXXX'
            
            return t

        def list_to_string_2(
            data: List[Text],
            is_pos: bool = True,
        )-> Text:
            """Concat list of features to string which color.
            If `is_pos` is True text color is `green`. Else, text color is `red`.

            Args:
                data (List[Text]): List of features.
                is_pos (bool): Check features are positive or negative. Default is True.

            Returns:
                Text: A string which lists the COLORED features.
            """
            t = data

            # region 1: Check positive or negative
            if is_pos:
                tc = 10 # green
            else:
                tc = 9 # red

            # endregion

            # region 2: Text formatter
            if len(data) > 1:
                t = ""
                data_1 = data[:-1]
                last = data[-1]

                for i in data_1:
                    i = self.text_formatter(i, bold= True, tc = tc)
                    t += f'{i}, '
                last = self.text_formatter(last, bold= True, tc = tc)
                t += f'and {last}'
            
            elif len(data) == 1:
                t = self.text_formatter(data[0], bold= True, tc = tc)

            # endregion

            return t

        def get_best_rule(dicts: Dict)-> Tuple:
            """_summary_

            Args:
                dicts (Dict): _description_

            Returns:
                Tuple: _description_
            """

            def get_apriori(path_dict):
                def dict_item_count(dict_list: List[Dict])-> List:
                    res = []
                    for d in dict_list:
                        dd = self.dict_to_list(d)

                        res.append(dd)
                    return res
                

                path_dict = dict_item_count(path_dict)
                association_rules = apriori(path_dict)
                association_rules = list(association_rules)

                res = []
                for item in association_rules:
                    if len(list(item[0])) > 1:
                        res.append(list(item[0]))

                return res

            rules = get_apriori(dicts)
            nums = []

            for rule in rules:
                keys = [x[:-1] for x in rule]
                vals = [x[-1] for x in rule]

                num = 0
                for d in dicts:
                    if all(d.get(k, '-') for k, v in zip(keys, vals)):
                        num+=1
                    nums.append(num* len(rule))

            maxnum = max(nums)
            maxind = nums.index(maxnum)
            return rules[maxind], maxnum

        def divide_rules(kind: Text):
            if kind == 'add':
                dicts = self.path_dict_add
            elif kind == 'del':
                dicts = self.path_dict_del
            elif kind == 'still':
                dicts = self.path_dict_still

            rule, _ = get_best_rule(dicts)

            
            matched = []
            rest = []

            for d in dicts:
                keys = [x[:-1] for x in rule]
                vals = [x[-1] for x in rule]

                if all(d.get(k, '-') == v for k, v in zip(keys, vals)):
                    matched.append(d)
                else:
                    rest.append(d)
                
            return ((matched, rule), rest)

        def rules_to_shared(data)-> Text:
            nums = [int(x[-1]) for x in data]
            words = [x[:-1] for x in data]
            pos_words = [x[:-1] for x in data if x[-1]=='1']
            neg_words = [x[:-1] for x in data if x[-1]=='0']
            _sum = sum(nums)
            if len(data) > 1:
                if _sum == 0:
                    return f'the document must {self.text_formatter("not", tc = 9, underline= True)} contain {list_to_string_2(words, False)}'
                if _sum == 1:
                    return f'the document must contain {list_to_string_2(pos_words)} and must not cntain {list_to_string_2(neg_words, False)}'
                if _sum == 2:
                    return f'the document must contain {list_to_string_2(words)}'
            elif len(data) == 1:
                if 1 in nums:
                    return f' the document must contain {list_to_string_2(words)}'
                return f'the document must {self.text_formatter("not", tc = 9, underline= True)} contain {list_to_string_2(words, False)}'


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

            # region 4:
            if kind == 'add':
                used_paths = self.used_paths_add
            elif kind == 'del':
                used_paths = self.used_paths_del
            elif kind == 'still':
                used_paths = self.used_paths_still

            # region 4.1: Basic case
            if sum(Ns) <= 4:
                num_list = []
                for i, item in enumerate(used_paths):
                    zeros, ones = agg_0_1(item)
                    num_list.append((i, len(zeros)+ len(ones)))
                
                num_list = list(sorted(num_list, key = lambda x: x[1]))
                num_list = [x[0] for x in num_list]

                for ni in num_list:
                    item = used_paths[ni]
                    rem = self.text_formatter('Having', tc = 10)
                    rem2 = self.text_formatter('not', tc =9, underline= True)

                    zeros, ones = agg_0_1(item)
                    string_on = list_to_string(ones, add_feature= False)
                    string_ze = list_to_string(zeros, add_feature = False)
                    string_ze_pos = list_to_string_2(zeros)
                    string_ze_neg = list_to_string_2(zeros, False)
                    string_on_pos = list_to_string_2(ones)

                    if 'XXXX' in string_on:
                        if 'XXXX' not in string_ze:
                            if len(zeros) > 1:
                                iii = 'are'
                            else:
                                iii = 'is'
                            
                            print_text = f' - If there {iii} {rem2} {string_ze_pos}.'
                            f.write(print_text)
                            print(print_text)
                    elif 'XXXX' not in string_on:
                        if 'XXXX' not in string_ze:
                            if len(zeros) > 1:
                                iii = 'are'
                            else:
                                iii = 'is'
                            print_text = f' - {rem} {string_on_pos} but {rem2} {string_ze_neg}.'
                            f.write(print_text)
                            print(print_text)   
                        else:
                            print_text = f' - {rem} {string_on_pos}.'
                            f.write(print_text)
                            print(print_text)
            # endregion
            # region 4.2: Complex case
            else:
                matched, rest = divide_rules(kind)
                has_rule = matched[0]
                rule = matched[1]

                # region: Getting the number of rules with some shared part inside.
                num_list = []
                for i, item in enumerate(has_rule):
                    zeros, ones = agg_0_1(item)
                    num_list.append((i, len(zeros) + len(ones)))
                num_list = list(sorted(num_list, key=lambda x: x[1]))
                sag = 0

                for ni in num_list:
                    item = has_rule[ni]
                    if Ns[ni]:
                        sag += 1

                # endregion
                
                # number of paths w/0 shared parts
                # some_num = sum(Ns[-(tot_num - len(matched[0])):])
                # paths used for classification

                final_remaining = sum(Ns)

                vaz = 1
                if sag > 1 and sum(Ns) > sag:
                    print_text = f'\n Out of these {sum(Ns)} classification rules, {sag} share the following criteria.'
                    f.write(print_text)
                    print(print_text)
                else:
                    vaz = 0
                
                keys_to_remove = []

                if final_remaining != sag:
                    if sag != 1:
                        print_text = f'{rules_to_shared(rule)}'
                        f.write(print_text)
                        print(print_text)

                        keys_to_remove = [x[:-1] for x in rule]
                
                ###############################################
                ###### STATE REMAINDER OF SHARED RULES ########
                ###############################################
                if len(has_rule) > 1: 
                    #################
                    # DELETED STUFF #
                    #################

                    if sag != 1:
                        if sum(Ns) > 2:
                            msg = 'In addition, one of the following must hold:'
                        elif sum(Ns) == 2:
                            msg = "in addition, the following must hold:"
                        
                        if vaz == 1:
                            print(msg)
                        
                        num_list = []
                        for i , item in enumerate(has_rule):
                            zeros, ones = agg_0_1(item)
                            num_list.append((i, len(zeros) + len(ones)))
                        num_list = list(sorted(num_list, key= lambda x: x[1]))
                        num_list = [x[0] for x in num_list]

                        #################
                        #### LISTING ####
                        #################

                        for ni in num_list:
                            item = has_rule[ni]

                            # checking for the frequency
                            if Ns[ni] and vaz ==1:

                                for k in keys_to_remove:
                                    self.remove_key(item, k)
                                rem = self.text_formatter("Having", tc = 10)
                                rem2 = self.text_formatter(text= 'not', tc = 9, underline= True)

                                zeros, ones = agg_0_1(item)
                                string_on = list_to_string(ones, add_feature= False)
                                string_ze = list_to_string(zeros, add_feature = False)
                                string_ze_pos = list_to_string_2(zeros)
                                string_ze_neg = list_to_string_2(zeros, False)
                                string_on_pos = list_to_string_2(ones)

                                if 'XXXX' in string_on:
                                    if 'XXXX' not in string_ze:
                                        if len(zeros) > 1:
                                            iii = 'are'
                                        else:
                                            iii = 'is'
                                        
                                        print_text = f' - If there {iii} {rem2} {string_ze_pos}.'
                                        f.write(print_text)
                                        print(print_text)
                                elif 'XXXX' not in string_on:
                                    if 'XXXX' not in string_ze:
                                        if len(zeros) > 1:
                                            iii = 'are'
                                        else:
                                            iii = 'is'
                                        print_text = f' - {rem} {string_on_pos} but {rem2} {string_ze_neg}.'
                                        f.write(print_text)
                                        print(print_text)   
                                    else:
                                        print_text = f' - {rem} {string_on_pos}.'
                                        f.write(print_text)
                                        print(print_text)
                        
                        remaining = tot_num - sag
                        final_remaining = sum(Ns[-(remaining):])

                        paths_to_list = rest

                        # if there is nothing shared
                        if sag != 1:
                            if final_remaining == 1:
                                rem = '\nHere is the remaining rule:'
                                f.write(f'{rem}')
                                print(f'{rem}')
                            elif final_remaining > 1:
                                rem = '\n Here are the remaining rules:'
                                f.write(rem)
                                print(rem)
                        
                        if sag == 1:
                            paths_to_list = used_paths
                        
                        to_show = sum(Ns) - sag
                        num_list = []

                        for i, item in enumerate(paths_to_list[:to_show]):
                            zeros, ones = agg_0_1(item)
                            num_list.append((i, len(zeros) + len(ones)))
                        num_list = list(sorted(num_list, key = lambda x: x[1]))
                        num_list = [x[0] for x in num_list]

                        ###############
                        ### LISTING ###
                        ###############

                        for ni in num_list:
                            item = paths_to_list[ni]

                            rem = self.text_formatter("Having", tc = 10)
                            rem2 = self.text_formatter('not', tc = 9, underline= True)
                            zeros, ones = agg_0_1(item)
                            string_on = list_to_string(ones, add_feature= False)
                            string_ze = list_to_string(zeros, add_feature = False)
                            string_ze_pos = list_to_string_2(zeros)
                            string_ze_neg = list_to_string_2(zeros, False)
                            string_on_pos = list_to_string_2(ones)

                            if 'XXXX' in string_on:
                                if 'XXXX' not in string_ze:
                                    if len(zeros) > 1:
                                        iii = 'are'
                                    else:
                                        iii = 'is'
                                    
                                    print_text = f' - If there {iii} {rem2} {string_ze_pos}.'
                                    f.write(print_text)
                                    print(print_text)
                            elif 'XXXX' not in string_on:
                                if 'XXXX' not in string_ze:
                                    if len(zeros) > 1:
                                        iii = 'are'
                                    else:
                                        iii = 'is'
                                    print_text = f' - {rem} {string_on_pos} but {rem2} {string_ze_neg}.'
                                    f.write(print_text)
                                    print(print_text)   
                                else:
                                    print_text = f' - {rem} {string_on_pos}.'
                                    f.write(print_text)
                                    print(print_text)

                        f.write('\n')
                        print()
            # endregion
            # endregion
