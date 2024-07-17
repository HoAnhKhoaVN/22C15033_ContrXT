import customtkinter
import tkinter
from typing import Text
from utils.load_data import read_json_file, write_json_file

DATA_PATH = 'data/elementary.json'

class App(customtkinter.CTk):
    def __init__(self, data_path: Text)-> None:
        super().__init__()
        self.data_path = data_path
        self.full_data = read_json_file(self.data_path)
        self.current_id = self.full_data['current_id']
        self.data = self.full_data['data']

        self.char2int = {
            'a':0,
            'b':1,
            'c':2,
            'd':3,
        }
        
        self.int2char = {
            0: 'A',
            1: 'B',
            2: 'C',
            3: 'D'
        }

        self.configure_window()
        self.configure_title()
        self.configure_questions()
        self.configure_choices()
        self.configure_next_button()
        self.configure_previous_button()
        self.configure_save_button()
        self.configure_label_index()
        self.configure_search_index()
    
    def configure_search_index(self):
        self.enter_index_label = customtkinter.CTkLabel(
            self,
            text="Enter index: ",
            font=customtkinter.CTkFont(
                size=20,
                weight="bold"
            ),
            text_color= 'red'
        )
        self.enter_index_label.grid(
            row = 1,
            column = 3,
            pady=(0, 0), 
            padx=0, 
            sticky="n"
        )

    def configure_label_index(self):
        idx = self.current_id
        _id = self.data[idx]['id']
        self.index_label = customtkinter.CTkLabel(
            self,
            text=f"Index: {idx} - ID: {_id}",
            font=customtkinter.CTkFont(
                size=20,
                weight="bold"
            ),
            text_color= 'green'
        )
        self.index_label.grid(row=1, column=2, padx=0, pady=(0, 0))



    def configure_window(self):
        self.title("Tool label")
        self.geometry(f"{1000}x{500}")

    def configure_title(self):
        self.logo_label = customtkinter.CTkLabel(
            self,
            text="Gán nhãn LLM làm toán",
            font=customtkinter.CTkFont(
                size=30,
                weight="bold"
            ),
            text_color= 'red'
        )
        self.logo_label.grid(row=0, column=2, padx=0, pady=(0, 0))

    def configure_questions(self):
        question_text = self.split_question(self.data[self.current_id]["question"])
        self.question_label = customtkinter.CTkLabel(
            self,
            text= question_text,
            font=customtkinter.CTkFont(
                size=20,
                weight="bold"
            ),
            text_color= 'black'
        )
        self.question_label.grid(row=1, column=0, padx=0, pady=(0, 0))

    @staticmethod
    def get_index_of_answer(answer: Text)->Text:
        res = answer.split('. ')[0]
        res = res.strip().lower()
        return res
    
    @staticmethod
    def split_question(question: Text)->Text:
        lst_question = question.split()
        length = len(lst_question)
        len_sent = length//10

        start = 0
        end = 0
        res = []
        for idx in range(len_sent):
            start = idx*10
            end = (idx + 1) * 10
            tmp_sent = ' '.join(lst_question[start: end])
            res.append(tmp_sent)
        
        res.append(' '.join(lst_question[end: length]))

        return '\n'.join(res)

        


    def configure_choices(self):
        self.multi_choices = []
        answer = self.data[self.current_id].get('answer', None)
        if answer is not None:
            answer = self.get_index_of_answer(answer)
            self.radio_var = tkinter.IntVar(value=self.char2int[answer])
        else:
            self.radio_var = tkinter.IntVar(0)
        current_choices = self.data[self.current_id]["choices"]
        for idx, canidate in enumerate(current_choices):
            x = customtkinter.CTkRadioButton(
                master=self, 
                variable=self.radio_var, 
                value=idx,
                text = canidate
            )
            x.grid(
                row=2+idx, 
                column=1, 
                pady=(0, 0),
                padx=0, 
                sticky="n"
            )
            self.multi_choices.append(x)

    def configure_next_button(self):
        self.next_bt = customtkinter.CTkButton(
            master=self, 
            fg_color="transparent", 
            border_width=2, 
            text_color=("gray10", "#DCE4EE"),
            text= 'Next',
            command=self.process_next
        )
        self.next_bt.grid(
            row = 6,
            column = 5,
            pady=(0, 0), 
            padx=0, 
            sticky="n"
        )

    def configure_previous_button(self):
        self.prev_bt = customtkinter.CTkButton(
            master=self, 
            fg_color="transparent", 
            border_width=2, 
            text_color=("gray10", "#DCE4EE"),
            text= 'Prev',
            command=self.process_prev
        )
        self.prev_bt.grid(
            row = 6,
            column = 7,
            pady=(0, 0), 
            padx=0, 
            sticky="n"
        )

    def configure_save_button(self):
        self.save_bt = customtkinter.CTkButton(
            master=self, 
            fg_color="transparent", 
            border_width=2, 
            text_color=("gray10", "#DCE4EE"),
            text= 'Save',
            command=self.process_save
        )
        self.save_bt.grid(
            row = 6,
            column = 9,
            pady=(0, 0), 
            padx=0, 
            sticky="n"
        )

    def process_save(self):
        self.full_data['current_id'] = self.current_id
        self.full_data['data'] = self.data
        write_json_file(path=self.data_path, json_obj= self.full_data)

    def process_next(self):
        # region write current answer
        self.data[self.current_id]['answer'] = self.int2char[self.radio_var.get()]
        # endregion

        # region display next question
        self.current_id+=1
        question_text = self.split_question(self.data[self.current_id]['question'])
        self.question_label.configure(text = question_text )

        idx = self.current_id
        _id = self.data[idx]['id']
        self.index_label.configure(text = f'Index: {idx} - ID : {_id}')
        
        answer = self.data[self.current_id].get('answer', None)
        if answer is not None:
            answer = self.get_index_of_answer(answer)
            self.radio_var = tkinter.IntVar(value=self.char2int[answer])
        else:
            self.radio_var = tkinter.IntVar(0)
        current_choices = self.data[self.current_id]["choices"]
        for idx, canidate in enumerate(current_choices):
            self.multi_choices[idx].configure(text = canidate, variable = self.radio_var)
        # endregion
    
    def process_prev(self):
        # region write current answer
        self.data[self.current_id]['answer'] = self.int2char[self.radio_var.get()]
        # endregion

        # region display next question
        self.current_id-=1
        question_text = self.split_question(self.data[self.current_id]['question'])
        self.question_label.configure(text = question_text )

        idx = self.current_id
        _id = self.data[idx]['id']
        self.index_label.configure(text = f'Index: {idx} - ID : {_id}')
        
        answer = self.data[self.current_id].get('answer', None)
        if answer is not None:
            answer = answer.strip().lower()
            self.radio_var = tkinter.IntVar(value=self.char2int[answer])
        else:
            self.radio_var = tkinter.IntVar(0)
        current_choices = self.data[self.current_id]["choices"]
        for idx, canidate in enumerate(current_choices):
            self.multi_choices[idx].configure(text = canidate, variable = self.radio_var)
        # endregion
if __name__ == "__main__":
    app = App('data/math_test_chatgpt_4.json')
    app.mainloop()

