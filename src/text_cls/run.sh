source .venv/Scripts/activate

PATH="src/text_cls/dataset/20newsgroups/full_data/val.csv"

python run.py --path $PATH \
              --noun_phrase