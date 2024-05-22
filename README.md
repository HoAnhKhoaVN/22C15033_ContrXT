# Duplicate Contrastive Explanation

Reference from paper [ContrXT: Generating contrastive explanations from any text classifier](https://www.sciencedirect.com/science/article/abs/pii/S1566253521002426)

## Install
1. Connect conda
```sh
D:

cd D:\Desktop\xAI\exp\22C15033_ContrXT
conda env list

conda activate 'env_conda'
```

### Run demo

```sh
sh scripts/demo/demo.sh

python scripts/demo/demo.py
```


### Run on server
1. Connect virtual enviroment
```sh
conda activate khoaha_venv
```

2. List conda enviroment
```sh
conda env list
```

3. Install `pip` in Conda
```sh
conda install pip
```


4. Install Graphviz
```sh
sudo apt-get install graphviz
```

5. Experiment with test classification
```sh
sh scripts/text_cls/run_text_cls.sh
```


```sh
conda activate khoaha_venv
```