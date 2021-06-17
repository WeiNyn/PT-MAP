# PT-MAP 

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

The source code for PT-MAP few-shot learning, including

- Model implementation
- Trainer
- Source code for server deployment that run on the pattern pipeline

## Requirements

- Torch
- Pillow
- numpy
- Torchvison
- Pytorch lightning
- Uvicorn
- FastAPI

## Setting

File setting/setting.py stores all setting:

```python
class Setting:
    data_folder: str = '120_samples_database_cut/'
    train_data_len: int = 10000
    test_data_len: int = 1200
    start_epoch: int = 0
    stop_epoch: int = 1000
    train_batch_size: int = 16
    test_batch_size: int = 16
    train_aug: bool = True
    image_size: int = 84 #model input size
    num_classes: int = 120
    save_freq: int = 5
    lr: float = 0.001
    alpha: float = 2.0
    checkpoint_dir: str = 'checkpoints/'
    resume: str = None
    es_scale: float = 0.1
    patience: int = 5
```

## Usage

Change the data_folder in setting, the folder must follow the ImageFolder dataset format.

You can change the train souce code for replacing the resnet18 with resnet50, se source code implementation for mor details

Checkpoint stored at __checkpoints/__

Run training code by:

```sh
python src/train/trainer.py
```

You can run a demo server by:
```sh
uvicorn app.main:app --port 8888 --host 0.0.0.0
```

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
