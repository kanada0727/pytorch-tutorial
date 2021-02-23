# pytorch-tutorial
Simple pytorch tutorial for MNIST, integrated with 
[pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) 
and 
[hydra](https://github.com/facebookresearch/hydra)

## tested environment
* windows10, wsl2 (ubuntu-18.04)
* docker, docker-compose (inside wsl)
* nvidia RTX 3090
* cuda 11.2 (Driver Version: 465.12)
* developed with VSCode

## setup
1. inside WSL2,
    ```
    git@github.com:kanada0727/pytorch-tutorial.git
    cd pytorch-tutorial
    make setup
    ```
2. modify `.env` to set directories for log and dataset



## activate environment
two options:


1. on WSL2:
    ```
    make up 
    make sh
    ```
2. VSCode on WSL2:
    1. launch VSCode on current directory
    2. install [remote containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) plugin
    3. use command: `Remote-Containers: Open Folder in Container ...`

## launch jupyter lab
### prepare: set password for jupyter lab
two options:
1. on WSL2:
    ```
    make update-password 
    ```
2. inside container:
    ```
    poetry run task jupyter-update-password
    ```

### launch
two options:
1. on WSL:
    ```
    make jupyter
    ```
2. inside container:
    ```
    poetry run task jupyter
    ```
then access http://localhost:8888 and type the password you set

If you want to use another port, activate environment by following command
```
port=<port-number-you-want-to-use> make up
```

### run training
inside container:
```
poetry run python -m pytorch_tutorial.train tag=<specifier-for-this-experiment>
```
you can modify hyperparameters hydra-way like:
```
poetry run python -m pytorch_tutorial.train tag=experiment-1 hparams.batch_size=256
```

### visualize training logs
see [tensorboard notebook](notebooks/tensorboard.ipynb)
