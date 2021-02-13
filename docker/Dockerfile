
FROM nvidia/cuda:11.0-runtime-ubuntu18.04 as base
ENV LC_ALL=C.UTF-8

###-----------------------------------------------
### python builder
###-----------------------------------------------
FROM base as python-builder
###### --- add repository for latest minor version of python 3.8 ---
RUN apt update && apt install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt update && apt upgrade -y

###### --- install python3.8 and pip ---
RUN apt install -y python3.8 python3.8-dev python3.8-venv python3-pip
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN pip3 install --upgrade pip

###-----------------------------------------------
### poetry builder
###-----------------------------------------------
FROM python-builder as poetry-builder
RUN apt update && apt install -y curl
RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -

###-----------------------------------------------
### project base
###-----------------------------------------------
FROM python-builder as project-base
###### --- install poetry ---
COPY --from=poetry-builder /root/.poetry /root/.poetry
COPY --from=poetry-builder /root/.bashrc /root/.bashrc
ENV PATH $PATH:/root/.poetry/bin
RUN poetry config virtualenvs.in-project true

###-----------------------------------------------
### development
###-----------------------------------------------
FROM project-base as development

###### --- install nodejs for jupyter ---
COPY --from=node:15.6-buster-slim /usr/local/bin/node /usr/local/bin/node
COPY --from=node:15.6-buster-slim /usr/local/bin/nodejs /usr/local/bin/nodejs
COPY --from=node:15.6-buster-slim /usr/local/lib/node_modules/ /usr/local/lib/node_modules/
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm

###### --- install basic commands ---
RUN apt update && apt install -y procps git vim curl

###### --- setting project path ---
###### NOTE: put this code in the end to share cache with other projects
ENV PROJECT_PATH /root/pytorch-tutorial
WORKDIR $PROJECT_PATH
CMD ["/bin/bash"]

###-----------------------------------------------
### production builder
###-----------------------------------------------
FROM project-base as production-builder

###### --- setting project path ---
ENV PROJECT_PATH /root/pytorch-tutorial
WORKDIR $PROJECT_PATH

###### --- install python packages ---
COPY ./poetry.lock ./poetry.lock
COPY ./pyproject.toml ./pyproject.toml
RUN poetry install --no-dev

###-----------------------------------------------
### production
###-----------------------------------------------
FROM base as production

###### --- setting project path ---
ENV PROJECT_PATH /root/pytorch-tutorial
WORKDIR $PROJECT_PATH

COPY ./ ./

###### --- copy python runtime ---
COPY --from=production-builder $PROJECT_PATH/.venv/ ./venv
ENV PATH $PATH:$PROJECT_PATH/.venv/bin/python3

CMD ["/bin/bash"]
