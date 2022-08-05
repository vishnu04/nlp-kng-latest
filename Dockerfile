FROM ubuntu:18.04

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Python package management and basic dependencies
RUN apt-get install -y curl python3.7 python3.7-dev python3.7-distutils wget
RUN apt-get -y install gcc make build-essential libssl-dev libffi-dev python3.7-dev
RUN apt-get -y install git python3-venv

# Register the version in alternatives
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.7 1

# Set python 3 as the default python
RUN update-alternatives --set python /usr/bin/python3.7

# Upgrade pip to latest version
RUN curl -s https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py --force-reinstall && \
    rm get-pip.py && \
    python -m pip install --upgrade setuptools pip wheel && \
    python -m pip install --upgrade pip && \
    python -m pip install --upgrade wheel && \
    python -m pip install gunicorn

# RUN apt-get update -y
# RUN apt-get install -y python3.7 python3-pip python3.7-dev python3.7-distutils 
# RUN apt-get install -y git
# RUN pip3 install --upgrade pip setuptools wheel

ENV PORT 5000
ENV HOST 0.0.0.0
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PYTHONMALLOC debug

EXPOSE 5000

WORKDIR /usr/src/nlp-kng

COPY ./requirements.txt ./requirements.txt
COPY . .

RUN git clone https://github.com/huggingface/neuralcoref.git
RUN cd neuralcoref && \
    pip3 install -r requirements.txt && \
    pip3 install -e .

RUN cd .. 
RUN mv neuralcoref neuralcoref_dummy
RUN mv neuralcoref_dummy/neuralcoref /usr/local/lib/python3.7/dist-packages

RUN pip3 install -r requirements.txt
RUN python -m spacy download en_core_web_sm
RUN pip3 install nltk
RUN python -c "import nltk;nltk.download('punkt')"

# RUN pip3 install -U Cython
# RUN pip3 install -U numpy
# RUN pip3 install -U spacy==2.1.3
# RUN pip3 install -U textacy==0.10.0
# RUN pip3 install -e git+https://github.com/huggingface/neuralcoref.git@60338df6f9b0a44a6728b442193b7c66653b0731#egg=neuralcoref
# RUN pip3 install -e git+https://github.com/huggingface/neuralcoref.git
# RUN pip3 install -r requirements.txt
# RUN pip install neuralcoref --no-binary neuralcoref
# RUN python -m spacy download en_core_web_sm
# RUN python -m nltk.downloader punkt
# RUN pip uninstall spacy
# RUN pip uninstall neuralcoref
# RUN pip3 install spacy==2.1.0
# RUN pip install neuralcoref --no-binary neuralcoref

# WORKDIR /usr/src/nlp-kng
# RUN rm -rf neuralcoref

RUN apt-get clean all

WORKDIR /usr/src/nlp-kng

ENTRYPOINT ["python"]
CMD ["app.py"]
