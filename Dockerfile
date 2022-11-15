ARG UBUNTU_VER=18.04
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.9

FROM ubuntu:${UBUNTU_VER}

# System packages 
RUN apt-get update && apt-get install -yq curl wget jq vim zip unzip
RUN apt-get install -y software-properties-common
RUN apt install -y gpg-agent
RUN apt install -y default-jre
RUN apt install -y default-jdk
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
ENV PATH=$JAVA_HOME/bin:${PATH}
# Use the above args during building https://docs.docker.com/engine/reference/builder/#understand-how-arg-and-from-interact
ARG CONDA_VER
ARG OS_TYPE

# Install miniconda to /miniconda
# RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
# RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
# RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
# ENV PATH=/miniconda/bin:${PATH}
# RUN conda update -y conda

# ARG PY_VER
# ARG TF_VER
# # Install packages from conda and downgrade py (optional).
# RUN conda install -c anaconda -y python=${PY_VER}

# RUN pip3 install --upgrade pip setuptools wheel

# ENV PORT 5001
# ENV HOST 0.0.0.0
# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONFAULTHANDLER 1
# ENV PYTHONMALLOC debug

# EXPOSE 5001 9001

# WORKDIR /usr/src/nlp-kng

# RUN wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.1.zip
# RUN unzip stanford-corenlp-4.5.1.zip
# RUN rm -r stanford-corenlp-4.5.1.zip

# COPY ./requirements.txt ./requirements.txt
# COPY . .

# RUN pip3 install nltk
# RUN python -c "import nltk;nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/nltk_data')"
# RUN python -m nltk.downloader stopwords
# RUN pip3 install -U spacy
# RUN python -m spacy download en_core_web_lg
# RUN conda install -c conda-forge sentence-transformers
# RUN conda install -c anaconda tensorflow
# RUN conda install -c anaconda tensorflow-hub
# RUN pip3 install -r requirements.txt

# RUN python -c "import gensim.downloader  as api;from gensim.models import Word2Vec;import pickle;import os;dataset = api.load('text8');model = Word2Vec(dataset);model_path = os.path.join('/usr/bin/nlp-kng/website', '/models/');pickle.dump(model,open(model_path+'gensim_text8_model','wb'), pickle.HIGHEST_PROTOCOL)"
# RUN python -c "import stanza; from stanza.server import CoreNLPClient; import os; from website.actions import config; os.environ['CORENLP_HOME'] = '/usr/bin/nlp-kng/stanza_nlp'; stanza.download('en')"

# RUN echo "java -mx4g -cp '/usr/src/nlp-kng/stanford-corenlp-4.5.1/*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,pos,depparse -status_port 9001 -port 9001 -timeout 300000" >> ~/.bashrc

# ENV CLASSPATH=/usr/src/nlp-kng/stanford-corenlp-4.5.1/stanford-corenlp-4.5.1-models.jar:/usr/src/nlp-kng/stanford-corenlp-4.5.1/stanford-corenlp-4.5.1.jar
# ENV PATH=/usr/src/nlp-kng/stanford-corenlp-4.5.1:${PATH}
# ENV CORENLP_DIR=/usr/src/nlp-kng/stanford-corenlp-4.5.1

RUN apt-get clean all

# WORKDIR /usr/src/nlp-kng

# RUN chmod 777 starter.sh
# ENTRYPOINT ["/bin/bash","-c","python"]
# ENTRYPOINT ["java -mx4g -cp '/usr/src/nlp-kng/stanford-corenlp-4.5.1/*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,parse,depparse -status_port 9001 -port 9001 -timeout 300000"]
# CMD ["java -mx4g -cp '/usr/src/nlp-kng/stanford-corenlp-4.5.1/*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,parse,depparse -status_port 9001 -port 9001 -timeout 300000", "python app.py"]
# ENTRYPOINT ["/bin/bash", "java","-mx4g","-cp","'/usr/src/nlp-kng/stanford-corenlp-4.5.1/*'","edu.stanford.nlp.pipeline.StanfordCoreNLPServer", \
#            "-preload","depparse" ,"-status_port", "9001", "-port", "9001","-timeout","300000", "&& python ../app.py"]
# WORKDIR /usr/src/nlp-kng
# CMD ["python app.py"]
# CMD ./starter.sh
ENTRYPOINT ["/bin/bash"]
# CMD ["starter.sh"]
