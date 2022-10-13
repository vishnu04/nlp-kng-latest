FROM continuumio/miniconda3

WORKDIR /usr/src/nlp_kng

# Create the environment:
COPY environment.yml .
# RUN conda env create -n nlp_kng_venv
RUN conda env create -f environment.yml
# Make RUN commands use the new environment:
# RUN conda env create -n nlp_kng_venv python==3.9 && \
#     conda activate nlp_kng_venv

SHELL ["conda", "run", "-n", "nlp_kng_venv", "/bin/bash", "-c"]

RUN pip3 install --upgrade pip setuptools wheel

# # Demonstrate the environment is activated:
# RUN echo "Make sure flask is installed:"
# RUN python -c "import flask"

ENV PORT 5001
ENV HOST 0.0.0.0
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONFAULTHANDLER 1
ENV PYTHONMALLOC debug
EXPOSE 5001

WORKDIR /usr/src/nlp_kng

COPY ./requirements.txt ./requirements.txt
COPY . .
# RUN pip3 install --upgrade pip setuptools wheel

RUN pip3 install nltk
RUN python -c "import nltk;nltk.download('punkt', download_dir='/usr/local/nltk_data'); nltk.download('stopwords', download_dir='/usr/local/nltk_data')"
RUN python -m nltk.downloader stopwords
RUN pip3 install -U spacy
RUN python -m spacy download en_core_web_lg
RUN conda install -c conda-forge sentence-transformers
RUN conda install -c anaconda tensorflow
RUN conda install -c anaconda tensorflow-hub

# RUN pip3 install --upgrade tensorflow-hub
# RUN pip3 install sentence-transformers
# RUN pip3 install tensorflow
RUN pip3 install -r requirements.txt


# The code to run when container is started:
# COPY test.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "nlp_kng_venv", "python", "app.py"]