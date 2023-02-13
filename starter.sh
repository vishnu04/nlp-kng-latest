#!/bin/bash
  
# turn on bash's job control
set -m
  
# Start the primary process and put it in the background
echo "Calling app.py"
python app.py &

# Starting http server for D3 KNG
echo "Starting http server on port 9081"
python -m http.server 9081 --bind 127.0.0.1 &

# Start the helper process
echo "Calling Stanford server start"
# java -mx4g -cp '/usr/src/nlp-kng/stanford-corenlp-4.5.1/*' edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload depparse -status_port 9001 -port 9001 -timeout 3000000
java -mx4g -cp '/Users/vishnu/Drive/BANA/IndependentStudy/nlp_latest_gcp/nlp-kng-latest/stanford-corenlp-4.5.1/*'  edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,pos,lemma,ner,parse,dcoref,depparse -status_port 9001 -port 9001 -timeout 3000000 -threads 4

# the my_helper_process might need to know how to wait on the
# primary process to start before it does its work and returns
  
# now we bring the primary process back into the foreground
# and leave it there
fg %1