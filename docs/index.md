# microNER: A microservice for German Named Entity Recognition

We publish several pre-trained models for Named Entity Recognition wrapped into a micro-service based on Docker to allow for easy integration of German NER into other applications via a JSON API. 
With F-Scores above 82\% for the GermEval'14 dataset and above 85\% for the CoNLL'03 dataset, the micro-service achieves (near) state-of-the-art performance for this task.

The service relies on bidirectional recurrent neural networks and CRF. For details, see [this paper](https://www.oeaw.ac.at/fileadmin/subsites/academiaecorpora/PDF/konvens18_19.pdf).

## Installation

Using microNER requires Docker to be installed and running.

Further, it uses pre-trained fastText embeddings which require quite some disk space. 15 GB should be available on your hard disk where Docker images are stored.

Download the image form Docker hub:

```
docker pull uhhlt/microner:v0.1
```

And run the image with a port exposed on your host, e.g. 5001:

```
docker run -p 5001:5001 uhhlt/microner:v0.1
```


## Usage

Once the docker container is running, it accepts JSON requests in the following format:
```
{
  "meta" : {
    "model" : "MODELNAME"
  },
  "data" : {
    "sentences" : [
        "This is sentence 1.",
        "This is sentence 2."
    ],
    "tokens" : [
		["Token", "sequence", "as", "array"], 
		["Token", "seqeunce", "as", "array"]
	]
  }
}
```

Four models are available which can be selected via the `meta` parameter in the JSON request:
1. germeval.h5
2. germeval-inner.h5
3. germeval-conll.h5
4. conll.h5

Text for named entity recognition needs to be given in the `data` object either pre-tokenized 
(which is advised since you probably want to keep control on that) 
passed as an array of an array of strings in the `tokens` property, or in a array of separate 
sentences in the `sentences` property.

The micro-service returns sentence-wise arrays of tokens attached with their respective 
entity label.

## Test

From the command line, you can test a running instance of microNER with curl:

```
curl --request POST http://localhost:5001/api --header "Content-Type: application/vnd.api+json" --data '
{
  "meta" : {
    "model" : "germeval-conll.h5"
  },
  "data" : {
    "sentences" : [
        "Ich bin Sabine Müllers erster CDU-Generalsekretär.",
        "Tom F. Manteufel wohnt auf der Insel Borkum."
    ],
    "tokens" : [
		["Ich", "bin", "Franz", ",", "du", "bist", "Micha", "."], 
		["Micha", "wohnt", "in", "Berlin", "."]
	]
  }
}
'
```

## Build your own Docker image

Checkout this git repository and run within the project directory

`docker build -t my_ner_image .`


## Citation

Wiedemann, Gregor; Jindal, Raghav; Biemann, Chris (2018): microNER: A Micro-Service for German Named Entity Recognition based on BiLSTM-CRF. In: Proceedings of the 14th Conference on Natural Language Processing (KONVENS), S. 165–171.

`
@inproceedings{Wiedemann.2018h,
 author = {Wiedemann, Gregor and Jindal, Raghav and Biemann, Chris},
 title = {microNER: A Micro-Service for German Named Entity Recognition based on BiLSTM-CRF},
 pages = {165--171},
 booktitle = {Proceedings of the 14th Conference on Natural Language Processing (KONVENS)},
 year = {2018}
}
`
