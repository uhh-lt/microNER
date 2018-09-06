# microNER: A microservice for German Named Entity Recognition

Abstract


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
    "model" : "germeval-conll.h5"
  },
  "data" : {
    "sentences" : [
        "Ich bin Sabine Müllers erster CDU-Generalsekretär.",
        "Tom F. Manteufel wohnt auf der Insel Borkum."
    ],
    "tokens" : [["Ich", "bin", "Franz", ",", "du", "bist", "Micha"], ["Micha"]]
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

## Build your own Docker image

Checkout this git repository and run within the project directory

`docker build -t my_ner_image .`


## Citation

text

`
bibtex
`
