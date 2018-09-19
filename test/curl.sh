# empty request
curl --request POST http://localhost:5001/api

# post data from file
curl --request POST --data @example.json http://localhost:5001/api --header "Content-Type: application/vnd.api+json"

# post data from command line
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
    "tokens" : [["Ich", "bin", "Franz", ",", "du", "bist", "Micha"], ["Micha"]]
  }
}
'