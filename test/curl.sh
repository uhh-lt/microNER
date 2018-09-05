# empty request
curl --request POST http://ltgpu1:5001/api

# post data from file
curl --request POST --data @example.json http://ltgpu1:5001/api --header "Content-Type: application/vnd.api+json"
