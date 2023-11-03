# Federated Learning

This project has implemented FedAvg algorithm to solve the hand digit recognition problem using MNIST dataset.

## Demo:
First, install all required packages
```
pip install -r requirements.txt
```

Next, setup Kafka:
```
docker-compose up -d
```

Then, run Visdom server to plot loss metric and accuracy metric real time:
```
Visdom
```

Next, start server:
```
python server.py ./data/MNIST/splitted/validate_dataset 
```

Then, start 2 clients in different terminals:
```
python client.py ./data/MNIST/splitted/train_dataset_0
```

```
python client.py ./data/MNIST/splitted/train_dataset_1
```

Finally, trigger training process:
```
curl localhost:5000/model/train
```
