from enum import Enum

KAFKA_HOST = 'localhost:9092'
KAFKA_CONSUMER_GROUP = 'federated-learning-group'
FD_KAFKA_TOPIC = 'federated-learning'
MIN_RECEIVE_NO_CLIENT = 2
MAX_TRAIN_ROUND = 200

class TrainStatus(str, Enum):
  START = 1
  END = 2

class Message:
  def __init__(self, client_id, model_parameter):
    self.client_id = client_id
    self.model_parameter = model_parameter