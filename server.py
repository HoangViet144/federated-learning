from flask import Flask, Response
from kafka import KafkaConsumer
import json
import threading
import time
import logging
from model import NeuralNet
import pickle
import base64
from constant import KAFKA_HOST, KAFKA_CONSUMER_GROUP, FD_KAFKA_TOPIC, MIN_RECEIVE_NO_CLIENT, TrainStatus, MAX_TRAIN_ROUND
import torch
import sys

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class EarlyStopper:
  def __init__(self, patience=1, min_delta=0):
    self.patience = patience
    self.min_delta = min_delta
    self.counter = 0
    self.min_validation_loss = float('inf')

  def early_stop(self, validation_loss):
    if validation_loss < self.min_validation_loss:
      self.min_validation_loss = validation_loss
      self.counter = 0
    elif validation_loss > (self.min_validation_loss + self.min_delta):
      self.counter += 1
      if self.counter >= self.patience:
        return True
    return False

class EndpointAction(object):
  def __init__(self, action):
    self.action = action
  def __call__(self, *args):
    data = self.action()
    return Response(response=data, status=200, headers={})

class FlaskAppWrapper(object):
  def __init__(self, name, validate_dataset_path):
    self.lock = threading.Semaphore()

    self.app = Flask(name)
    self.kafka_consumer = KafkaConsumer( 
      bootstrap_servers=[KAFKA_HOST],
      auto_offset_reset="latest",
      enable_auto_commit=True,
      group_id=KAFKA_CONSUMER_GROUP,
      value_deserializer=lambda x: pickle.loads(x)
    )
    self.loss_fn = torch.nn.CrossEntropyLoss()

    raw_validate_dataset = torch.load(validate_dataset_path)
    self.validate_dataset = torch.utils.data.DataLoader(
      dataset=raw_validate_dataset,
      batch_size=100,
      shuffle=False
    )

    # share state, need lock
    self.model = NeuralNet()
    self.model_version = 1
    self.train_status = TrainStatus.END
    self.start_train_time = 10**10
    self.aggregate_state = dict()
    self.client_ids = dict()
    self.train_round = 0
    self.early_stoper = None

  def run(self):
    self.kafka_consumer.subscribe([FD_KAFKA_TOPIC])
    handle_weight_recv_thread = threading.Thread(
      target=self.handle_receive_model_weight
    )
    handle_weight_recv_thread.daemon = True
    handle_weight_recv_thread.start()

    self.app.run()
  
  def handle_receive_model_weight(self):
    for message in self.kafka_consumer:
      client_id = message.value.client_id
      client_parameter = message.value.model_parameter

      if message.timestamp < self.start_train_time or self.train_status != TrainStatus.START:
        print("Discard out of date parameter of client {}".format(client_id))
        continue

      self.lock.acquire()

      if client_id in self.client_ids:
        print("Client {} already sent parameter this round".format(client_id))
        self.lock.release()
        continue
      
      self.client_ids[client_id] = True
      for layer, value in client_parameter.items():
        self.aggregate_state[layer] += value

      if len(self.client_ids) >= MIN_RECEIVE_NO_CLIENT:
        print('done train round {}, training time: {} seconds'.format(self.train_round, time.time() - self.start_train_time))
        self.start_train_time = 10**10

        self.update_model_parameter()
        validate_loss = self.evaluate_model()

        if self.should_stop_training(validate_loss):
          self.train_status = TrainStatus.END
          self.start_train_time = 10**10
          torch.save(self.model.state_dict(), "./trained_model/{}".format(self.model_version))
          self.lock.release()
          continue

        self.start_next_training_round()

      self.lock.release()
  

  def should_stop_training(self, validate_loss):
    if self.train_round > MAX_TRAIN_ROUND:
      return True
    return self.early_stoper.early_stop(validate_loss)
  
  def update_model_parameter(self):
    no_client = len(self.client_ids)
    for layer in self.aggregate_state:
      self.aggregate_state[layer] = self.aggregate_state[layer] / no_client

    self.model.load_state_dict(self.aggregate_state)

  def evaluate_model(self):
    self.model.eval()

    running_vloss = 0.0
    correct_predict, total_predict = 0, 0

    with torch.no_grad():
      for i, validate_data in enumerate(self.validate_dataset):
        validate_input, validate_label = validate_data
        validate_output = self.model(validate_input)
        vloss = self.loss_fn(validate_output, validate_label)
        running_vloss += vloss

        _, predicted = torch.max(validate_output.data, 1)

        # calculate accuracy
        total_predict += validate_label.size(0)
        correct_predict += (predicted == validate_label).sum().item()

    avg_vloss = running_vloss / (i + 1)
    print('LOSS train round{} valid {}'.format(self.train_round, avg_vloss))
    print('Accuracy: {.2f} %'.format(correct_predict / total_predict * 100))

    return avg_vloss

  def start_next_training_round(self):
    self.reset_aggregate_state()
    self.client_ids = {}
    self.train_round += 1
    self.start_train_time = time.time()
    print('start train round {}'.format(self.train_round))

  def add_endpoint(self, endpoint=None, endpoint_name=None, handler=None, method='GET'):
    self.app.add_url_rule(endpoint, endpoint_name, EndpointAction(handler), methods=[method])

  def helloworld(self):
    return "hello-world"

  def get_model_version(self):
    self.lock.acquire()
    model_version = str(self.model_version)
    self.lock.release()
    return model_version
  
  def start_training(self):
    self.lock.acquire()
    if self.train_status != TrainStatus.END:
      self.lock.release()
      return

    self.model_version += 1
    self.train_status = TrainStatus.START
    self.early_stoper = EarlyStopper(5, 0.01)
    self.train_round = 0
    self.start_next_training_round()

    self.lock.release()

  def reset_aggregate_state(self):
    self.aggregate_state = {}
    state_dict = self.model.state_dict()
    for layer in state_dict:
      self.aggregate_state[layer] = torch.zeros(state_dict[layer].shape)

  def get_model_parameter(self):
    self.lock.acquire()
    model_parameter = base64.b64encode(pickle.dumps(self.model.state_dict()))
    self.lock.release()
    return model_parameter
  
  def get_training_info(self):
    self.lock.acquire()
    training_info = json.dumps({
      'train_round': self.train_round,
      'model_version': self.model_version,
      'train_status': self.train_status
    })
    self.lock.release()
    return training_info

if __name__ == "__main__":
  try:
    _, validate_dataset_path = sys.argv
    server = FlaskAppWrapper('fd', validate_dataset_path)
    server.add_endpoint(endpoint='/', endpoint_name='hello-world', handler=server.helloworld)
    server.add_endpoint(endpoint='/model/version', endpoint_name='get-model-version', handler=server.get_model_version)
    server.add_endpoint(endpoint='/model/train', endpoint_name='train-model', handler=server.start_training)
    server.add_endpoint(endpoint='/model/parameter', endpoint_name='get-model-parameter', handler=server.get_model_parameter)
    server.add_endpoint(endpoint='/model/train/status', endpoint_name='get-model-train-status', handler=server.get_training_info)
    server.run()
  except Exception as e:
    print(e)
  