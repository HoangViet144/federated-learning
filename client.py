from kafka.producer import KafkaProducer
import sched, time
import torch
import requests
from model import NeuralNet
import sys
from torchvision import transforms
from kafka import KafkaAdminClient
from constant import KAFKA_HOST, FD_KAFKA_TOPIC, TrainStatus, Message
import uuid
import json
import pickle
import base64
from visdom import Visdom

class Client:
  def __init__(self, train_dataset_path):
    self.client_id = str(uuid.uuid4())

    self.viz = Visdom(port="8097", server="http://localhost", base_url="/", username="", password="")
    self.viz_loss_window = self.viz.line(
          Y=torch.zeros((1)).cpu(),
          X=torch.zeros((1)).cpu(),
          opts=dict(xlabel='epoch',ylabel='Loss',title='training loss',legend=['Loss {}'.format(self.client_id)]))

    raw_train_dataset = torch.load(train_dataset_path)
    self.train_dataset = torch.utils.data.DataLoader(
      dataset=raw_train_dataset,
      batch_size=100,
      shuffle=False
    )

    self.model_version = 1

    self.model = NeuralNet()
    self.optimizer = torch.optim.SGD(self.model.parameters(), 0.0001)
    self.loss_fn = torch.nn.CrossEntropyLoss()
    self.no_epoch = 1

    self.kafka_producer = KafkaProducer(
      bootstrap_servers=[KAFKA_HOST],
      value_serializer=lambda x: pickle.dumps(x)
    )  

    self.train_round = 0

    self.init_model()

  def init_model(self):
    is_model_outdated, server_model_version = self.is_model_outdated()

    if is_model_outdated:
      self.update_model()
      self.model_version = server_model_version
      print('finish init model, model version {}'.format(server_model_version))

  def update_model(self):
    print('updating model...')
    resp = requests.get('http://127.0.0.1:5000/model/parameter')
    server_parameter = pickle.loads(base64.b64decode(resp.text))
    self.model.load_state_dict(server_parameter)

  def run(self):
    while True:
      is_model_outdated, _ = self.is_model_outdated()
      if is_model_outdated:
        self.train_round = 0

        while True:
          server_train_status, server_model_version, server_train_round = self.get_train_status()
          if server_train_round > self.train_round:
            self.train_round = server_train_round
            print('start train round {} of client {}'.format(self.train_round, self.client_id))
            self.update_model()
            self.train()
            self.publish_model_parameter()
          else:
            if server_train_status == TrainStatus.END:
              self.model_version = server_model_version
              break
            else:
              time.sleep(1)
        
      time.sleep(1)

  def is_model_outdated(self):
    try:
      resp = requests.get('http://127.0.0.1:5000/model/version')
      server_model_version = int(resp.text)
      return server_model_version != self.model_version, server_model_version
    except Exception as e:
      print(e)
      return False, self.model_version

  def train(self):
    train_loss = 0.0
    start_time = time.time()

    for epoch in range(self.no_epoch):
      running_loss = 0
      epoch_loss = 0

      for i, (data, label) in enumerate(self.train_dataset):
        self.optimizer.zero_grad()
        output = self.model(data)
        loss = self.loss_fn(output, label)
        loss.backward()
        self.optimizer.step()

        running_loss += loss.item()
        epoch_loss += loss.item()

        if i % 100 == 0:
          running_loss = running_loss / 100 # loss per batch
          print('epoch {} batch {} loss: {}'.format(epoch, i, running_loss))
          running_loss = 0
    
      epoch_loss = epoch_loss / len(self.train_dataset)
      train_loss += epoch_loss
      print('epoch {} loss: {}'.format(epoch, epoch_loss))

      self.viz.line(X=torch.ones((1,1)).cpu()*self.train_round,Y=torch.Tensor([epoch_loss]).unsqueeze(0).cpu(),win=self.viz_loss_window,update='append')

    train_loss = train_loss / self.no_epoch
    print('train loss: {}'.format(train_loss))

  def publish_model_parameter(self):
    print('publish_model_parameter...')
    self.kafka_producer.send(FD_KAFKA_TOPIC, Message(self.client_id, self.model.state_dict()))

  def get_train_status(self):
    resp = requests.get('http://127.0.0.1:5000/model/train/status')
    data = json.loads(resp.text)
    return data['train_status'], data['model_version'], data['train_round']

if __name__ == "__main__":
  try:
    _, train_dataset_path = sys.argv
    client = Client(train_dataset_path)
    client.run()
  except Exception as e:
    print(e)
