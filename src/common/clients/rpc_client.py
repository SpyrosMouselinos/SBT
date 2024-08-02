import json
import os
import pika
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class RpcClient(object):
    def __init__(self):
        self.credentials = pika.PlainCredentials(os.getenv("RABBITMQ_USERNAME"), os.getenv("RABBITMQ_PASSWORD"))
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOSTNAME"),
                                      credentials=self.credentials))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.response = None
        self.corr_id = None

    def call(self, body, routing_key='simulation_rpc_queue'):
        self.channel.basic_publish(
            exchange='',
            routing_key=routing_key,
            body=json.dumps(body)
        )
        return


