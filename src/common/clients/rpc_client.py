import json
import os
import pika
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


# Creates a client for communicating with the server. It is used to send and recieve data
class RpcClient(object):
    def __init__(self):
        """
         @brief Initialize RabbitMQ and connect to RabbitMQ. This is called by __init__ and should not be called directly
        """
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
        """
         @brief Send a message to Simulation RPC queue. This is a blocking call. The message will be routed to the exchange defined by routing_key
         @param body The message to be sent
         @param routing_key The routing key for the message
         @return True if message was sent False if it was not sent ( in which case the message will be dropped
        """
        self.channel.basic_publish(
            exchange='',
            routing_key=routing_key,
            body=json.dumps(body)
        )
        return
