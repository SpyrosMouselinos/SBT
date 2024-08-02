# TakerMakerWorker Module Documentation

## Description
The **TakerMakerWorker** module is part of a larger system designed for running trading simulations in a distributed environment. This component is implemented using Python and primarily interacts with RabbitMQ to manage simulation tasks.

### Dependencies
This module relies on several Python libraries:

- `pika`: For RabbitMQ connections.
- `os`, `json`, `threading`, `functools`, `traceback`: Standard Python libraries for various utilities.
- `dotenv`: For environment variable management.
- `src.simulations.simulation_codebase.execute_simulations.simulation_maker_taker_function`: Custom functions for executing simulations.

### Environment Variables
The module requires the following environment variables, which should be defined in a `.env` file or passed directly to the execution environment:
- `RABBITMQ_USERNAME`
- `RABBITMQ_PASSWORD`
- `RABBITMQ_HOSTNAME`

## **TakerMakerWorker** Class
The `TakerMakerWorker` class is responsible for handling and processing messages from the RabbitMQ queue to run trading simulations.

### Class Initialization
```python
class TakerMakerWorker:
    def __init__(self, queue="simulation_rpc_queue"):
        self.credentials = pika.PlainCredentials(
            os.getenv("RABBITMQ_USERNAME"), os.getenv("RABBITMQ_PASSWORD"))
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=os.getenv("RABBITMQ_HOSTNAME"), credentials=self.credentials, heartbeat=5))
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue=queue)
        self.threads = []
```
- **queue**: The name of the RabbitMQ queue (default is `"simulation_rpc_queue"`).

### Methods
#### **on_message**
Handles incoming messages from the RabbitMQ queue.
```python
def on_message(self, ch, method, props, body):
    delivery_tag = method.delivery_tag
    t = threading.Thread(target=self.work, args=(
        self.connection, self.channel, delivery_tag, body))
    t.start()
    self.threads.append(t)
```
- **ch**: Channel.
- **method**: Method frame.
- **props**: Properties.
- **body**: Message body.

#### **work**
Processes the actual workload in a separate thread.
```python
def work(self, connection, channel, delivery_tag, body):
    thread_id = threading.get_ident()
    print(f'Thread id: {thread_id} Delivery tag: {delivery_tag}')
    params = json.loads(body)
    print(f"Received simulation with params: {json.dumps(params, indent=2)}")
    try:
        functions[params['function']](params=params)
    except Exception as e:
        traceback.print_exc()
    cb = functools.partial(self.ack_message, channel, delivery_tag)
    connection.add_callback_threadsafe(cb)
```
- **connection**: The RabbitMQ connection object.
- **channel**: The RabbitMQ channel object.
- **delivery_tag**: Unique identifier for the delivery.
- **body**: The message content, expected to be a JSON-encoded string containing the simulation parameters.

#### **on_debug**
This method could be used for debugging purposes.
```python
def on_debug(self, body):
    params = json.loads(body)
    print(f"Received simulation with params: {json.dumps(params, indent=2)}")
    functions[params['function']](params=params)
```
- **body**: The message content.

#### **ack_message**
Acknowledges a RabbitMQ message.
```python
def ack_message(self, channel, delivery_tag):
    if channel.is_open:
        channel.basic_ack(delivery_tag)
    else:
        pass
```
- **channel**: The RabbitMQ channel object.
- **delivery_tag**: Unique identifier for the delivery.

### Example Usage
In the main execution block, you can create and run a `TakerMakerWorker` as follows:
```python
if __name__ == '__main__':
    worker = TakerMakerWorker()
```

### Summary Table
| Method         | Description                                         | Input Parameters |
|----------------|-----------------------------------------------------|------------------|
| **on_message** | Handles incoming messages from RabbitMQ             | ch, method, props, body |
| **work**       | Processes the workload in a separate thread         | connection, channel, delivery_tag, body |
| **on_debug**   | Used for debugging purposes                         | body |
| **ack_message**| Acknowledges a RabbitMQ message                     | channel, delivery_tag |

### Notes
- Ensure RabbitMQ is properly set up and running with the correct environment variables set up to establish the connection.
- Handle exceptions gracefully to avoid crashing the worker and ensure to log errors for debugging and maintenance.

This detailed documentation covers the structure, dependencies, and functionality of the `TakerMakerWorker` class, making it easier for developers to understand and integrate or enhance the functionality as needed.