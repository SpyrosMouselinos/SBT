import requests
import json
from time import sleep
from src.common.connections.DatabaseConnections import InfluxMeasurements


def posting_alerts(influx_measurements):
    strategies = influx_measurements.get_active_strategies()
    strategies_not_posting = []
    strategies_not_trading = []
    for strategy in strategies:
        # posting
        posting_count, app = get_posts_count_for_strategy(strategy=strategy)
        if posting_count == 0:
            strategies_not_posting.append(strategy)
            print(f"WARNING: Strategy: {strategy}, Posting Count: {posting_count}")
        else:
            print(f"INFO: Strategy: {strategy}, Posting Count: {posting_count}")
        # trading
        executions_count, app = get_executions_count_for_strategy(strategy=strategy)
        if executions_count == 0:
            strategies_not_trading.append(strategy)
            print(f"WARNING: Strategy: {strategy}, Executions Count: {executions_count}")
        else:
            print(f"INFO: Strategy: {strategy}, Executions Count: {executions_count}")
        sleep(4)

    if len(strategies_not_posting) > 0:
        alert(strategy=",".join(strategies_not_posting), message="Not Posting for more than 1 Day", priority=1)
    if len(strategies_not_trading) > 0:
        alert(strategy=",".join(strategies_not_trading), message="Not Trading for more than 1 Day", priority=0)


def get_posts_count_for_strategy(strategy):
    buy_order = 'Placing Buy order succeeded'
    sell_order = 'Placing Sell order succeeded'

    headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "message": buy_order,
                                    }
                                },
                                {
                                    "match": {
                                        "message": sell_order,
                                    }
                                }
                            ]
                        }

                    },
                    {
                        "match": {
                            "strategy": strategy,
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": "now-1d",
                                "lt": "now"
                            }
                        }
                    }
                ]
            }
        }
    }
    data = json.dumps(body)
    url = f"http://elasticsearch:9200/_search"
    response = requests.get(url=url, headers=headers, data=data)
    text = json.loads(response.text)
    hits = text["hits"]["total"]
    if hits > 0:
        return hits, text['hits']['hits'][0]['_source']['node_name']
    else:
        return hits, None


def get_executions_count_for_strategy(strategy):
    already_balanced = 'Execution received, but already balanced'
    handle_more_short_than_long = 'diff:'
    error_catch = 'Error during balancing:'

    headers = {'Accept': 'application/json', 'Content-type': 'application/json'}
    body = {
        "query": {
            "bool": {
                "must": [
                    {
                        "bool": {
                            "should": [
                                {
                                    "match": {
                                        "message": already_balanced
                                    }
                                },
                                {
                                    "match": {
                                        "message": handle_more_short_than_long
                                    }
                                },
                                {
                                    "match": {
                                        "message": error_catch
                                    }
                                }
                            ]
                        }

                    },
                    {
                        "match": {
                            "strategy": strategy,
                        }
                    },
                    {
                        "range": {
                            "@timestamp": {
                                "gte": "now-1d",
                                "lt": "now"
                            }
                        }
                    }
                ]
            }
        }
    }
    data = json.dumps(body)
    url = f"http://elasticsearch:9200/_search"
    response = requests.get(url=url, headers=headers, data=data)
    text = json.loads(response.text)
    hits = text["hits"]["total"]
    if hits > 0:
        return hits, text['hits']['hits'][0]['_source']['node_name']
    else:
        return hits, None


def alert(strategy, message, priority, **kwargs):
    url = "http://nodered:1880/alerts"
    headers = {'Content-type': 'application/json'}
    params = {
        "text": message,
        "priority": priority,
        "strategy": strategy
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(params))
    print(params, response.status_code, response.text)


if __name__ == '__main__':
    measurements = InfluxMeasurements(single_measurement="executed_spread")
    posting_alerts(influx_measurements=measurements)
