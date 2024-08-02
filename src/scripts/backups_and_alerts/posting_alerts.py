import requests
import json
from time import sleep
from src.common.connections.DatabaseConnections import InfluxMeasurements


def posting_alerts(influx_measurements):
    """
     @brief alert if there are posting or trading strategies in Influx. This is a function to be called from outside
     @param influx_measurements An instance of Influx
    """
    strategies = influx_measurements.get_active_strategies()
    strategies_not_posting = []
    strategies_not_trading = []
    # This function will run the strategy s tests.
    for strategy in strategies:
        # posting
        posting_count, app = get_posts_count_for_strategy(strategy=strategy)
        # Add a strategy to the list of strategies that are not posting.
        if posting_count == 0:
            strategies_not_posting.append(strategy)
            print(f"WARNING: Strategy: {strategy}, Posting Count: {posting_count}")
        else:
            print(f"INFO: Strategy: {strategy}, Posting Count: {posting_count}")
        # trading
        executions_count, app = get_executions_count_for_strategy(strategy=strategy)
        # Add a strategy to the list of strategies that are not trading.
        if executions_count == 0:
            strategies_not_trading.append(strategy)
            print(f"WARNING: Strategy: {strategy}, Executions Count: {executions_count}")
        else:
            print(f"INFO: Strategy: {strategy}, Executions Count: {executions_count}")
        sleep(4)

    # alerts the user if there are not posting strategies
    if len(strategies_not_posting) > 0:
        alert(strategy=",".join(strategies_not_posting), message="Not Posting for more than 1 Day", priority=1)
    # alerts if there are not trading strategies
    if len(strategies_not_trading) > 0:
        alert(strategy=",".join(strategies_not_trading), message="Not Trading for more than 1 Day", priority=0)


def get_posts_count_for_strategy(strategy):
    """
     @brief Get the number of posts for a strategy. This is a count of how many posts have been purchased for a strategy
     @param strategy The strategy to check.
     @return The number of posts for the strategy as a float between 0 and 1. If there are no posts for the strategy the value will be 0
    """
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
    # Returns hits text hits 0 _source node_name
    if hits > 0:
        return hits, text['hits']['hits'][0]['_source']['node_name']
    else:
        return hits, None


def get_executions_count_for_strategy(strategy):
    """
     @brief Get executions count for a strategy. This is a helper function for get_executions_count.
     @param strategy The strategy to query. Can be'full'or'all '.
     @return An integer count of executions that match the strategy or None if no match is found. Note that the number will be less than or equal to the number of executions
    """
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
    # Returns hits text hits 0 _source node_name
    if hits > 0:
        return hits, text['hits']['hits'][0]['_source']['node_name']
    else:
        return hits, None


def alert(strategy, message, priority, **kwargs):
    """
     @brief Send an alert to Nordic. This is a convenience function for making alerts easier to interact with
     @param strategy The strategy to alert on
     @param message The message to display in the alert ( can be HTML )
     @param priority The priority of the alert ( 0 - 9
    """
    url = "http://nodered:1880/alerts"
    headers = {'Content-type': 'application/json'}
    params = {
        "text": message,
        "priority": priority,
        "strategy": strategy
    }
    response = requests.post(url=url, headers=headers, data=json.dumps(params))
    print(params, response.status_code, response.text)


# This is the main function that is called from the main module.
if __name__ == '__main__':
    measurements = InfluxMeasurements(single_measurement="executed_spread")
    posting_alerts(influx_measurements=measurements)
