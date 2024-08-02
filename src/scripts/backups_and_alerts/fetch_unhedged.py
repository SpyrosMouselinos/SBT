import json
import psycopg2, os
import requests
import datetime
import urllib.parse
import time
from requests.auth import HTTPBasicAuth

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


def get_unhedged():
    connection = psycopg2.connect(user=os.getenv("POSTGRES_USERNAME"),
                                  password=os.getenv("POSTGRES_PASSWORD"),
                                  host="pgbouncer",
                                  port=6432)

    cursor = connection.cursor()
    cursor.execute("SELECT id, timestamp "
                   "FROM strategy_mark_to_market_unhedged_amount_events "
                   "WHERE price is NULL ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    cursor.close()

    for row in rows:
        row_time = row[1]
        # time_from = int(datetime.datetime.strptime(row_time, "%Y-%m-%d %H:%M:%S.%f").timestamp())
        time_from = int(row_time.timestamp())
        time_to = time_from + 60
        auth = HTTPBasicAuth(os.getenv("BKEY"), os.getenv("BSECRET"))
        data = {
            "symbol": "XBTUSD",
            "binSize": "1m",
            "startTime": datetime.datetime.fromtimestamp(time_from).isoformat(),
            "endTime": datetime.datetime.fromtimestamp(time_to).isoformat()
        }
        data_parsed = urllib.parse.urlencode(data)
        url = f"https://www.bitmex.com/api/v1/quote/bucketed?{data_parsed}"

        response = requests.get(url=url, auth=auth)
        print(response.text)
        data = json.loads(response.text)
        price = data[0]["askPrice"]
        sql = """ UPDATE strategy_mark_to_market_unhedged_amount_events SET price = %s WHERE id = %s"""

        cursor = connection.cursor()
        cursor.execute(sql, (price, row[0]))
        connection.commit()
        cursor.close()
        time.sleep(3)
    return


if __name__ == '__main__':
    get_unhedged()
