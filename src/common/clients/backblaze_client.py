import base64
import json
import urllib.request as urllib2
from io import StringIO
import os
import requests
import hashlib
import pandas as pd
import time
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
import datetime
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


class BackblazeClient:
    def __init__(self):
        self.authorize_account_url = "https://api.backblazeb2.com/b2api/v2/b2_authorize_account"
        self.id_and_key = f"{os.getenv('BACKBLAZE_KEY_ID')}:{os.getenv('BACKBLAZE_KEY_SECRET')}"

        self.basic_auth_string = 'Basic ' + base64.b64encode(self.id_and_key.encode()).decode()
        self.headers = {'Authorization': self.basic_auth_string}
        self.account_id = None
        self.api_url = None
        self.account_authorization_token = None
        self.download_url = None
        self.upload_url = None
        self.upload_authorization_token = None
        self.bucket_name = "equinoxai-trades-db"

    def authorize(self):
        request = urllib2.Request(
            url=self.authorize_account_url,
            headers=self.headers
        )
        response = urllib2.urlopen(request)
        response_data = json.loads(response.read())
        response.close()
        self.account_id = response_data["accountId"]
        self.api_url = response_data["apiUrl"]
        self.account_authorization_token = response_data["authorizationToken"]
        self.download_url = response_data["downloadUrl"]

    def get_bucket_id(self, bucket_name):
        request = requests.post(
            url=f"{self.api_url}/b2api/v2/b2_list_buckets",
            data=json.dumps({"accountId": self.account_id}),
            headers={'Authorization': self.account_authorization_token}
        )

        response_data = json.loads(request.text)
        buckets = response_data["buckets"]
        bucket_id = None
        for bucket in buckets:
            if bucket["bucketName"] == "equinoxai-trades-db":
                self.bucket_name = bucket_name
                return bucket["bucketId"]
        return bucket_id

    def get_simulations(self, bucket_id):
        request = requests.post(
            url=f"{self.api_url}/b2api/v2/b2_list_file_names",
            data=json.dumps({"bucketId": bucket_id, "prefix": "simulations", "maxFileCount": 10000}),
            headers={'Authorization': self.account_authorization_token}
        )

        response_data = json.loads(request.text)
        files = response_data["files"]
        simulations = []
        for file in files:
            simulations.append([file['uploadTimestamp'], file["fileName"].split("/")[1]])
        return simulations

    def download_simulation(self, filename):
        request = requests.get(
            url=f"{self.download_url}/file/{self.bucket_name}/simulations/{filename}",
            data=None,
            headers={'Authorization': self.account_authorization_token}
        )
        response_data = base64.b64decode(request.text).decode()
        response = pd.read_csv(StringIO(response_data))
        return response

    def download_transfers(self, filename):
        request = requests.get(
            url=f"{self.download_url}/file/{self.bucket_name}/transfers/{filename}",
            data=None,
            headers={'Authorization': self.account_authorization_token}
        )
        response_data = base64.b64decode(request.text).decode()
        response = pd.read_csv(StringIO(response_data))
        return response

    def delete_simulation(self, filename, file_id):
        request = requests.post(
            url=f"{self.api_url}/b2api/v2/b2_delete_file_version",
            data=json.dumps({"fileName": filename, "fileId": file_id}),
            headers={'Authorization': self.account_authorization_token}
        )
        response_data = json.loads(request.text)
        return response_data

    def list_file_versions(self, start_file_id, start_file_name, prefix="simulations/"):

        bucket_id = self.get_bucket_id("equinoxai-trades-db")
        request = None
        if start_file_id:
            request = requests.post(
                url=f"{self.api_url}/b2api/v2/b2_list_file_versions",
                data=json.dumps(
                    {"bucketId": bucket_id, "maxFileCount": 10000, "prefix": prefix, "startFileId": start_file_id,
                     "startFileName": start_file_name}),
                headers={'Authorization': self.account_authorization_token}
            )
        else:
            request = requests.post(
                url=f"{self.api_url}/b2api/v2/b2_list_file_versions",
                data=json.dumps({"bucketId": bucket_id, "maxFileCount": 100, "prefix": prefix}),
                headers={'Authorization': self.account_authorization_token}
            )
        response_data = json.loads(request.text)
        return response_data["files"]

    def get_upload_url(self, bucket_id):
        request = requests.post(
            url=f"{self.api_url}/b2api/v2/b2_get_upload_url",
            data=json.dumps({"bucketId": bucket_id}),
            headers={'Authorization': self.account_authorization_token}
        )

        response_data = json.loads(request.text)
        self.upload_url = response_data["uploadUrl"]
        self.upload_authorization_token = response_data["authorizationToken"]
        return

    def upload_simulation(self, filename, df):
        print(f"Uploading file {filename} to backblaze")
        if type(df) != pd.DataFrame:
            try:
                df = pd.DataFrame(data=df)
            except:
                df = pd.DataFrame(data=df, index=[0])
        file_data = self.dataframe_to_file_data(dataframe=df)
        sha1_of_file_data = hashlib.sha1(file_data).hexdigest()
        if "+" in filename:  # if has month in id
            month = filename.split("+")[0]
            filename = f"{month}/{filename}"
        upload_headers = {
            'Authorization': self.upload_authorization_token,
            'X-Bz-File-Name': f"simulations/{filename}",
            'Content-Type': "text/csv",
            'X-Bz-Content-Sha1': sha1_of_file_data,
            'X-Bz-Info-Author': 'equinoxai',
            'X-Bz-Server-Side-Encryption': 'AES256'
        }
        request = urllib2.Request(self.upload_url, file_data, upload_headers)

        response = urllib2.urlopen(request)
        response_data = json.loads(response.read())
        response.close()
        return response_data

    def upload_tranfers(self, filename, df):
        print(f"Uploading file {filename} to backblaze")
        if type(df) != pd.DataFrame:
            try:
                df = pd.DataFrame(data=df)
            except:
                df = pd.DataFrame(data=df, index=[0])
        file_data = self.dataframe_to_file_data(dataframe=df)
        sha1_of_file_data = hashlib.sha1(file_data).hexdigest()
        if "+" in filename:  # if has month in id
            month = filename.split("+")[0]
            filename = f"{month}/{filename}"
        upload_headers = {
            'Authorization': self.upload_authorization_token,
            'X-Bz-File-Name': f"transfers/{filename}",
            'Content-Type': "text/csv",
            'X-Bz-Content-Sha1': sha1_of_file_data,
            'X-Bz-Info-Author': 'equinoxai',
            'X-Bz-Server-Side-Encryption': 'AES256'
        }
        request = urllib2.Request(self.upload_url, file_data, upload_headers)

        response = urllib2.urlopen(request)
        response_data = json.loads(response.read())
        response.close()
        return response_data

    def upload_parameters(self, filename, df):
        print(f"Uploading file {filename} to backblaze")
        print(df.describe())
        file_data = self.dataframe_to_file_data(dataframe=df)
        sha1_of_file_data = hashlib.sha1(file_data).hexdigest()
        upload_headers = {
            'Authorization': self.upload_authorization_token,
            'X-Bz-File-Name': f"parameter_optimization/{filename}",
            'Content-Type': "text/csv",
            'X-Bz-Content-Sha1': sha1_of_file_data,
            'X-Bz-Info-Author': 'equinoxai',
            'X-Bz-Server-Side-Encryption': 'AES256'
        }
        request = urllib2.Request(self.upload_url, file_data, upload_headers)

        response = urllib2.urlopen(request)
        response_data = json.loads(response.read())
        response.close()
        return response_data

    def download_parameters(self, filename):
        request = requests.get(
            url=f"{self.download_url}/file/{self.bucket_name}/parameter_optimization/{filename}",
            data=None,
            headers={'Authorization': self.account_authorization_token}
        )
        response_data = base64.b64decode(request.text).decode()
        response = pd.read_csv(StringIO(response_data))
        return response

    def dataframe_to_file_data(self, dataframe):
        return base64.b64encode(dataframe.to_csv(index=False).encode())

    # boto implementation

    # Return a boto3 client object for B2 service
    def get_b2_client(self):
        b2_client = boto3.client(service_name='s3',
                                 endpoint_url='https://s3.us-west-002.backblazeb2.com',  # Backblaze endpoint
                                 aws_access_key_id=os.getenv('BACKBLAZE_KEY_ID'),  # Backblaze keyID
                                 aws_secret_access_key=os.getenv('BACKBLAZE_KEY_SECRET'))  # Backblaze applicationKey
        return b2_client

    # Return a boto3 resource object for B2 service
    def get_b2_resource(self):
        b2 = boto3.resource(service_name='s3',
                            endpoint_url='https://s3.us-west-002.backblazeb2.com',  # Backblaze endpoint
                            aws_access_key_id=os.getenv('BACKBLAZE_KEY_ID'),  # Backblaze keyID
                            aws_secret_access_key=os.getenv('BACKBLAZE_KEY_SECRET'),  # Backblaze applicationKey
                            config=Config(
                                signature_version='s3v4',
                            ))
        return b2

    def get_simulations_by_name_and_date(self,
                                bucket,
                                b2,
                                prefix="",
                                time_from=datetime.datetime.strptime("01/01/22 09:55:00", '%d/%m/%y %H:%M:%S'),
                                time_to=datetime.datetime.strptime("31/12/22 10:40:00", '%d/%m/%y %H:%M:%S')):
        try:
            response = b2.Bucket(bucket).objects.filter(Prefix=f'simulations/{prefix}')
            files = []
            for file in response:  # iterate over response
                # compare dates
                if time_from.timestamp() < file.last_modified.timestamp() < time_to.timestamp():
                    files.append({"filename": file.key, "time": file.last_modified})
            cols = ["filename", "time"]
            return pd.DataFrame(columns=cols, data=files)  # return list of keys from response

        except ClientError as ce:
            print('error', ce)

    def get_tranfers_by_name_and_date(self,
                                      bucket,
                                      b2,
                                      prefix="",
                                      time_from=datetime.datetime.strptime("01/01/22 09:55:00", '%d/%m/%y %H:%M:%S'),
                                      time_to=datetime.datetime.strptime("31/12/22 10:40:00", '%d/%m/%y %H:%M:%S')):
        try:
            response = b2.Bucket(bucket).objects.filter(Prefix=f'transfers/{prefix}')
            files = []
            for file in response:  # iterate over response
                # compare dates
                if time_from.timestamp() < file.last_modified.timestamp() < time_to.timestamp():
                    files.append({"filename": file.key, "time": file.last_modified})
            cols = ["filename", "time"]
            return pd.DataFrame(columns=cols, data=files)  # return list of keys from response

        except ClientError as ce:
            print('error', ce)

    def get_simulations_by_id(self,
                              bucket,
                              b2,
                              id_="",
                              time_from=datetime.datetime.strptime("01/01/22 09:55:00", '%d/%m/%y %H:%M:%S'),
                              time_to=datetime.datetime.strptime("31/12/22 10:40:00", '%d/%m/%y %H:%M:%S')
                              ):
        try:
            if "+" in id_:  # if has month in id
                month = id_.split("+")[0]
                id_ = id_.replace("+", " ")
                response = b2.Bucket(bucket).objects.filter(Prefix=f'simulations/{month}/{id_}')
            else:
                response = b2.Bucket(bucket).objects.filter(Prefix=f'simulations/{id_}')

            files = []
            for file in response:  # iterate over response
                # compare dates
                if time_from.timestamp() < file.last_modified.timestamp() < time_to.timestamp():
                    files.append({"filename": file.key, "time": file.last_modified})
            cols = ["filename", "time"]
            return pd.DataFrame(columns=cols, data=files)  # return list of keys from response

        except ClientError as ce:
            print('error', ce)



    def delete_simulations_by_date_using_api(self,
                                             bucket,
                                             b2,
                                             time_from=datetime.datetime.strptime("1/1/22 00:00:00",
                                                                                  '%d/%m/%y %H:%M:%S'),
                                             time_to=datetime.datetime.strptime("1/7/23 00:00:00",
                                                                                '%d/%m/%y %H:%M:%S')):
        try:
            start_file_id = "4_z77cad1c597fab2c6722b0116_f11100ec04ddbd9e3_d20240610_m145528_c002_v0001121_t0021_u01718031328403"
            start_file_name = "simulations/.bzEmpty"
            # simulations/XMfSwVYvM8jJUkJWVi1o9CIq_Parameters_MT__from_12-03-2022_to_01-14-2023_id_XMfSwVYvM8jJUkJWVi1o9CIq 4_z77cad1c597fab2c6722b0116_f10190ccdb446acae_d20230304_m004900_c002_v0001127_t0034_u01677890940614
            # start_file_name = "simulations/.bzEmpty"
            files = self.list_file_versions(start_file_id=start_file_id, start_file_name=start_file_name, prefix="simulations/")
            count = 0
            deleted = 0
            while len(files) > 2:
                print(start_file_name, start_file_id)
                for file in files:  # iterate over response
                    count += 1
                    # compare dates  
                    print(file["fileName"], file["fileId"])
                    if time_from.timestamp() * 1000 < file["uploadTimestamp"] < time_to.timestamp() * 1000:
                            # and "ETHUSD" not in file["fileName"]:

                        self.delete_simulation(file["fileName"], file["fileId"])
                        time.sleep(0.5)
                        # same_files = self.list_file_versions(start_file_id=None, start_file_name=None,
                        #                                      prefix=file["fileName"])
                        # for same_file in same_files:
                        #     time.sleep(1)
                        #     self.delete_simulation(same_file["fileName"], same_file["fileId"])
                        #     deleted += 1
                        # time.sleep(1)
                        deleted += 1
                        if deleted % 100 == 0:
                            print(f"Break, deleted: {deleted}")
                            time.sleep(1)

                start_file_id = files[-1]["fileId"]
                start_file_name = files[-1]["fileName"]
                print(f"Total files parsed {count}, deleted {deleted}")
                time.sleep(2)
                files = self.list_file_versions(start_file_id=start_file_id, start_file_name=start_file_name, prefix="simulations/")

            return files

        except ClientError as ce:
            print('error', ce)


