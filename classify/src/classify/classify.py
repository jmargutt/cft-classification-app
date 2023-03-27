import datetime
import json
import pandas as pd
import os
import traceback
import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.exceptions as exceptions
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
import click
from setfit import SetFitModel, SetFitTrainer, sample_dataset
import logging
logging.root.handlers = []
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG, filename='ex.log')
# set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
# set a format which is simpler for console use
formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
console.setFormatter(formatter)
logging.getLogger("").addHandler(console)
twilio_logger = logging.getLogger('twilio.http_client')
twilio_logger.setLevel(logging.WARNING)
# load_dotenv(dotenv_path="../credentials/.env")
load_dotenv(dotenv_path=r"C:\Users\JMargutti\OneDrive - Rode Kruis\Rode Kruis\digital-cea\few-shot-classification-app\credentials\.env")

MODELS = ['COVID-19 (English)']

def get_cosmos_db():
    settings = {
        'host': os.environ.get('ACCOUNT_HOST', 'https://emergencycosmos.documents.azure.com:443/'),
        'master_key': os.getenv('COSMOS_KEY'),
        'database_id': os.environ.get('COSMOS_DATABASE', 'ReliefApp'),
        'container_id': os.environ.get('COSMOS_CONTAINER', 'Beneficiaries'),
    }
    client = cosmos_client.CosmosClient(settings['host'],
                                        {'masterKey': settings['master_key']},
                                        user_agent="ReliefApp",
                                        user_agent_overwrite=True)
    return client.get_database_client(settings['database_id'])

cosmos_db = get_cosmos_db()


def get_local_data_path(user_email, ds_id):
    data_dir = 'instance'
    os.makedirs(data_dir, exist_ok=True)
    return f"{data_dir}/user_{user_email}_ds_{ds_id}_data.csv"


def get_feedback_data(user_email, ds_id, keep_id=False):
    cosmos_container = cosmos_db.get_container_client('Feedback')
    user_email = user_email + "                                           "
    item_list = query_items_by_partition_key(cosmos_container, user_email)
    item_list = filter_by_dataset(item_list, ds_id)
    df = pd.DataFrame(item_list)
    if df.empty:
        return None
    else:
        if not keep_id:
            df = df.drop(columns=['partitionKey', 'id', 'ds_id'], errors='ignore')
        df = df.drop(columns=[c for c in df.columns if c.startswith('_')])
        return df


def update_feedback_entry(feedback_id, user_email, replace_body):
    try:
        cosmos_container = cosmos_db.get_container_client('Feedback')
        user_email = user_email + "                                           "
        read_item = cosmos_container.read_item(item=feedback_id, partition_key=user_email)
        for key in replace_body.keys():
            read_item[key] = replace_body[key]
        cosmos_container.replace_item(item=read_item, body=read_item)
        return "success"
    except exceptions.CosmosResourceNotFoundError:
        return "not_found"


def query_items_by_partition_key(container, key):
    # Including the partition key value of account_number in the WHERE filter results in a more efficient query
    items = list(container.query_items(
        query="SELECT * FROM r WHERE r.partitionKey=@key",
        parameters=[
            {"name": "@key", "value": key}
        ]
    ))
    return items


def filter_by_dataset(item_list, ds_id):
    for item in item_list[:]:
        if 'ds_id' in item.keys():
            if str(item['ds_id']) != str(ds_id):
                item_list.remove(item)
    return item_list


def pandas_to_html(df, replace_values={}, replace_columns={}, titlecase=False):
    df = df.replace(replace_values)
    df = df.rename(columns=replace_columns)
    if titlecase:
        df.columns = [x.title() for x in df.columns]
    columns = df.columns
    rows = []
    for ix, row in df.iterrows():
        rows.append(row.to_dict())
    return columns, rows

########################################################################################################################

def get_blob_service_client(blob_path, container):
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION"))
    return blob_service_client.get_blob_client(container=container, blob=blob_path)


def get_container_service_client(container):
    blob_service_client = BlobServiceClient.from_connection_string(os.getenv("CONNECTION"))
    return blob_service_client.get_container_client(container=container)


def download_blob(container, blob_directory):
    container_client = get_container_service_client(container)
    blob_files = container_client.list_blobs()
    for blob in blob_files:
        if str(blob_directory) in blob.name and '.' in blob.name:
            blob_client = get_blob_service_client(blob.name, container)
            try:
                if not os.path.exists(blob.name):
                    os.makedirs(os.path.dirname(blob.name), exist_ok=True)
                    with open(blob.name, "wb") as download_file:
                        download_file.write(blob_client.download_blob().readall())
            except FileNotFoundError:
                continue


def inference(data, model_path, target_col, text_col="text"):

    # check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model {model_path} not found.")

    # load model and run inference
    model = SetFitModel.from_pretrained(model_path)
    texts_pred = data[text_col].to_list()
    predictions = model(texts_pred).numpy()

    # map to label names
    with open(f"{model_path}/label_dict.json", 'r') as openfile:
        label_dict = json.load(openfile)
    n = 0
    for ix, row in data.iterrows():
        data.at[ix, target_col] = label_dict[str(predictions[n])]
        n += 1

    return data


@click.command()
@click.option('--email', help='user email')
@click.option('--ds_id', help='dataset ID')
@click.option('--model', type=click.Choice(MODELS), help='model name')
@click.option('--overwrite', is_flag=True, default=False, help='overwrite pre-existing codes')
def main(email, ds_id, model, overwrite):

    data = get_feedback_data(user_email=email, ds_id=ds_id, keep_id=True)
    logging.info(f"got {len(data)} messages to analyse")

    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()
    directory = os.getenv('DIRECTORY')
    container = os.getenv('CONTAINER')

    logging.info('downloading model')
    model_path = os.path.join(directory, model).replace('\\', '/')
    try:
        download_blob(container, model_path)
    except:
        raise FileNotFoundError

    logging.info('start classifying type')
    # first type
    data = inference(data, os.path.join(model_path, 'classify_type'), target_col="type", text_col="feedback message")

    logging.info('start classifying category')
    data['type'] = data['type'].replace('', None)
    # then category
    for type in data['type'].dropna().unique():
        data_type = data[data['type'] == type]
        model_cat_path = os.path.join(model_path, 'classify_category', type).replace('\\', '/')
        if os.path.exists(model_cat_path):
            data_type = inference(data_type, model_cat_path, target_col="category", text_col="feedback message")
            data.at[data_type.index, 'category'] = data_type['category']
        else:
            data.at[data_type.index, 'category'] = None

    logging.info('start classifying code')
    data['category'] = data['category'].replace('', None)
    # then codes
    for category in data['category'].dropna().unique():
        data_category = data[data['category'] == category]
        model_code_path = os.path.join(model_path, 'classify_code', category).replace('\\', '/')
        if os.path.exists(model_code_path):
            data_category = inference(data_category, model_code_path, target_col="code", text_col="feedback message")
            data.at[data_category.index, 'code'] = data_category['code']
        else:
            data.at[data_category.index, 'code'] = None

    logging.info('fill in gaps')
    if data['code'].isnull().any():
        df_framework = pd.read_excel(os.path.join(model_path, 'framework.xlsx'), sheet_name='framework')
        df_framework = df_framework.fillna(method="ffill")
        df_framework = df_framework.replace({'\'': ''}, regex=True)
        for col in df_framework.columns:
            df_framework[col] = df_framework[col].astype(str)
            df_framework[col] = df_framework[col].str.normalize('NFC')

        for ix, row in data.iterrows():
            if row['category'] is None:
                category = df_framework[df_framework['type'] == row['type']]['category'].unique()[0]
                data.at[ix, 'category'] = category
            else:
                category = row['category']
            if row['code'] is None:
                data.at[ix, 'code'] = df_framework[df_framework['category'] == category]['code'].unique()[0]

    # data.to_csv('test.csv', index=False)
    logging.info('update data')
    for ix, row in data.iterrows():
        replace_body = {}
        for level in ['type', 'category', 'code']:
            if overwrite:
                replace_body[level] = row[level]
            else:
                if row[level] is not None:
                    replace_body[level] = row[level]
        result = update_feedback_entry(feedback_id=row['id'],
                                       user_email=email,
                                       replace_body=replace_body)
        if result != "success":
            logging.warning(f"message {row['id']} not found")

    logging.info('Python timer trigger function ran at %s', utc_timestamp)
    return data.to_json()


if __name__ == "__main__":
    main()

