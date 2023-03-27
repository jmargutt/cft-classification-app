from flask import session, current_app
from flask_login import current_user
import pandas as pd
import azure.cosmos.exceptions as exceptions
import os
import azure.cosmos.cosmos_client as cosmos_client
from dotenv import load_dotenv
load_dotenv()

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


def delete_feedback_data(user_email, ds_id):
    if os.getenv("MODE") == "online":
        cosmos_container = cosmos_db.get_container_client('Feedback')
        item_list = query_items_by_partition_key(cosmos_container, user_email)
        item_list = filter_by_dataset(item_list, ds_id)
        for item in item_list:
            cosmos_container.delete_item(item=item['id'], partition_key=user_email)
    elif os.getenv("MODE") == "offline":
        database = get_local_data_path(user_email, ds_id)
        if os.path.exists(database):
            os.remove(database)


def get_feedback_data(user_email, ds_id, keep_id=False):
    df = pd.DataFrame()

    if os.getenv("MODE") == "online":
        cosmos_container = cosmos_db.get_container_client('Feedback')
        item_list = query_items_by_partition_key(cosmos_container, user_email)
        item_list = filter_by_dataset(item_list, ds_id)
        df = pd.DataFrame(item_list)
    elif os.getenv("MODE") == "offline":
        database = get_local_data_path(user_email, ds_id)
        if os.path.exists(database):
            df = pd.read_csv(database, index_col=0, sep=';')
        else:
            return None
    if df.empty:
        return None
    else:
        if not keep_id:
            df = df.drop(columns=['partitionKey', 'id', 'ds_id'], errors='ignore')
        df = df.drop(columns=[c for c in df.columns if c.startswith('_')])
        return df


def get_feedback_entry(feedback_id, user_email, ds_id):
    if os.getenv("MODE") == "online":
        cosmos_container = cosmos_db.get_container_client('Feedback')
        try:
            data = cosmos_container.read_item(item=feedback_id,
                                              partition_key=user_email)
            for internal_field in [k for k in data.keys() if k.startswith('_')]:
                data.pop(internal_field)
            return data
        except exceptions.CosmosResourceNotFoundError:
            return "not_found"
    elif os.getenv("MODE") == "offline":
        database = get_local_data_path(user_email, ds_id)
        if os.path.exists(database):
            df = pd.read_csv(database, sep=';', dtype={'id': str}).set_index('id')
            if feedback_id in df.index:
                return df.loc[feedback_id].to_dict()
            else:
                return "not_found"
        else:
            return "no_data"


def update_feedback_entry(feedback_id, user_email, ds_id, replace_body):
    if os.getenv("MODE") == "online":
        try:
            cosmos_container = cosmos_db.get_container_client('Feedback')
            read_item = cosmos_container.read_item(item=feedback_id, partition_key=user_email)
            for key in replace_body.keys():
                read_item[key] = replace_body[key]
            cosmos_container.replace_item(item=read_item, body=read_item)
            return "success"
        except exceptions.CosmosResourceNotFoundError:
            return "not_found"
    elif os.getenv("MODE") == "offline":
        database = get_local_data_path(user_email, ds_id)
        if os.path.exists(database):
            df = pd.read_csv(database, sep=';', dtype={'id': str}).set_index('id')
            if feedback_id in df.index:
                for key in replace_body.keys():
                    df.at[feedback_id, key] = replace_body[key]
                df.to_csv(database, sep=';')
                return "success"
            else:
                return "not_found"
        else:
            return "no_data"


def save_feedback_data(data, ds_id, user_email):
    data_to_save = []
    for ix, row in data.iterrows():
        body = {
            'id': str(ds_id) + '-' + str(ix),
            'partitionKey': user_email,
            'ds_id': str(ds_id)
        }
        for key in row.keys():
            if key != 'id' and key != 'partitionKey' and key != 'ds_id':
                body[key] = str(row[key])
        data_to_save.append(body)

    if os.getenv("MODE") == "online":
        cosmos_container = cosmos_db.get_container_client('Feedback')
        # save to cosmos db
        for entry in data_to_save:
            cosmos_container.create_item(body=entry)
    elif os.getenv("MODE") == "offline":
        database = get_local_data_path(user_email, ds_id)
        df = pd.DataFrame.from_records(data_to_save)
        df.index = df['id']
        df = df.drop(columns=['id'])
        df.to_csv(database, sep=';')


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