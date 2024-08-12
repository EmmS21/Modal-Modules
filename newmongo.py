from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from urllib.parse import quote_plus
import json

def connect_to_mongo_and_get_collection(connection_string, db_name, collection_name):
    """
    Establishes a connection to the MongoDB server and retrieves the specified collection.

    Args:
    - connection_string (str): The MongoDB connection string.
    - db_name (str): The name of the database.
    - collection_name (str): The name of the collection.

    Returns:
    - Collection: The requested MongoDB collection, or None if authentication failed.
    """
    try:
        client = MongoClient(connection_string)
        client.admin.command('ismaster') 
        print("MongoDB connection successful.")
        
        db = client[db_name]
        collection = db[collection_name]
        print(f"Successfully authenticated to the database '{db_name}' and accessed collection '{collection_name}'.")
        return collection
    except (ConnectionFailure, OperationFailure) as e:
        print(f"MongoDB connection or operation failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return None

def fetch_data(business_name, connection_string):
    """
    Fetches all data from the MongoDB collection.

    Args:
    - business_name (str): The name of the collection (business).
    - connection_string (str): The MongoDB connection string.

    Returns:
    - list: A list of documents retrieved from the collection.
    """
    collection = connect_to_mongo_and_get_collection(connection_string, "marketing_agent", business_name)
    if collection is None:
        return {"error": "Failed to access collection."}

    try:
        results = collection.find()
        documents = list(results)
        print(f"Fetched {len(documents)} documents from the collection.")
        return documents
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {"error": f"Error fetching data: {e}"}

def encode_credentials(connection_string: str):
    """Encodes the username and password in the connection string."""
    parts = connection_string.split('@')
    creds, host = parts[0], parts[1]
    username, password = creds.split('//')[1].split(':')
    encoded_username = quote_plus(username)
    encoded_password = quote_plus(password)
    return f"mongodb+srv://{encoded_username}:{encoded_password}@{host}"

def structure_data(data):
    item = data[0]
    structured_data = {
        "list_of_keywords": item['list_of_keywords'],
        "list_of_ad_text": item['list_of_ad_text'],
        "business": item['business'],
        "user_persona": item['user_persona']
    }
    json_string = json.dumps(structured_data, indent=2)
    
    return json_string
if __name__ == "__main__":
    connection_string = "mongodb+srv://EmmanuelSibanda:Kaleidoscope69@adalchemyai.q3tzkok.mongodb.net/?retryWrites=true&w=majority&appName=AdAlchemyAI"
    
    encoded_connection_string = encode_credentials(connection_string)
    data = fetch_data("fractaltech", encoded_connection_string)
    json_result = json.loads(structure_data(data))
    print(json_result)
