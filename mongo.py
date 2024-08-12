from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

def connect_to_mongo(connection_string):
    """
    Establishes a connection to the MongoDB server.

    Args:
    - connection_string (str): The MongoDB connection string.

    Returns:
    - MongoClient: A MongoClient instance connected to the MongoDB server.
    """
    try:
        client = MongoClient(connection_string)
        # The ismaster command is cheap and does not require auth.
        client.admin.command('ismaster')
        print("MongoDB connection successful.")
        return client
    except ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
    except OperationFailure as e:
        print(f"Operation failed: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def authenticate_and_get_collection(client, db_name, collection_name):
    """
    Authenticates the MongoDB connection and retrieves the specified collection.

    Args:
    - client (MongoClient): The MongoClient instance.
    - db_name (str): The name of the database.
    - collection_name (str): The name of the collection.

    Returns:
    - Collection: The requested MongoDB collection, or None if authentication failed.
    """
    try:
        db = client[db_name]
        collection = db[collection_name]
        print(f"Successfully authenticated to the database '{db_name}' and accessed collection '{collection_name}'.")
        return collection
    except Exception as e:
        print(f"Error accessing collection: {e}")
        return None

def fetch_data_from_collection(collection, query={}, projection=None):
    """
    Fetches data from the MongoDB collection.

    Args:
    - collection (Collection): The MongoDB collection from which to fetch data.
    - query (dict): The query to filter documents.
    - projection (dict): The projection to specify which fields to include.

    Returns:
    - list: A list of documents retrieved from the collection.
    """
    try:
        results = collection.find(query, projection)
        documents = list(results)
        print(f"Fetched {len(documents)} documents from the collection.")
        return documents
    except Exception as e:
        print(f"Error fetching data: {e}")
        return []

# Main execution
if __name__ == "__main__":
    # Connection string
    connection_string = "mongodb+srv://EmmanuelSibanda:Kaleidoscope69@adalchemyai.q3tzkok.mongodb.net/?retryWrites=true&w=majority&appName=AdAlchemyAI"

    # Database and collection names
    db_name = "marketing_agent"
    collection_name = "fractaltech"

    # Connect to MongoDB
    client = connect_to_mongo(connection_string)

    if client is not None:
        # Authenticate and get the collection
        collection = authenticate_and_get_collection(client, db_name, collection_name)

        if collection is not None:
            # Fetch data from the collection
            data = fetch_data_from_collection(collection)

            # Process data as needed
            for doc in data:
                print(doc)
