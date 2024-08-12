import modal
from modal import Image
image = Image.debian_slim().pip_install("langchain", 
                                        "langchain_openai", 
                                        "pymongo", 
                                        "google_auth_oauthlib",
                                        "google-ads"
                                        )
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.pydantic_v1 import BaseModel, Field, validator

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

from google.ads.googleads.client import GoogleAdsClient
from google.ads.googleads.errors import GoogleAdsException
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from urllib.parse import quote_plus
from typing import List, Optional

import requests
import json
import os
import time

app = modal.App(
    name="JudgeAgent", 
    image=image, 
    secrets=[modal.Secret.from_name("JUDGEKEYS")]
)

class SearchInput(BaseModel):
    cmd: str = Field(description="should be a search query")

class KeywordPlannerInput(BaseModel):
    keywords: List[str] = Field(
        ...,
        description="List of seed keywords to get ideas for"
    )

    @validator('keywords')
    def check_keywords_not_empty(cls, v):
        if not v or len(v) < 1:
            raise ValueError('keywords must contain at least one item')
        return v

    class Config:
        extra = 'forbid'
        schema_extra = {
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                }
            }
        }

class AdText(BaseModel):
    headlines: List[str]
    descriptions: List[str]

class MarketingResult(BaseModel):
    list_of_keywords: List[str] = Field(
        description="A list of keywords generated for marketing purposes. Each entry is a keyword string."
    )
    list_of_ad_text: AdText = Field(
        description="An AdText object containing advertisement text variations generated to attract the target audience."
    )
    list_of_paths_taken: List[str] = Field(
        description="A list of paths taken during the marketing research. Each path describes the approach, thought process, and specific actions taken, including the content types explored."
    )
    business: str = Field(
        description="A description of the business for which the marketing materials are being generated."
    )
    user_persona: str = Field(
        description="A description of the target user persona for the marketing campaign."
    )

class KeywordIdeasInput(BaseModel):
    seed_keywords: str = Field(..., description="Comma-separated list of seed keywords")
    language: str = Field(default="en", description="Language code (default is 'en' for English)")
    location_ids: Optional[List[str]] = Field(default=None, description="List of location IDs (default is United States)")

def load_prompt() -> str:
    """
    Fetches the content of a prompt file from a given URL.

    Returns:
    - str: The content of the prompt file as a string.

    Raises:
    - Exception: If the request to fetch the prompt fails.
    """
    url = "https://raw.githubusercontent.com/EmmS21/Prompts/main/JudgeAgent.md"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to load prompt. Status code: {response.status_code}") 

JUDGE_PROMPT = load_prompt()

def map_locations_ids_to_resource_names(client, location_ids):
    """
    Maps location IDs to their corresponding resource names using Google Ads API.

    Args:
    - client: The GoogleAdsClient object.
    - location_ids (list): A list of location IDs to be mapped.

    Returns:
    - list: A list of resource names corresponding to the location IDs.
    """
    build_resource_name = client.get_service("GeoTargetConstantService").geo_target_constant_path
    return [build_resource_name(location_id) for location_id in location_ids]

@app.function(secrets=[modal.Secret.from_name("fractal-demo")])
def authentication_google():
    """
    Handles the authentication process for Google Ads API using OAuth2.

    This function supports both refreshing an existing token and performing
    a new OAuth2 Device Flow if a refresh token is not available.

    Returns:
    - Credentials: Google OAuth2 credentials for accessing Google Ads API.

    Raises:
    - Exception: If an error occurs during the OAuth2 Device Flow or token refresh.
    """
    client_config = {"client_id": os.environ["client_id"].strip('"'),
              "client_secret": os.environ["client_secret"].strip('"'),
              "auth_uri": os.environ["auth_uri"].strip('"'),
              "token_uri": os.environ["token_uri"].strip('"'),
              "developer_token": os.environ["developer_token"].strip('"'),
              "use_proto_plus": os.environ["use_proto_plus"].strip('"'),
              "location_ids":  os.environ["location_ids"].strip('"'),
              "language_id": os.environ["language_id"].strip('"'),
              "customer_id": os.environ["customer_id"].strip('"')
              }
    refresh_token = os.getenv("refresh_token")
    if refresh_token:
        credentials = Credentials.from_authorized_user_info(
            {
                "client_id": client_config["client_id"],
                "client_secret": client_config["client_secret"],
                "refresh_token": refresh_token
            },
            scopes=['https://www.googleapis.com/auth/adwords']
        )
        return credentials
    if not refresh_token:
        flow = Flow.from_client_config(
            {"installed": client_config},
            scopes=['https://www.googleapis.com/auth/adwords'],
        )
        device_flow_info = flow.authorization_url()
        verification_url = device_flow_info["verification_url"]
        user_code = device_flow_info["user_code"]
        interval = device_flow_info["interval"]
        print(f"Please visit {verification_url} and enter the code: {user_code}")
        while True:
            try:
                flow.fetch_token()
                credentials = flow.credentials
                refresh_token = credentials.refresh_token
                modal.Secret.from_dict({"refresh_token": refresh_token}).persist("fractal-demo")
                break
            except Exception:
                print("Waiting for user to complete authorization...")
                time.sleep(interval)
    else:
        flow = Flow.from_client_config(
            {"installed": client_config},
            scopes=['https://www.googleapis.com/auth/adwords'],
        )
        credentials = flow.credentials
        credentials.refresh_token = refresh_token
    return credentials

@tool("generate_keywords-tool", args_schema=KeywordPlannerInput)
def generate_keyword_ideas(keywords):
    """
    Generate keyword ideas using Google Ads Keyword Planner API.

    This function authenticates with Google Ads, sends a request to the Keyword Planner,
    and returns keyword ideas based on the input keywords, language, and location.

    Args:
        keywords (list): A list of seed keywords to generate ideas from.

    Returns:
        dict: A dictionary containing:
            - 'keyword_ideas': A list of dictionaries, each containing:
                - 'text': The keyword text.
                - 'avg_monthly_searches': Average monthly searches for the keyword.
                - 'competition': Competition level for the keyword.
        If an error occurs, the dictionary will contain an 'error' key with details.
    """
    try:
        print("Starting authentication...")
        authentication = authentication_google.remote()

        client = GoogleAdsClient(
            credentials=authentication,  
            developer_token="KqvZ_dgzuH6zm8foV-7Krw",
            use_proto_plus=True
        )
        keyword_plan_idea_service = client.get_service("KeywordPlanIdeaService")
        keyword_plan_network = client.enums.KeywordPlanNetworkEnum.GOOGLE_SEARCH_AND_PARTNERS
        location_ids = [os.environ["location_ids"].strip('"')]  
        language_id = os.environ["language_id"].strip('"')
        location_rns = map_locations_ids_to_resource_names(client, location_ids)
        language_rn = client.get_service("GoogleAdsService").language_constant_path(language_id)
        request = client.get_type("GenerateKeywordIdeasRequest")
        request.customer_id = os.environ["customer_id"].strip('"')
        request.language = language_rn
        request.geo_target_constants = location_rns
        request.include_adult_keywords = False
        request.keyword_plan_network = keyword_plan_network
        request.keyword_seed.keywords.extend(keywords)
        keyword_ideas = keyword_plan_idea_service.generate_keyword_ideas(request=request)
        results = []
        for idea in keyword_ideas:
            competition_value = idea.keyword_idea_metrics.competition.name
            avg_monthly_searches = idea.keyword_idea_metrics.avg_monthly_searches
            if (competition_value != "HIGH" and avg_monthly_searches >= 1000):
                results.append({
                    "text": idea.text,
                    "avg_monthly_searches": avg_monthly_searches,
                    "competition": competition_value
                })
        return {"keyword_ideas": results}
    except GoogleAdsException as ex:
        error_messages = [f"Google Ads API Error: {error.message}" for error in ex.failure.errors]
        print(f"Google Ads API Error: {error_messages}")
        return {"error": "Google Ads API Error", "details": error_messages}
    except Exception as e:
        print(f"An unexpected error occurred while interacting with the Google Ads API: {str(e)}")
        return {"error": "Unexpected error during Google Ads API interaction", "details": str(e)}

@tool("validate_ad_length-tool")
def validate_ad_length(data):
    """
    Validate headlines and descriptions based on their character length.
    Returns information about which headlines or descriptions are too long.

    Args:
    - data (dict): Dictionary containing a list of headlines and descriptions.

    Returns:
    - dict: A dictionary with keys "status" and "details".
         - "status" is "success" if all headlines and descriptions are within limits,
            "too long (headline)" if any headline exceeds the limit,
            "too long (description)" if any description exceeds the limit,
            "too long (headline/description)" if both exceed their limits.
        - "details" contains only the headlines or descriptions that are too long.
    """
    try:
        items = []
        for ad in data['ads_data']:
            for headline, description in zip(ad['ad']['headlines'], ad['ad']['descriptions']):
                items.append({"headline": headline, "description": description})
                
        long_headlines = []
        long_descriptions = []
                
        for item in items:
            if len(item["headline"]) > 30:
                long_headlines.append(item["headline"])
            if len(item["description"]) >= 90:
                long_descriptions.append(item["description"])
                
        if long_headlines and long_descriptions:
            status = "too long (headline/description)"
            details = long_headlines + long_descriptions
        elif long_headlines:
            status = "too long (headline)"
            details = long_headlines
        elif long_descriptions:
            status = "too long (description)"
            details = long_descriptions
        else:
            status = "success"
            details = []
                
        return {
            "status": status,
            "details": details
        }
    except Exception as e:
        return {
            "status": f"Error: {str(e)}",
            "details": []
        }

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
    print(f"mongodb+srv://{encoded_username}:{encoded_password}@{host}")
    return f"mongodb+srv://{encoded_username}:{encoded_password}@{host}"


@app.function(timeout=3600)
@modal.web_endpoint(method="GET")
def run():
    """
    Main function to execute the AI agent workflow. 

    It connects to MongoDB, retrieves marketing data for a business, 
    and processes it using the AI agent.

    Returns:
    - str: Final output of the AI agent after processing the input data.
    """
    mongo_string = os.environ["MONGO"]
    connection_string = encode_credentials(mongo_string)
    connection_string += "&tls=true&tlsVersion=TLS1_2"
    marketing_agent_output = fetch_data("fractaltech", connection_string)

    extracted_data = {
        "list_of_keywords": marketing_agent_output[0].get('list_of_keywords', []),
        "list_of_ad_text": marketing_agent_output[0].get('list_of_ad_text', {}),
        "business": marketing_agent_output[0].get('business', ''),
        "user_persona": marketing_agent_output[0].get('user_persona', '')
    }
    openai_llm = ChatOpenAI(model="gpt-4o",
                            openai_api_key=os.environ["OPENAI_API_KEY"],
                            temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", JUDGE_PROMPT),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    tools = [generate_keyword_ideas, validate_ad_length]
    agent =  create_openai_tools_agent(openai_llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5000,
        max_execution_time=1200,
        return_intermediate_steps=True
    )
    try:
        input_str = json.dumps(extracted_data)
        input_data = {
            "input": input_str,
            "chat_history": []
        }
        print("Starting AgentExecutor invocation...")
        result = agent_executor.invoke(input_data)
        print("AgentExecutor invocation completed.")
        if result and "output" in result:
            final_output = result["output"]
            return final_output
    except Exception as e:
        return e
