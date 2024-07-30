from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from langchain.pydantic_v1 import BaseModel, Field
import os
import json
from typing import List, Optional
import requests
from modal import App, Image, Secret, web_endpoint, method
import modal

image = Image.debian_slim().pip_install("langchain", "langchain_openai")
app = App(
    name="marketing_agent",
    image=image,
    secrets=[Secret.from_name("MARKETINGAGENT")]
)

class Command(BaseModel):
    persona: str = Field(..., description="A description of the user persona.")
    business_name: str = Field(..., description="The name of the business.")

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

    memory = ConversationBufferMemory(return_messages=True)

def load_prompt() -> str:
    url = "https://raw.githubusercontent.com/EmmS21/Prompts/main/MarketingAgent.md"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        raise Exception(f"Failed to load prompt. Status code: {response.status_code}") 

OPENAI_PROMPT = load_prompt()

@tool
def multi_on_agent(cmd: str, restart_session: Optional[bool] = False) -> str:
    """
    Use this tool to interact with the MultiOn agent to browse and extract information from the web.

    Args:
        cmd: The command to be executed by MultiOn.
        restart_session: Whether to restart the session. Defaults to False.

    Returns:
        The response message from the MultiOn agent.
    """
    url = "https://emms21--multionagent-run.modal.run"
    headers = {"Content-Type": "application/json"}
    data = {"cmd": cmd, "restart_session": restart_session}
        
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status() 
            
        result = json.loads(response.text)
        return result
    except requests.RequestException as e:
        return f"An error occurred while calling the MultiOn agent: {str(e)}"

@app.function()
@modal.web_endpoint(method="POST")
def agent(command: Command):
    openai_llm = ChatOpenAI(model="gpt-4o", 
                    openai_api_key=os.environ["OPENAI_API_KEY"], 
                    temperature=0)

    shared_memory = ConversationBufferMemory(input_key="input", return_messages=True)
    input_data = {
        "input": f"Persona: {command.persona}, Business: {command.business_name}",
        "persona": command.persona,
        "business_name": command.business_name,
        "agent_scratchpad": []
    }
    try:
        parser = PydanticOutputParser(pydantic_object=MarketingResult)
        memory = shared_memory.load_memory_variables(input_data)
        input_data.update(memory)
        if isinstance(openai_llm, ChatOpenAI):
            prompt = ChatPromptTemplate.from_messages([
                ("system", OPENAI_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad")
            ])
            tools = [multi_on_agent]
            agent = create_openai_tools_agent(openai_llm, tools, prompt)
            agent_executor = AgentExecutor(
                agent=agent,
                tools=tools,
                verbose=True,
                handle_parsing_errors=True,
                max_terations=2000,
                max_execution_time=189,
                return_intermediate_steps=True
            )
            try:
                result = agent_executor.invoke(input_data)
                final_output = result['output']
                print('***', final_output)
                parsed_result = parser.parse(final_output)
                paths_taken = parsed_result.list_of_paths_taken
                shared_memory.save_context(
                    {"input": json.dumps(paths_taken)},
                    {"output": json.dumps(parsed_result.dict(), indent=2)}
                )
                return parsed_result.dict()
            except Exception as e:
                raise e
    except Exception as e:
        raise e


