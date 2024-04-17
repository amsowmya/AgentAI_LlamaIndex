from dotenv import load_dotenv
import os
import pandas as pd
# from llama_index.core.query_engine import PandasQueryEngine
from llama_index.experimental.query_engine import PandasQueryEngine
from prompts import new_prompt, instruction_str, context
from note_engine import note_engine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.azure_openai import AzureOpenAI
from pdf import canada_engine

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')

population_path = os.path.join("data", 'Population.csv')
population_df = pd.read_csv(population_path)

population_query_engine = PandasQueryEngine(
    df=population_df, verbose=True, instruction_str=instruction_str
    )

population_query_engine.update_prompts({"pandas_prompt": new_prompt})
# population_query_engine.query("What is the population of canada")

tools = [
    note_engine,
    QueryEngineTool(query_engine=population_query_engine, 
                    metadata=ToolMetadata(
                        name="population_data",
                        description="this gives information about the world population and demographics"
        ),
    ),
    QueryEngineTool(query_engine=canada_engine, 
                    metadata=ToolMetadata(
                        name="canada_Data",
                        description="this gives detailed information about the Canada country"
        ),
    ),
]

llm = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_deployment='atttestgpt35turbo',
    azure_endpoint='https://cb-att-openai-instance.openai.azure.com/',
    api_version='2024-02-15-preview'    
)

agent = ReActAgent.from_tools(tools=tools, llm=llm, verbose=True, context=context)

while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    result = agent.query(prompt)
    print(result)