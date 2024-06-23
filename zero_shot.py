from typing import Any, List, Mapping, Optional
import requests
import json
import os
import langchain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import atexit
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.chains import LLMChain
import time
from langchain.agents.agent import AgentExecutor
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.output_parsers import PydanticOutputParser
from langchain.agents.agent_toolkits import create_python_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from pydantic import BaseModel, Field, validator
import argparse
from typing import List
import openai
from langchain.llms import AzureOpenAI
from tools import eval_ex_match

parser = argparse.ArgumentParser()
parser.add_argument(
    "--option",
    default="zero-shot",
    choices=["zero-shot", "code"],
    type=str,
)
parser.add_argument(
    "--if_debug",
    default=True,
    type=str,
)
parser.add_argument(
    "--temperature",
    default=0.3,
    type=float,
)
parser.add_argument(
    "--max_token",
    default=64,
    type=float,
)
args = parser.parse_args()
openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")


def generate_html_table(json_data):
    title = json_data["title"]
    top_root = json_data["top_root"]
    left_root = json_data["left_root"]
    texts = json_data["texts"]
    merged_regions = json_data["merged_regions"]

    html_table = f"<h3>{title}</h3>"
    html_table += '<table border="1">'

    # Add merged cells information to texts using rowspan and colspan
    for merged_region in merged_regions:
        first_row, last_row = merged_region["first_row"], merged_region["last_row"]
        first_column, last_column = (
            merged_region["first_column"],
            merged_region["last_column"],
        )
        rowspan = last_row - first_row + 1
        colspan = last_column - first_column + 1

        if rowspan > 1:
            texts[first_row][
                first_column
            ] = f'<td rowspan="{rowspan}">{texts[first_row][first_column]}</td>'
        if colspan > 1:
            texts[first_row][
                first_column
            ] = f'<td colspan="{colspan}">{texts[first_row][first_column]}</td>'

    # Generate HTML table rows
    for row in texts:
        html_table += "<tr>"
        for cell in row:
            if cell.startswith("<td"):
                html_table += cell
            elif cell != "":
                html_table += f"<td>{cell}</td>"
        html_table += "</tr>"

    html_table += "</table>"
    return html_table


dataset_file = "test_samples_clean.jsonl"
output_file_name = args.option + "_prediction-gpt-4-32k.jsonl"

if os.path.exists(output_file_name):
    label_file_name = output_file_name
else:
    label_file_name = dataset_file

test_data_list = []
with open(label_file_name, "r") as file:
    for line in file:
        json_line = json.loads(line)
        test_data_list.append(json_line)

llm = AzureOpenAI(
    model_name="gpt-4-32k",
    engine="gpt-4-32k",
    temperature=args.temperature,
    max_tokens=args.max_token,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["Observation:"],
)
output_parser = CommaSeparatedListOutputParser()
format_instructions = (
    output_parser.get_format_instructions()
    + ". If the answer only contains one element, the list should only contains one element."
)

langchain.debug = args.if_debug
for i in range(len(test_data_list)):
    data = test_data_list[i]
    if args.option + " prediction" in data.keys():
        print("skip")
        continue
    dic_output = {}
    data_id = data["id"]
    table_id = data["table_id"]
    question = data["question"]
    dic_output["label"] = data["answer"]
    dic_output["id"] = data_id
    dic_output["table_id"] = table_id
    dic_output["question"] = question
    file_name = "raw/" + str(table_id) + ".json"
    f = open(file_name, "r")
    content = f.read()
    json_table = json.loads(content)
    html_table = generate_html_table(json_table)
    if args.option == "zero-shot":
        prompt = PromptTemplate(
            template="Answer the question based on the following html of a hierarchical table \n{Table}\n{format_instructions}\n{question}",
            input_variables=["question", "Table"],
            partial_variables={"format_instructions": format_instructions},
        )
        _input = prompt.format_prompt(question=question, Table=html_table)
        output = llm(_input.to_string())
        parsed_output = output_parser.parse(output)
        print(parsed_output)
        print("******************************************")
        print("Label: ")
        print(data["answer"])
        dic_output[args.option + " prediction"] = parsed_output
        test_data_list[i] = dic_output
        with open(output_file_name, "w") as f:
            for item in test_data_list:
                json_item = json.dumps(item)
                f.write(json_item + "\n")

corret = 0
total = 0
for data in test_data_list:
    if "label" in data.keys():
        total += 1
        if len(data["label"]) == len(data[args.option + " prediction"]):
            flag = True
            for i in range(len(data["label"])):
                if_match, pred, label = eval_ex_match(
                    str(data["label"][i]), data[args.option + " prediction"][i]
                )
                if if_match == False:
                    flag = False
            if flag == True:
                corret += 1

print("Total: " + str(total))
print("Correct: " + str(corret))
print("Total accuracy for " + args.option + " :" + str(corret / total))
