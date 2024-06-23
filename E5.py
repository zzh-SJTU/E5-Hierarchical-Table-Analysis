from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
import argparse
from langchain.agents import load_tools
from typing import Any, List, Mapping, Optional
import requests
import json
import os
import langchain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
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
import openai
from langchain.llms import AzureOpenAI
from tools import eval_ex_match

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = "2023-05-15"
openai.api_key = os.getenv("AZURE_OPENAI_KEY")

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-05-15"
os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_KEY")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--option",
    default="code",
    choices=["zero-shot", "code"],
    type=str,
)
parser.add_argument(
    "--if_debug",
    default=False,
    type=str,
)
parser.add_argument(
    "--temperature",
    default=0.7,
    type=float,
)
parser.add_argument(
    "--iter",
    default=10,
    type=int,
)
parser.add_argument(
    "--max_token",
    default=2048,
    type=float,
)
args = parser.parse_args()


def generate_html_table(json_data):
    title = json_data["title"]
    top_root = json_data["top_root"]
    left_root = json_data["left_root"]
    texts = json_data["texts"]
    merged_regions = json_data["merged_regions"]

    html_table = f"<h3>{title}</h3>"
    html_table += '<table border="1">'
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
output_file_name = args.option + "_prediction-gpt-4-32k-" + str(args.iter) + ".jsonl"

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
)
tools = load_tools(["python_repl"], llm=llm)
tools = [
    Tool(
        name="python_repl",
        func=tools[0].run,
        description="useful for running generated python code",
    )
]
template = """Answer the following questions based on the html table. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Table Structure: you should describe the table in detail including different levels of headers and their meanings. In the end, you should clearly specify which columns AND rows and their corresponding levels are related to the question.
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (python code), you should first use pandas to first create a dataframe according to the Related Headers and write code to accomplish the goal. To accomplish that, you should not load the raw html data. Instead, you should use df = pd.DataFrame(dict) as the following:
import pandas as pd
df = pd.DataFrame({{ column1 }}: [], {{ column2 }}: [] ) # dict's keys are related column names and values are cell values. Note that you should not load the entire table (such as pd.read_html) and only load the related part.
... # you may need to Filter out some columns (such as "total") based semantic relationship. For example: df = df[df["column"] != "total"]
print(...) # Finally, you MUST explicitly print the final result. For example: print(df)
Observation: print output of the python code
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (should be a list of comma separated values, eg: `foo, bar, baz`. If the answer only contains one element, the list should only contains one element. Please not that if you want to exact information from the table, make sure it is the exact the same content including the order.)

Begin! (Remember to explicitly print something in your genereated code)

Table: {table}
Question: {question}
{agent_scratchpad}"""


class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["table", "question", "intermediate_steps"],
)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={
                    "output": llm_output.split("Final Answer:")[-1]
                    .strip()
                    .split("\n")[0]
                    .replace("<|im_end|>", "")
                },
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )


output_parser = CustomOutputParser()
output_parser_2 = CommaSeparatedListOutputParser()

llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, max_iterations=10
)

if os.path.exists(output_file_name):
    label_file_name = output_file_name
else:
    label_file_name = dataset_file

test_data_list = []
with open(label_file_name, "r") as file:
    for line in file:
        json_line = json.loads(line)
        test_data_list.append(json_line)

langchain.debug = args.if_debug

for i in range(len(test_data_list)):
    data = test_data_list[i]
    table_id = data["table_id"]
    file_name = "raw/" + str(table_id) + ".json"
    f = open(file_name, "r")
    content = f.read()
    json_table = json.loads(content)
    html_table = generate_html_table(json_table)
    if args.option + " prediction" in data.keys():
        print("skip")
        continue
    dic_output = {}
    data_id = data["id"]
    question = data["question"]
    dic_output["label"] = data["answer"]
    dic_output["id"] = data_id
    dic_output["table_id"] = table_id
    dic_output["question"] = question
    print(question)
    prediction = agent_executor.run(table=html_table, question=question)
    prediction = output_parser_2.parse(prediction)
    dic_output[args.option + " prediction"] = prediction
    test_data_list[i] = dic_output
    with open(output_file_name, "w") as f:
        for item in test_data_list:
            json_item = json.dumps(item)
            f.write(json_item + "\n")


corret = 0
total = 0
ambig = 0
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
print("ambig: " + str(ambig))
print("Total accuracy for " + args.option + " :" + str(corret / total))
