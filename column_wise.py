from typing import Any, List, Mapping, Optional
import requests
import json
import os
import langchain
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import tiktoken  
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
import re  
from bs4 import BeautifulSoup  
from tools import eval_ex_match, print_ambig
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
import argparse
from langchain.agents import load_tools

parser = argparse.ArgumentParser()
parser.add_argument(
    "--option",
    default="recognize",
    choices=["zero-shot", "code", "recognize"],
    type=str,
)
parser.add_argument(
    "--if_debug",
    default=False,
    type=str,
)
parser.add_argument(
    "--temperature",
    default=0.3,
    type=float,
)
parser.add_argument(
    "--max_token",
    default=1024,
    type=float,
)
parser.add_argument(
    "--token_limit",
    default=2000,
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
  
    html_table = f'<h3>{title}</h3>'  
    html_table += '<table border="1">'  
    for merged_region in merged_regions:  
        first_row, last_row = merged_region["first_row"], merged_region["last_row"]  
        first_column, last_column = merged_region["first_column"], merged_region["last_column"]  
        rowspan = last_row - first_row + 1  
        colspan = last_column - first_column + 1  
  
        if rowspan > 1:  
            texts[first_row][first_column] = f'<td rowspan="{rowspan}">{texts[first_row][first_column]}</td>'  
        if colspan > 1:  
            texts[first_row][first_column] = f'<td colspan="{colspan}">{texts[first_row][first_column]}</td>'  
  
    for row in texts:  
        html_table += '<tr>'  
        for cell in row:  
            if cell.startswith('<td'):   
                html_table += cell  
            elif cell != '':  
                html_table += f'<td>{cell}</td>'  
        html_table += '</tr>'  
  
    html_table += '</table>'  
    return html_table  
  
def shrink_html_table(html_table, top_header_rows_num, left_header_columns_num):  
    soup = BeautifulSoup(html_table, 'html.parser')  
    table = soup.find('table')  
    rows = table.find_all('tr')  
    for row in rows[top_header_rows_num:]:  
        cells = row.find_all(['td', 'th'])  
        special_row = any(cell.has_attr('colspan') for cell in cells)  
  
        if not special_row:  
            for cell in cells[left_header_columns_num:]:  
                cell.string = ''  
        else:  
            continue  
  
    for row in rows:  
        cells = row.find_all(['td', 'th'])  
  
        special_row = any(cell.has_attr('colspan') for cell in cells)  
  
        if not special_row:  
            for cell in cells[left_header_columns_num:]:  
                if row not in rows[:top_header_rows_num]:  
                    cell.decompose()  
  
    return str(table) 
  
def clean_string(s):  
    return re.sub(r'\W+', '', s).lower()  

def process_range(range_str):  
    start, end = range_str.split("-")  
    return list(range(int(start), int(end) + 1))

def filter_html_table(html_table, filter_dict, top_header_rows_num=0, left_header_columns_num=0, index_base=1):  
    soup = BeautifulSoup(html_table, 'html.parser')  
    table = soup.table  
  
    from difflib import SequenceMatcher  
  
    def find_nearby_name(row_col, names, target_index, distance=5, similarity_threshold=0.8):  
        def is_similar(a, b):  
            return SequenceMatcher(None, a, b).ratio() >= similarity_threshold  
    
        nearest_index = target_index  
        min_distance = float('inf')  
        for i in range(target_index - distance, target_index + distance + 1):  
            if i >= 0 and i < len(row_col) and any(is_similar(clean_string(row_col[i].text), name) for name in names):  
                current_distance = abs(target_index - i)  
                if current_distance < min_distance:  
                    min_distance = current_distance  
                    nearest_index = i  
        return nearest_index  


  
    row_indexes = [i + top_header_rows_num - index_base for i in filter_dict['Row indexes']]  
    col_indexes = [i + left_header_columns_num - index_base for i in filter_dict['Column indexes']]  
    row_names = [clean_string(name) for name in filter_dict['Row Names']]  
    col_names = [clean_string(name) for name in filter_dict['Column Names']]  
  
    rows = table.find_all('tr')  
    header_cells = rows[top_header_rows_num - 1].find_all(['td', 'th'])  
  
    for i, col_index in enumerate(col_indexes):  
        if col_index < len(header_cells) and clean_string(header_cells[col_index].text) != col_names[i]:  
            col_indexes[i] = find_nearby_name(header_cells, col_names, col_index)  
  
    adjusted_row_indexes = [row_index - 1 for row_index in row_indexes]  
  
    for k, (row_index, row_name) in enumerate(zip(row_indexes, row_names)):  
        
        if row_index >= len(rows):  
            row_index = len(rows) - 1  
  
        if clean_string(rows[row_index].find('td').text) == row_name:  
            adjusted_row_indexes[k] = row_index  
        else:  
            adjusted_row_indexes[k] = find_nearby_name([row.find('td') for row in rows], [row_name], row_index)  
  
    for i, row in enumerate(rows):  
        cells = row.find_all(['td', 'th'])  
        row.clear()  
  
        if i == top_header_rows_num - 1:  
            for cell in cells:  
                row.append(cell)  
        else:  
            for j, cell in enumerate(cells):  
                if i < top_header_rows_num or j < left_header_columns_num:  
                    row.append(cell)  
                else:  
                    if i in adjusted_row_indexes and j in col_indexes:  
                        row.append(cell)  
                    else:  
                        new_cell = soup.new_tag('td')  
                        row.append(new_cell)  
  
    new_table = soup.new_tag('table')  
    for row in rows:  
        new_table.append(row)  
  
    return str(new_table)  

def num_tokens_from_string(string, model_type="gpt-4-32k"):  
    """Returns the number of tokens in a text string."""  
    encoding = tiktoken.encoding_for_model(model_type)  
    num_tokens = len(encoding.encode(string))  
    return num_tokens  
  
def find_non_empty_cell_positions(top_header_rows_num, left_header_columns_num, html_table, initial_prompt, max_tokens, original_html_table, strategy='nearby_column'):  
    # near column strategy
    max_tokens = max_tokens - args.max_token
    soup = BeautifulSoup(html_table, 'html.parser')  
    original_soup = BeautifulSoup(original_html_table, 'html.parser')  
    table = soup.table  
    original_table = original_soup.table  
      
    current_tokens = num_tokens_from_string(initial_prompt + str(table))  
      
    if current_tokens > max_tokens:  
        return html_table  
  
    positions = []  
      
    total_rows = len(table.find_all('tr')) - top_header_rows_num  
    total_columns = len(table.find_all('tr')[top_header_rows_num].find_all('td')) - left_header_columns_num  
  
    position_range = (total_rows, total_columns)  
      
    for row_index, row in enumerate(table.find_all('tr')[top_header_rows_num:]):  
        for col_index, cell in enumerate(row.find_all('td')[left_header_columns_num:]):  
            if cell.text.strip():  
                positions.append((row_index, col_index))  
  
    matrix = [[0 if (i, j) in positions else None for j in range(total_columns)] for i in range(total_rows)]  
  
    def update_matrix_and_tokens(row, col):  
        nonlocal current_tokens  
        original_rows = original_table.find_all('tr')  
        if row + top_header_rows_num >= len(original_rows):  
            return False  
        original_cells = original_rows[row + top_header_rows_num].find_all('td')  
        if col + left_header_columns_num >= len(original_cells):  
            return False  
        original_cell = original_cells[col + left_header_columns_num]  
        cell_tokens = num_tokens_from_string(original_cell.text.strip())  
        if current_tokens + cell_tokens <= max_tokens:  
            matrix[row][col] = 1  
            current_tokens += cell_tokens  
            return True  
        return False  
    
    for row, col in positions:  
        for row_index in range(total_rows):  
            if row_index < len(matrix):
                if col < len(matrix[row_index]):
                    if matrix[row_index][col] is None:  
                        update_matrix_and_tokens(row_index, col)  
                  
    for row, col in positions:  
        for col_index in range(total_columns):  
            if row < len(matrix):
                if col_index < len(matrix[row]):
                    if matrix[row][col_index] is None:  
                        update_matrix_and_tokens(row, col_index)  
  
    current_positions = [(i, j) for i in range(total_rows) for j in range(total_columns) if matrix[i][j] == 1]  

    if current_tokens >= max_tokens:  
        for row_index, row in enumerate(table.find_all('tr')[top_header_rows_num:]):  
            for col_index, cell in enumerate(row.find_all('td')[left_header_columns_num:]):  
                if row_index < len(matrix):
                    if col_index < len(matrix[row_index]):
                        if matrix[row_index][col_index] == 1 and not cell.text.strip():  
                            original_cell = original_table.find_all('tr')[row_index + top_header_rows_num].find_all('td')[col_index + left_header_columns_num]  
                            cell.string = original_cell.text.strip()  
        return str(table)  

    while True:  
        updated = False  
        for row, col in current_positions:  
            if strategy == 'nearby_column':   
                if col - 1 >= 0:  
                    for row_index in range(total_rows):  
                        if matrix[row_index][col - 1] is None:  
                            if update_matrix_and_tokens(row_index, col - 1):  
                                updated = True  
                if col + 1 < total_columns:  
                    for row_index in range(total_rows):  
                        if matrix[row_index][col + 1] is None:  
                            if update_matrix_and_tokens(row_index, col + 1):  
                                updated = True  
        if not updated or all(val == 1 for row in matrix for val in row):  
            break  
          
        current_positions = [(i, j) for i in range(total_rows) for j in range(total_columns) if matrix[i][j] == 1]  
   
    for row_index, row in enumerate(table.find_all('tr')[top_header_rows_num:]):  
        for col_index, cell in enumerate(row.find_all('td')[left_header_columns_num:]): 
            if row_index < len(matrix):
                if col_index < len(matrix[row_index]):
                    if matrix[row_index][col_index] == 1 and not cell.text.strip():  
                        original_cell = original_table.find_all('tr')[row_index + top_header_rows_num].find_all('td')[col_index + left_header_columns_num]  
                        cell.string = original_cell.text.strip()  
                  
    return str(table)


dataset_file = "test_samples_clean.jsonl"
output_file_name = args.option + "_prediction-gpt-4-32k_column_" + str(args.token_limit) + "-1.jsonl"

if os.path.exists(output_file_name):
    label_file_name = output_file_name
else:
    label_file_name = dataset_file

test_data_list = [] 
with open(label_file_name, 'r') as file:  
    for line in file:  
        json_line = json.loads(line)  
        test_data_list.append(json_line)  

response_schemas = [
    ResponseSchema(name="Row indexes", description="A list of row indexes that are useful for answering the question (note that the minimal index is 1 and if it is a row span, please list every items)"),
    ResponseSchema(name="Row Names", description="A list of row names that are useful for answering the question (note that the row names are from the lowest level of the header and if there are commas in the name, please ignore them)"),
    ResponseSchema(name="Column indexes", description="A list of column indexes that are useful for answering the question (note that the minimal index is 1 and if it is a row span, please list every items)"),
    ResponseSchema(name="Column Names", description="A list of column names that are useful for answering the question (note that the column names are from the lowest level of the header and if there are commas in the name, please ignore them)")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = "Let's think step-by-step to find the related columns and rows. You MUST first generate reasons and in the end, output the final answer." + output_parser.get_format_instructions().replace("output","final answer") + "If you think there is no related information, try to use commonsense reasoning to mapping items in the question to the table and output the most related part with the best guess."  

llm = AzureOpenAI(
    model_name="gpt-4",
    engine = "gpt-4",
    temperature=args.temperature,
    max_tokens=args.max_token,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0
)
llm_2 = AzureOpenAI(
    model_name="gpt-4",
    engine = "gpt-4",
    temperature=args.temperature,
    max_tokens=256,
    top_p=0.95,
    frequency_penalty=0,
    presence_penalty=0
)
template = """Given the HTML table's headers and a question, please specify which columns and rows are useful for answering the question. Keep in mind that column indexes or row indexes are the order after the header, starting from 1. For example, if there is a three-column left header and the related column is right next to the header, the index is 1 (do not include the span of the header). 

Begin!
Table title: {title}
Table: {table}
Number of rows in the top header: {top_header_rows_num} (This is just for reference so you do not count this in the answer's index)
Number of columns in the left header: {left_header_columns_num} (This is just for reference so you do not count this in the answer's index)
Please note that besides these rows and columns, others are not headers even followed by colspan or rowspan.
Please also note that no matter how many rows/ columns the header covers, indexes are the orders AFTER the header.
Question: {question}
{format_instructions}
"""



tools = load_tools(["python_repl"], llm=llm)
tools = [
    Tool(
        name = "python_repl",
        func=tools[0].run,
        description="useful for running generated python code"
    )
]
template_tool = """Answer the following questions based on the html table. You have access to the following tools:

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
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template_tool,
    tools=tools,
    input_variables=["table", "question", "intermediate_steps"]
)

class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip().split("\n")[0].replace('<|im_end|>', '')},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise OutputParserException(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

output_parser_1 = CustomOutputParser()
output_parser_2 = CommaSeparatedListOutputParser()
llm_chain = LLMChain(llm=llm, prompt=prompt)

tool_names = [tool.name for tool in tools]
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser_1,
    stop=["\nObservation:"],
    allowed_tools=tool_names
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, max_iterations=10)
langchain.debug = args.if_debug
for i  in range(len(test_data_list)):
    data = test_data_list[i]
    if args.option + ' prediction' in data.keys() or args.option + ' process' in data.keys():
        print("skip")
        continue
    dic_output = {}
    data_id = data['id']
    table_id = data['table_id']
    question = data['question']
    dic_output['label'] = data['answer']
    dic_output['id'] = data_id
    dic_output['table_id'] = table_id
    dic_output['question'] = question
    file_name = "raw/" + str(table_id) + ".json"
    f = open(file_name, "r")
    content = f.read()
    json_table = json.loads(content)
    html_table = generate_html_table(json_table) 
    html_table_2 = shrink_html_table(html_table, json_table["top_header_rows_num"], json_table["left_header_columns_num"])
    if args.option == "zero-shot":
        prompt = PromptTemplate(template="Answer the question based on the following html of a hierarchical table \n{Table}\n{format_instructions}\n{question}",input_variables=["question", "Table"],partial_variables={"format_instructions": format_instructions})
        _input = prompt.format_prompt(question=question, Table = html_table)
        output = llm(_input.to_string())
        parsed_output = output_parser.parse(output)
        print(parsed_output)
        print("******************************************")
        print("Label: ")
        print(data['answer'])
        dic_output[args.option + ' prediction'] = parsed_output
        test_data_list[i] = dic_output
        with open(output_file_name, "w") as f:  
            for item in test_data_list:  
                json_item = json.dumps(item)  
                f.write(json_item + "\n") 
    elif args.option == "recognize":
        print(question)
        prompt = PromptTemplate(template=template,input_variables=["title", "question", "table", "top_header_rows_num", "left_header_columns_num"],partial_variables={"format_instructions": format_instructions})
        _input = prompt.format_prompt(title = json_table["title"], question=question, table = html_table_2, top_header_rows_num = json_table["top_header_rows_num"], left_header_columns_num = json_table["left_header_columns_num"])
        output = llm_2(_input.to_string())
        parsed_output = output_parser.parse(output)
        print(parsed_output)
        dic_filter = {}
        dic_output[args.option + ' process'] = output
        if "-" in parsed_output["Row indexes"]:  
            dic_output["Row indexes"] = process_range(parsed_output["Row indexes"])  
        elif len(parsed_output["Row indexes"]) == 0:
            dic_output["Row indexes"] = list(range(json_table["left_header_columns_num"]))
        else:  
            dic_output["Row indexes"] = [int(item.strip()) for item in parsed_output['Row indexes'].split(",")]  
        
        dic_output["Row Names"] = [item.strip() for item in parsed_output["Row Names"].split(",")]  
        
        if "-" in parsed_output["Column indexes"]:  
            dic_output["Column indexes"] = process_range(parsed_output["Column indexes"])  
        elif len(parsed_output["Column indexes"]) == 0:
            dic_output["Column indexes"] = list(range(json_table["top_header_rows_num"]))
        else:  
            dic_output["Column indexes"] = [int(item.strip()) for item in parsed_output["Column indexes"].split(",")]  
        dic_output["Column Names"] = [item.strip() for item in parsed_output["Column Names"].split(",")]  
        dic_filter["Row indexes"] = dic_output["Row indexes"]
        dic_filter["Row Names"] = dic_output["Row Names"]
        dic_filter["Column indexes"] = dic_output["Column indexes"]
        dic_filter["Column Names"] = dic_output["Column Names"] 

        filtered_table = filter_html_table(html_table, dic_filter, top_header_rows_num=json_table["top_header_rows_num"], left_header_columns_num=json_table["left_header_columns_num"])  
        
        initial_prompt = template_tool + question + json_table["title"] 
        filtered_table_2 = find_non_empty_cell_positions(top_header_rows_num=json_table["top_header_rows_num"], left_header_columns_num=json_table["left_header_columns_num"], html_table = filtered_table, initial_prompt = initial_prompt, max_tokens = args.token_limit, original_html_table = html_table)
        prediction = agent_executor.run(table = filtered_table_2, question = question)
        prediction = output_parser_2.parse(prediction)
        dic_output[args.option + ' prediction'] = prediction
        test_data_list[i] = dic_output
        with open(output_file_name, "w") as f:  
            for item in test_data_list:  
                json_item = json.dumps(item)  
                f.write(json_item + "\n") 

