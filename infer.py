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
import argparse
from typing import List
from tools import eval_ex_match, print_ambig
parser = argparse.ArgumentParser()
parser.add_argument(
    "--option",
    default="code",
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
    default=256,
    type=float,
)
args = parser.parse_args()
class LLMClient:

    # _ENDPOINT = 'https://httpqas26-frontend-qasazap-prod-dsm02p.qas.binginternal.com/completions'
    _ENDPOINT = 'https://httpqas26-frontend-qas-sdf-mw1p.qas.binginternal.com/completions'
    _SCOPES = ['api://68df66a4-cad9-4bfd-872b-c6ddde00d6b2/access']

    def __init__(self):
        self._cache = SerializableTokenCache()
        atexit.register(
            lambda: open('.llmapi.bin', 'w').write(self._cache.serialize())
            if self._cache.has_state_changed else None)

        self._app = PublicClientApplication(
            '68df66a4-cad9-4bfd-872b-c6ddde00d6b2',
            authority=
            'https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47',
            token_cache=self._cache)
        if os.path.exists('.llmapi.bin'):
            self._cache.deserialize(open('.llmapi.bin', 'r').read())

    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()
        # populate the headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token,
            'X-ModelType': model_name
        }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT,
                                 data=body,
                                 headers=headers)
        return response.json()

    def send_stream_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer ' + token,
            'X-ModelType': model_name
        }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT,
                                 data=body,
                                 headers=headers,
                                 stream=True)
        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES,
                                                    account=chosen)

        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=LLMClient._SCOPES)

            if "user_code" not in flow:
                raise ValueError("Fail to create device flow. Err: %s" %
                                 json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]


llm_client = LLMClient()


class CustomLLM(LLM):
    max_token: int
    model_type: str
    temperature: float
    
    @property
    def _llm_type(self) -> str:
        return "LLMClient"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = ["Table:", "Question:", "Observation:"],
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        request_data = {
        "prompt": prompt,
        "max_tokens": self.max_token,
        "temperature": self.temperature,
        "top_p": 1,
        "stream": False,
        "stop": ""
    }
        response = llm_client.send_request(self.model_type, request_data)
        if self.model_type != 'dev-gpt-35-turbo' or self.model_type != 'text-chat-davinci-002':
            time.sleep(15)
        else:
            time.sleep(2)
        text_output = response['choices'][0]['text'].strip()
        min_pos = len(text_output)
        for target in ["Table:", "Question:", "Observation:"]:  
            pos = text_output.find(target)  
            if pos != -1 and pos < min_pos:  
                min_pos = pos
        text_output = text_output[:min_pos]
        return text_output

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"model_type": self.model_type, "max_token": self.max_token, "temperature": self.temperature}

def generate_html_table(json_data):  
    title = json_data["title"]  
    top_root = json_data["top_root"]  
    left_root = json_data["left_root"]  
    texts = json_data["texts"]  
    merged_regions = json_data["merged_regions"]  
  
    html_table = f'<h3>{title}</h3>'  
    html_table += '<table border="1">'  
  
    # Add merged cells information to texts using rowspan and colspan  
    for merged_region in merged_regions:  
        first_row, last_row = merged_region["first_row"], merged_region["last_row"]  
        first_column, last_column = merged_region["first_column"], merged_region["last_column"]  
        rowspan = last_row - first_row + 1  
        colspan = last_column - first_column + 1  
  
        if rowspan > 1:  
            texts[first_row][first_column] = f'<td rowspan="{rowspan}">{texts[first_row][first_column]}</td>'  
        if colspan > 1:  
            texts[first_row][first_column] = f'<td colspan="{colspan}">{texts[first_row][first_column]}</td>'  
  
    # Generate HTML table rows  
    for row in texts:  
        html_table += '<tr>'  
        for cell in row:  
            if cell.startswith('<td'):  # If the cell has rowspan or colspan  
                html_table += cell  
            elif cell != '':  
                html_table += f'<td>{cell}</td>'  
        html_table += '</tr>'  
  
    html_table += '</table>'  
    return html_table  

dataset_file = "test_samples.jsonl"
output_file_name = "code_prediction-gpt-4-32k.jsonl"

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
    ResponseSchema(name="answer", description="A list of answers to the user's question"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
llm = CustomLLM(max_token=args.max_token, model_type = "dev-gpt-35-turbo", temperature = args.temperature)
format_instructions = output_parser.get_format_instructions().replace("string", "List[string]")

langchain.debug = args.if_debug
for i  in range(len(test_data_list)):
    data = test_data_list[i]
    if args.option + ' prediction' in data.keys():
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
    elif args.option == "code":
        continue
        prompt = PromptTemplate(template="Table: \n{Table}\n\nQuestion:{question}",input_variables=["question", "Table"])
        prompt_agent = PromptTemplate(input_variables=[],template="Answer the question based on the following html of a hierarchical table using Python code (First use pandas to create a dataframe with relevant values)")
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt_agent
        )
        agent = ZeroShotAgent(llm_chain=llm_chain, allowed_tools= [PythonREPLTool().name])
        agent = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=[PythonREPLTool()])
        #agent = create_python_agent(llm,tool=PythonREPLTool(),verbose=True)
        _input = prompt.format_prompt(question=question, Table = html_table)
        print(_input)
        output = agent(_input.to_string())
        #parsed_output = output_parser.parse(output)
        print(parsed_output)
        print("******************************************")
        print("Label: ")
        print(data['answer'])

corret = 0
total = 0
ambig = 0
for data in test_data_list:
    if "label" in data.keys():
        total += 1
        if len(data["label"]) == 1:
            print(type(data["label"]))
            if_match, pred, label = eval_ex_match(str(data["label"][0]), data[args.option + ' prediction'])
        else:
            prediction_list = data[args.option + ' prediction'].split(',')  
            if len(prediction_list) == len(data["label"]):
                if_match = True
                for i in range(len(data["label"])):
                    if_match_item, pred, label = eval_ex_match(str(data["label"][i]), prediction_list[i])
                    if if_match_item == False:
                        if_match = False
            else:
                if_match = False
        if if_match == True:
            corret += 1
        else:
            flag = print_ambig(data, args.option)
            if flag == True:
                ambig += 1
print('Total: ' +  str(total))
print('Correct: '+  str(corret))
print("ambig: " + str(ambig))
print("Total accuracy for " + args.option + " :" + str(corret/total))

