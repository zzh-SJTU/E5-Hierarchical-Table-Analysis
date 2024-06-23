# $E^5$: Zero-shot Hierarchical Table Analysis using Augmented LLMs via Explain, Extract, Execute, Exhibit and Extrapolate

🔍 This is the official code for the NAACL 2024 paper titled ["$E^5$: Zero-shot Hierarchical Table Analysis using Augmented LLMs via Explain, Extract, Execute, Exhibit and Extrapolate"](https://aclanthology.org/2024.naacl-long.68/).

Authors: [Zhehao Zhang](https://zzh-sjtu.github.io/zhehaozhang.github.io/), Yan Gao, Jian-Guang Lou 👩‍💼👨‍💼

## 🌟 Abstract

<details><summary>Abstract</summary>

 Analyzing large hierarchical tables with multi-level headers presents challenges due to their complex structure, implicit semantics, and calculation relationships. While recent advancements in large language models (LLMs) have shown promise in flat table analysis, their application to hierarchical tables is constrained by the reliance on manually curated exemplars and the model's token capacity limitations. Addressing these challenges, we introduce a novel code-augmented LLM-based framework, $E^5$, for zero-shot hierarchical table question answering. This approach encompasses self-explaining the table's hierarchical structures, code generation to extract relevant information and apply operations, external code execution to prevent hallucinations, and leveraging LLMs' reasoning for final answer derivation. Empirical results indicate that our method, based on GPT-4, outperforms state-of-the-art fine-tuning methods with a 44.38 Exact Match improvement. Furthermore, we present \methodnameF, an adaptive algorithm designed for token-limited scenarios, effectively condensing large tables while maintaining useful information. Our experiments prove its efficiency, enabling the processing of large tables even with models having limited context lengths.

</details>

## Prerequisites
To run the code as is, you need to have access to Azure OpenAI. Set the following environment variables to configure the API access:

```bash
export AZURE_OPENAI_ENDPOINT=<Your Azure OpenAI Endpoint>
export AZURE_OPENAI_KEY=<Your Azure OpenAI Key>
```

These variables ensure that the Langchain library can properly authenticate with Azure's OpenAI service. Alternatively, if you wish to use a different LLM provider, adjust the `llm` variable in the code accordingly.

## File Structure
- **raw/**: Contains raw tables. *(Please refer to the previous work for the dataset link.)*
- **test_samples_clean.jsonl**: Stores cleaned Hitab [https://github.com/microsoft/HiTab]'s test data after removing incorrectly annotated data points.

## Running the Code
To execute the $E^5$ framework, run the following command:

```bash
python E5.py --option code
```

### Parameters
- `--option`: Chooses the experiment setting. Options are `zero-shot` for zero-shot prompting, and `code` for code-augmented LLM use. Default is `code`.
- `--if_debug`: Outputs the entire LLM response path in the terminal if set to `True`. Default is `False`.
- `--temperature`: Controls the generation temperature. Default is `0.7`.
- `--iter`: Specifies the number of iterations for execution. Default is `10`.
- `--max_token`: Sets the maximum token limit for the LLM. Default is `2048`.

## Configuration
Adjust the `max_iterations` parameter in the initialization of `AgentExecutor` to control the maximum number of API calls to the LLM.

## Utilities
- **tools.py**: Contains utility functions used across scripts.

## Implementing Baselines
To implement baseline methods, follow the structures provided in:
- **zero_shot.py**: For zero-shot prompting.
- **ReAct.py**: For ReAct prompting.

## Token-limited Scenario Strategies
For the Token-limited Scenario's $E^5$, we provide three strategies to reintegrate potentially relevant cells into the filtered table:
- **row_wise.py**: Example using GPT-3.5-turbo for row-wise processing with a token limit of 3000.
- **column_wise.py**: Example using GPT-4 for column-wise processing with a token limit of 2000.
- **spiral.py**: Example using GPT-4 for spiral processing with a token limit of 2000.

## Contact Information
For inquiries regarding the framework or the paper, please contact:

- Lead Author: Zhehao Zhang - [zhehao.zhang.gr@dartmouth.edu]

Feel free to reach out with questions, suggestions, or collaboration ideas.
## Citation
If you find our work useful or it contributes to your research, please consider citing our paper:

```bibtex
@inproceedings{zhang-etal-2024-e5,
    title = "$E^5$: Zero-shot Hierarchical Table Analysis using Augmented {LLM}s via Explain, Extract, Execute, Exhibit and Extrapolate",
    author = "Zhang, Zhehao  and
      Gao, Yan  and
      Lou, Jian-Guang",
    editor = "Duh, Kevin  and
      Gomez, Helena  and
      Bethard, Steven",
    booktitle = "Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.naacl-long.68",
    pages = "1244--1258",
    abstract = "Analyzing large hierarchical tables with multi-level headers presents challenges due to their complex structure, implicit semantics, and calculation relationships. While recent advancements in large language models (LLMs) have shown promise in flat table analysis, their application to hierarchical tables is constrained by the reliance on manually curated exemplars and the model{'}s token capacity limitations. Addressing these challenges, we introduce a novel code-augmented LLM-based framework, $E^5$, for zero-shot hierarchical table question answering. This approach encompasses self-explaining the table{'}s hierarchical structures, code generation to extract relevant information and apply operations, external code execution to prevent hallucinations, and leveraging LLMs{'} reasoning for final answer derivation. Empirical results indicate that our method, based on GPT-4, outperforms state-of-the-art fine-tuning methods with a 44.38 Exact Match improvement. Furthermore, we present $F^3$, an adaptive algorithm designed for token-limited scenarios, effectively condensing large tables while maintaining useful information. Our experiments prove its efficiency, enabling the processing of large tables even with models having limited context lengths. The code is available at https://github.com/zzh-SJTU/E5-Hierarchical-Table-Analysis.",
}
```