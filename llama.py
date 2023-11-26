# coding=utf-8
import numpy as np
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

import huggingface_hub
from llama_cpp import Llama

# Model hosted on HuggingFace, original CodeLlama model converted to GGUF for use with llama.cpp
MODEL_NAME_OR_PATH = "TheBloke/CodeLlama-13B-GGUF"
MODEL_BASENAME = "codellama-13b.Q5_0.gguf"

# Template provided to Llama2
def template(prompt, context):
  return f"""
    SYSTEM: You are a helpful assistant that accomplishes user tasks.

    CONTEXT: {context}

    USER: {prompt}

    ASSISTANT: 
    """ 

class EvaLlama(AbstractFunction):
    @setup(cacheable=True, function_type="chat-completion", batchable=False)
    def setup(self):
        # Load the weights from hugging face
        self.model_path = huggingface_hub.hf_hub_download(repo_id=MODEL_NAME_OR_PATH, filename=MODEL_BASENAME)
        self.llm = Llama(
          model_path=self.model_path,
          n_ctx=2048
        )

    @property
    def name(self) -> str:
        return "EvaLlama"

    # Annotate input and output data types
    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["prompt", "text"],
                column_types=[NdArrayType.STR, NdArrayType.STR],
                column_shapes=[(1), (1)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["response"],
                column_types=[NdArrayType.STR],
                column_shapes=[(1)],
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        def _forward(row: pd.Series) -> np.ndarray:
            prompt = row.iloc[0]
            text = row.iloc[1]
            
            # Call the LLM
            response = self.llm(
              prompt=template(prompt, text),
              max_tokens=3000,
              temperature=0.5,
              top_p=0.95,
              repeat_penalty=1.2,
              top_k=150,
              echo=False   
            )

            return response["choices"][0]["text"]

        ret = pd.DataFrame()
        ret["response"] = df.apply(_forward, axis=1)
        return ret