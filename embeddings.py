# coding=utf-8
import numpy as np
import pandas as pd

from evadb.catalog.catalog_type import NdArrayType
from evadb.functions.abstract.abstract_function import AbstractFunction
from evadb.functions.decorators.decorators import forward, setup
from evadb.functions.decorators.io_descriptors.data_types import PandasDataframe

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import ast

class Embeddings(AbstractFunction):
    @setup(cacheable=True, function_type="embeddings", batchable=False)
    def setup(self):
        self.client = OpenAI()
        self.prompt = ""

    @property
    def name(self) -> str:
        return "Embeddings"

    # Annotate input and output data types
    @forward(
        input_signatures=[
            PandasDataframe(
                columns=["prompt", "embeddings"],
                column_types=[NdArrayType.STR, NdArrayType.STR],
                column_shapes=[(1), (1)],
            )
        ],
        output_signatures=[
            PandasDataframe(
                columns=["distance"],
                column_types=[NdArrayType.FLOAT32],
                column_shapes=[(1)],
            )
        ],
    )
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        def _forward(row: pd.Series) -> np.ndarray:
            prompt = row.iloc[0]
            embeddings = row.iloc[1]

            parsed_embedding = ast.literal_eval(embeddings);

            if len(parsed_embedding) == 0:
              return -1.0

            reference_embedding = np.array(parsed_embedding)

            # Cache embeddings for prompts between calls to this function
            if prompt != self.prompt:
                embedding_response = self.client.embeddings.create(input = [prompt], model="text-embedding-ada-002")
                self.embeddings = embedding_response.data[0].embedding
                self.prompt = prompt

            document_embedding = np.array(self.embeddings)

            reference_embedding = reference_embedding.reshape(1, -1)
            document_embedding = document_embedding.reshape(1, -1)

            similarity_score = cosine_similarity(reference_embedding, document_embedding)

            return similarity_score[0][0]

        ret = pd.DataFrame()
        ret["distance"] = df.apply(_forward, axis=1)
        return ret