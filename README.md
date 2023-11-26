# Repository Based Question Answering
https://github.com/aditsachde/EvaDB-Repository-Q-A-Part-2

### Background
The intent of this project was to build on features from Part 1, where the goal was to provide EvaDB with the ability to answer questions based on information from a Git repository. The reference repository for this project was AI For Beginners by Microsoft (https://github.com/microsoft/AI-For-Beginners), however, there are a number of similar repositories with interactive lessons. Being able to use these as references to answer questions would be very useful.

Part 2 expands on the features of Part 1 by incorporating the OpenAI embeddings API and providing the option to use a local Llama 2 LLM, leveraging the llama.cpp project. The embeddings API removes the need for the user to specify which file to search. The local Llama 2 LLM can significantly reduce costs at the expense of speed, which may be a useful tradeoff depending on the number of requests being served by EvaDB.

### Implementation
The implementation is broken down into a few pieces. The first piece is extending EvaDB to be able to load a repository. To do this, once the function has been imported, the user should run it with a cursor to an existing database and a repository URL:

```
load_repository(cursor, "https://github.com/microsoft/AI-For-Beginners.git")
```

The implementation does a few things. First, it clones the repository. Next, it goes through all the files, loading the text files into the database. As a part of this, it also extracts the text from the .ipynb files, as many repositories make their content interactive through python notebooks, but this format is not very conducive for LLMs. Finally, it calls the OpenAI embeddings API using the text-embedding-ada-002 model.


The next implementation piece is the Embeddings function, implemented as a custom EvaDB function. The entire repository can be quickly searched to narrow down which files are relevant to a specific prompt. 

```
SELECT name, text, Embeddings("What are the Principles of Responsible AI?", embeddings) FROM repository ORDER BY distance DESC LIMIT 5;
```

The top file returned by this function is `lessons/7-Ethics/README.md`, with a calculated embedding relevance of `0.890151`. From here, a user can lean on EvaDB’s built in ChatGPT support to ask a question with the repository as context, as seen in the following command:

```
SELECT ChatGPT('What are the Principles of Responsible AI?', s.text) FROM
(
SELECT Embeddings("What are the Principles of Responsible AI?", embeddings), name, text FROM repository ORDER BY distance DESC LIMIT 1
) AS s;
```

The corresponding output is:

“The Principles of Responsible AI are as follows: 1. Fairness: Ensuring that AI systems are free from biases and treat all individuals fairly and equally. 2. Reliability and Safety: Recognizing that AI models can make mistakes and taking precautions to prevent harm caused by incorrect advice or decisions. 3. Privacy and Security: Protecting the privacy of individuals and ensuring that data used for training AI models is handled securely. 4. Inclusiveness: Building AI systems that augment human capabilities and work towards inclusivity, considering the needs of underrepresented communities. 5. Transparency: Being clear about the use of AI systems and making them interpretable whenever possible. 6. Accountability: Identifying and understanding the responsibility of AI decisions and involving human beings in important decision-making processes.”

This makes it very easy to ask questions that have contextual information from a repository. This query leans heavily on the built in SQL features of EvaDB, using nested queries, order by clauses, and limit clauses. These features made the implementation significantly easier compared to any other option.

The final implementation piece is the EvaLlama function, also implemented as a custom EvaDB function. This works the same as the built in ChatGPT function.

```
SELECT EvaLlama('What are the five Principles of Responsible AI?', s.text) FROM
(
SELECT Embeddings("What are the five Principles of Responsible AI?", embeddings), name, text FROM repository ORDER BY distance DESC LIMIT 1
) AS s;
```

There are several benefits to this. First, the local LLM can have its token limits tweaked to support very long contexts. Additionally, it is free, compared to the cost of using the OpenAI ChatGPT API. However, the results are not as good as ChatGPT. The results might be able to be improved by tweaking the prompts or the models.

### Lessons and challenges
There were a couple main challenges to the implementation. The first challenge was generating and using the embeddings. The embeddings API from OpenAI has a token limit of around 8000 tokens. Although this is more than enough for most files and most lessons, this limit could be hit by very large code files. Additionally, searching requires using the cosine similarity functionality from scikit-learn. This works well, but it also depends on the content of the prompt. Some questions provide better sets of relevant documents compared to others. 

The embedding API is very cheap, costing less than a cent to generate embeddings for the entire reference repository for this project. OpenAI quotes the pricing as about 8000 pages for a dollar. Additionally, generating the embedding when searching for relevant documents has a negligible cost, and significantly improves the user experience when searching for a question.

The local Llama functionality was a lot harder to make work. GPU support to work on Colab was extremely flakey and did not end up working properly. CPU inference works but is a lot slower than GPU inference. Additionally, the prompts and models required a lot more tuning to get useful results. The Llama model that can be run locally is a lot smaller than OpenAI’s cloud hosted ChatGPT. There certainly is quite a bit of potential to use Llama 2 to lower costs and integrates very well into EvaDB due to its support for custom functions. The user experience for swapping between ChatGPT and EvaLlama extremely smooth, just renaming a single function. However, making it work well will require additional development effort.

### References
The EvaDB Documentation (https://evadb.readthedocs.io/en/latest/index.html#)
The EvaDB Repository (https://github.com/georgia-tech-db/evadb)
CS-4420 Piazza
OpenAI Cookbook
CodeLlama-13B-GGUF model by TheBloke (https://huggingface.co/TheBloke/CodeLlama-13B-GGUF)
ChatGPT for help with Python
