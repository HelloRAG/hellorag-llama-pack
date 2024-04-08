[[English](https://github.com/HelloRAG/hellorag-llama-pack)] |  [[中文](https://github.com/HelloRAG/hellorag-llama-pack/blob/main/README_CN.md)]
# Instruction for LlamaIndex Integration

After your datasets have been processed and curated using the HelloRAG platform, you are ready to export the results for integration into the subsequent RAG workflow stages. This guide provides the instructions on how to establish a connection with the open sourced LlamaIndex RAG framework using HelloRAG Llama Pack. 

For instructions of HelloRAG platform itself, please check the "[How to use HelloRAG](https://hellorag.ai/tutorial)" tutorial.

# 1. Basic Setup
## 1.1 Get the HelloRAG Llama Pack
```git clone https://github.com/HelloRAG/hellorag-llama-pack.git```

## 1.2 Install the Running Environment
```pip install -r requirements.txt  ```

## 1.3 Prepare a Local Data Directory
Put the exported zip files from HelloRAG under a directory such as /data/hellorag/results 
>/data/hellorag/results  
> ├──test_only.pdf.zip  
> ├──some_other.pdf.zip  
> ...

## 1.4 Setup LLM (For example: OpenAI)
Make sure your API key is available to your code by setting it as an environment variable.   
In MacOS and Linux, this is the command:

```export OPENAI_API_KEY=XXXXX```  
or on Windows it is

```set OPENAI_API_KEY=XXXXX```

Also you can set the key in python code (NOT RECOMMENDED):


```
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="xxx", api_key="yyy")
```

# 2. Indexing for LlamaIndex

## 2.1 Indexing for Local Data File

This is a simple and quick way for you to do a rapid experimentation or internal product test. You can also modify it in further based on the HelloRAG Llama Pack. 

### 2.1.1 Setup for the Initial Indexing & Re-indexing

Assume the index file is to be placed under /data/hellorag/index.
```python
from hellorag_llama_index_pack.base import HelloragLlamaindexPack
hellorag_pack = HelloragLlamaindexPack(
    base_path="/data/hellorag/results",
    need_refresh=True,
    index_path="/data/hellorag/index",
)
```

### 2.1.2 Use the Local Index in RAG Query 

```python
from hellorag_llama_index_pack.base import HelloragLlamaindexPack
hellorag_pack = HelloragLlamaindexPack(
    index_path="/data/hellorag/index",
)
response = hellorag_pack.run("Enter your question here... ")
print(response)
print(response.source_nodes)
```

## 2.2 Indexing for Vector Database

In product-grade applications, services are typically provided by multiple applications under a load balancer. Therefore, storing referenced local files is not a suitable approach. S3 or other Object Storage Service (OSS) options could be used as alternatives. However, issues such as inconsistencies due to index updates may still arise. A relatively better approach is to store vectors in a dedicated vector database. For this instruction, we will use qdrant as an example.

### 2.2.1 Prerequisites
* Install Qdrant (as an example) and establish collection. For details please refer to qdrant [doc](https://qdrant.tech/documentation/). 
* Install LlamaIndex qdrant store: ```pip install llama-index-vector-stores-qdrant```

### 2.2.2 Setup for the Initial Indexing & Re-indexing

Assume the name of the collection is hellorag, and the Qdrant database connection is 192.168.1.10:1356 
```python
from hellorag_llama_index_pack.base import HelloragLlamaindexPack
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core import StorageContext
qdrant = qdrant_client.QdrantClient(
    location="192.168.1.10:1356"
)
# Create Qdrant vector storing
qdrant_vector_store = QdrantVectorStore(client=qdrant, collection_name="hellorag")
storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
hellorag_pack = HelloragLlamaindexPack(
    base_path="/data/hellorag-result",
    need_refresh=True,
    storage_context=storage_context,
)
```
### 2.2.3 Use the Vector Database Index in RAG Query 
```python
from hellorag_llama_index_pack.base import HelloragLlamaindexPack
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core import StorageContext
qdrant = qdrant_client.QdrantClient(
    location="192.168.1.10:1356"
)
# Create Qdrant Vector Store
qdrant_vector_store = QdrantVectorStore(client=qdrant, collection_name="hellorag")
storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
hellorag_pack = HelloragLlamaindexPack(
    storage_context=storage_context,
)
response = hellorag_pack.run("Enter your question here... ")
print(response)
print(response.source_nodes)
```

### 2.2.4 References for Qdrant and Other Vector Stores

[Qdrant Vector Store](https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo.html)  
For other types of vector database, please refer to [Vector Stores](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html)

# 3 RAG Workflow involving Images
By default, the extracted images are included in the RAG workflow.   
To exclude images from the retrieval, the parameter ```no_use_image_in_rag=True``` should be enabled at the index creating stage.   
Only images that have been exported with complete "title" and "description" annotations can be incorporated in the following RAG stages.   
After querying or retrieving, we can access these images by using the `source_nodes' to locate the corresponding ImageNode.

Notes:
* Images are typically stored in the index as base64 format. The returned images from 'source_node.node.image' are by default in PNG format. Format conversion, if necessary, can be added to the corresponding header of the ImageNode 
* The above-mentioned approach is rather for quick experiments or demos involving limited number of images. For production-grade implementation, it is recommended to store the images at a server that provides an accessible URL. Then, add ``image_to_url_function=fun_image_to_url`` as a parameter when building the index

# 4. Others
## Are there any parameter configurable? 
* Check the comments in the HelloRAG Llama Pack code.  

## How to deal with unrecognizable characters for Chinese?
* Add your local font when creating the initial indexing for hellorag_pack, for example,  
```python
hellorag_pack = HelloragLlamaindexPack(
    .......,
    need_refresh=True,
    font_path="/System/Library/Fonts/STHeiti Light.ttc"
)
```
* When using RAG, there's no need to assign an initial value to this variable.

## What if I want to use local LLM / Embedding models instead?
* Please refer to [Configuring Settings](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings.html).

## Are there demos that are faster to get started?
* See [DEMOS](https://github.com/HelloRAG/hellorag-demos)