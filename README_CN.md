[[English](https://github.com/HelloRAG/hellorag-llama-pack)] |  [[中文](https://github.com/HelloRAG/hellorag-llama-pack/blob/main/README_CN.md)]
# LlamaIndex 集成使用方式

当您使用HelloRAG.ai处理完您的数据，您已经准备好将结果导出以供后续的RAG工作流阶段使用。本教程提供了如何使用HelloRAG Llama Pack与开放源代码的LlamaIndex RAG框架建立连接的说明。

如果您需要使用HelloRAG平台，请参阅“[如何使用HelloRAG](https://hellorag.ai/tutorial)”教程。


# 1. 基础设置
## 1.1 获取HelloRAG Llama Pack
```git clone https://github.com/HelloRAG/hellorag-llama-pack.git```

## 1.2 安装运行环境依赖
```pip install -r requirements.txt  ```

## 1.3 准备本地数据目录
把HelloRAG导出的zip文件放在一个目录下，例如/data/hellorag/results
>/data/hellorag/results  
> ├──test_only.pdf.zip  
> ├──some_other.pdf.zip  
> ...

## 1.4 设置要用的（以下以OpenAI为例）
可以通过设置环境变量OPENAI_API_KEY来设置您的API密钥。  
如果您使用的是MacOS或Linux，请执行以下命令：  

```export OPENAI_API_KEY=XXXXX```  
如果您使用的是Windows，请执行以下命令：  

```set OPENAI_API_KEY=XXXXX```

当然您也可以在python代码中设置（不推荐）：

```
from llama_index.llms.openai import OpenAI
Settings.llm = OpenAI(model="xxx", api_key="yyy")
```

# 2. 用LlamaIndex建立索引

## 2.1 使用本地本间缓存索引方式

这是一种快速的快速实验或内部产品测试的方法。您也可以您的需要根据HelloRAG Llama Pack进行功能定制修改。

### 2.1.1 首次索引 & 重建索引
假设您的索引文件要放到/data/hellorag/index下。
```python
from hellorag_llama_index_pack.base import BetterTablesHelloragPack
hellorag_pack = BetterTablesHelloragPack(
    base_path="/data/hellorag/results",
    need_refresh=True,
    index_path="/data/hellorag/index",
)
```

### 2.1.2 在RAG流程中用本地索引进行查询 

```python
from hellorag_llama_index_pack.base import BetterTablesHelloragPack
hellorag_pack = BetterTablesHelloragPack(
    index_path="/data/hellorag/index",
)
response = hellorag_pack.run("What is the minimum and maximum TOEFL iBT score range for the Advanced level in the Speaking section? ")
print(response)
print(response.source_nodes)
```

## 2.2 使用向量数据库缓存索引方式

在产品级应用中，服务通常由多个应用程序在负载均衡器下提供。因此，索引存储的本地文件并不是最合适的方法。S3或其他对象存储服务（OSS）选项可以作为替代。然而，由于索引更新导致的不一致问题、IO性能仍然存在。一个建议的方法是将索引存储在外部向量库中，在性能、扩容、同步性上面逗有足够的保障和提升。

### 2.2.1 前置安装Prerequsites
* 安装Qdrant，并建立collection。由于不是本篇主要内容，请参考[Qdrant](htpps://qdrant.tech/documentation/)相关文档进行操作   
* 安装llama-index的qdrant相关向量存储库 ```pip install llama-index-vector-stores-qdrant```

### 2.2.2 首次索引 & 重建索引
以下代码示例假设您已经建立Qdrant的collection，名为hellorag，并且Qdrant的地址为192.168.1.10:1356
```python
from hellorag_llama_index_pack.base import BetterTablesHelloragPack
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core import StorageContext
qdrant = qdrant_client.QdrantClient(
    location="192.168.1.10:1356"
)
# Create Qdrant vector storing
qdrant_vector_store = QdrantVectorStore(client=qdrant, collection_name="hellorag")
storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
hellorag_pack = BetterTablesHelloragPack(
    base_path="/Users/xlent/Downloads/hellorag-result",
    need_refresh=True,
    storage_context=storage_context,
)
```
### 2.2.3 在RAG流程中使用向量库查询 
```python
from hellorag_llama_index_pack.base import BetterTablesHelloragPack
from llama_index.vector_stores.qdrant import QdrantVectorStore
import qdrant_client
from llama_index.core import StorageContext
qdrant = qdrant_client.QdrantClient(
    location="192.168.1.10:1356"
)
# Create Qdrant Vector Store
qdrant_vector_store = QdrantVectorStore(client=qdrant, collection_name="hellorag")
storage_context = StorageContext.from_defaults(vector_store=qdrant_vector_store)
hellorag_pack = BetterTablesHelloragPack(
    storage_context=storage_context,
)
response = hellorag_pack.run("What is the minimum and maximum TOEFL iBT score range for the Advanced level in the Speaking section? ")
print(response)
print(response.source_nodes)
```

### 2.2.4 参考文档及使用其他向量数据库文档

[Qdrant Vector Store](https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo.html)  
其他向量库存档使用方式参见 [Vector Stores](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores.html)

# 3. 其他注意事项
## 参数都是可选的？应该怎么使用？
具体请参见代码注释

## 中文出现乱码
在建立索引初始化hellorag_pack的时候加上本地的字体路径，例如
```python
hellorag_pack = BetterTablesHelloragPack(
    .......,
    need_refresh=True,
    font_path="/System/Library/Fonts/STHeiti Light.ttc"
)
```
* 使用的时候不用加这个变量初始值

## 我想用本地LLM/Embedding 模型
参考 [Configuring Settings](https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/settings.html).
