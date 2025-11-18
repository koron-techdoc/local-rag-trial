# Local RAG for Vim Documents

Gemma 3 1B と Embedding Gemma3 300M で
Vim のドキュメントのローカルRAGを構築してみる試み。

とりあえず使ってみるのは [LlamaIndex][llamaindex]

[llamaindex]:(https://github.com/run-llama/llama_index)

## 準備

```
$ python -m pip install -U pip
$ python -m pip install -U torch --index-url https://download.pytorch.org/whl/cu128
$ python -m pip install -U llama-index
$ python -m pip install -U hf_xet
$ python -m pip install -U llama-index-llms-huggingface
```

data を Vim のドキュメントフォルダにリンクする

## 実験その1

[./test01.py](./test01.py) メモリ上にベクトルDBを配置し、簡単なラグを実施。

<details>
<summary>How many modes does Vim have?</summary>

```
$ python ./test01.py "How many modes does Vim have?"
2025-11-18 15:58:06,063 - INFO - setup logger
2025-11-18 15:58:15,775 - INFO - import llama_index related
2025-11-18 15:58:15,795 - INFO - Load pretrained SentenceTransformer: google/embeddinggemma-300m
2025-11-18 15:58:21,574 - INFO - 14 prompts are loaded, with the keys: ['query', 'document', 'BitextMining', 'Clustering', 'Classification', 'InstructionRetrieval', 'MultilabelClassification', 'PairClassification', 'Reranking', 'Retrieval', 'Retrieval-query', 'Retrieval-document', 'STS', 'Summarization']
2025-11-18 15:58:21,577 - INFO - loaded embedding model
2025-11-18 15:58:21,780 - INFO - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
2025-11-18 15:58:25,114 - INFO - loaded LLM model
2025-11-18 15:58:25,419 - INFO - loaded documents
2025-11-18 16:01:19,456 - INFO - generated embeddings
2025-11-18 16:01:19,456 - INFO - generated the query engine
2025-11-18 16:01:20,364 - INFO - queried
7
---------------------
```

</details>

<details>
<summary>List the names and descriptions of Vim modes</summary>

```
$ python ./test01.py "List the names and descriptions of Vim modes"
2025-11-18 16:07:37,705 - INFO - setup logger
2025-11-18 16:07:47,425 - INFO - import llama_index related
2025-11-18 16:07:47,445 - INFO - Load pretrained SentenceTransformer: google/embeddinggemma-300m
2025-11-18 16:07:53,116 - INFO - 14 prompts are loaded, with the keys: ['query', 'document', 'BitextMining', 'Clustering', 'Classification', 'InstructionRetrieval', 'MultilabelClassification', 'PairClassification', 'Reranking', 'Retrieval', 'Retrieval-query', 'Retrieval-document', 'STS', 'Summarization']
2025-11-18 16:07:53,118 - INFO - loaded embedding model
2025-11-18 16:07:53,343 - INFO - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
2025-11-18 16:07:56,728 - INFO - loaded LLM model
2025-11-18 16:07:57,030 - INFO - loaded documents
2025-11-18 16:10:51,567 - INFO - generated embeddings
2025-11-18 16:10:51,568 - INFO - generated the query engine
2025-11-18 16:10:56,009 - INFO - queried
7

The provided text describes the different modes available in Vim.  The text mentions seven distinct modes: Normal, Normal-mode, command-mode, Ex mode, Terminal-Job mode, Operator-pending mode, and Visual mode.
```

</details>

<details>
<summary>List MURAOKA's major contributions</summary>

```
$ python ./test01.py "List MURAOKA's major contributions"
2025-11-18 16:30:57,901 - INFO - setup logger
2025-11-18 16:30:57,901 - INFO - query: List MURAOKA's major contributions
2025-11-18 16:31:07,474 - INFO - import llama_index related
2025-11-18 16:31:07,493 - INFO - Load pretrained SentenceTransformer: google/embeddinggemma-300m
2025-11-18 16:31:13,068 - INFO - 14 prompts are loaded, with the keys: ['query', 'document', 'BitextMining', 'Clustering', 'Classification', 'InstructionRetrieval', 'MultilabelClassification', 'PairClassification', 'Reranking', 'Retrieval', 'Retrieval-query', 'Retrieval-document', 'STS', 'Summarization']
2025-11-18 16:31:13,070 - INFO - loaded embedding model
2025-11-18 16:31:13,280 - INFO - We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
2025-11-18 16:31:16,606 - INFO - loaded LLM model
2025-11-18 16:31:16,911 - INFO - loaded documents
2025-11-18 16:34:10,670 - INFO - generated embeddings
2025-11-18 16:34:10,670 - INFO - generated the query engine
2025-11-18 16:34:16,272 - INFO - queried: List MURAOKA's major contributions

Muraoka Taro, a researcher, made contributions to GTK, GUI, and Raku programming.  He contributed to the development of GUI frameworks, improved rendering, and introduced new support for Unicode in Raku programming. Specifically, he worked on the concepts of multi-byte encoding and provided improvements to the UI rendering. He also refined the Raku programming language and improved the visual presentation.
```

</details>

微妙に間違っているけど、なかなか良い線は行ってそう。
`MURAOKA` から `Muraoka Taro` を持ってきたのは、特徴的な苗字かつVimのドキュメントという狭い範囲ではあるが、良く引っ張ってきたって感じ。

実験のたびにベクトルデータベースの構築をするので、時間がかかる。
次はベクトルデータベースを永続化しよう。

### 次の目標

-   ベクトルデータベースの永続化 & 再利用
    -   イテレーションに時間がかかりすぎてる
-   内部で起こってることを可視化
    -   どのようにvector storeへ分けてるか
    -   どのようにvector storeへ問い合わせているか
    -   LLMに何をどう食わせているか
-   HuggingFaceLLMで量子化モデルを使う
    -   Gemini3 4B it のQ8あたりを使えないか?
-   バックエンドLLMにllama-server (llama.cpp) を使う
    -   LLMだけではなくembeddingにも使えないか?
