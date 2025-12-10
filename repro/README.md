# ローカルRAG構築の再現手順

本文章ではローカルで動くRAGシステムを構築する手順を解説する。

## RAGシステムの概要

構築前にRAGシステムが凡そどのように構成・実装されているかを解説する。

RAGはインデックス構築と、その後の問い合わせ(クエリ)から成る。

インデックス構築は以下の手順で行われる:

1.  ドキュメントファイルを複数のテキストチャンクへ分割
2.  チャンク毎にベクトル化
3.  ベクトルとテキストチャンクをベクトルストアに追加
4.  全ドキュメントについて1から繰り返す

問い合わせは以下の手順で行われる:

1.  ユーザーから問い合わせ文(クエリー)を受け取る
2.  クエリーをベクトル化する
3.  ベクトルストアから2で得たベクトル近くのテキストチャンクを複数個取得する
4.  3で得たテキストチャンクを連結し、コンテキストとして1のクエリーと共にLLMへ与える
5.  LLMの返答をユーザーへ提示する

このような仕組みで動くRAGが必要とするモジュールは大きく分けて3つある。

*   テキストからベクトルを生成する Embedding
*   ベクトルを保存し検索できる ベクトルストア
*   連結したテキストチャンクをコンテキストとしてクエリーと共に処理する LLM

これらにRAGエンジンと適切なUIを加えることで、ローカルRAGシステムを構築できる。

## 構築するローカルRAGの詳細構成

今回構築するローカルRAGシステムの詳細構成、モジュールの選択は以下の通り。

カテゴリ       | 名称
---------------|----------------------------------
UI             | [Gradio][gradio]
RAGエンジン    | [LlamaIndex][llamaindex]
Embedding      | [EmbeddingGemma][embeddinggemma]
ベクトルストア | [DuckDB][duckdb]
LLM            | [Gemma 3 4B][gemma3_4b]

[gradio]:https://www.gradio.app/
[llamaindex]:https://developers.llamaindex.ai/python/framework/
[embeddinggemma]:https://huggingface.co/google/embeddinggemma-300m
[duckdb]:https://www.duckdb.org/
[llamacpp]:https://github.com/ggml-org/llama.cpp
[gemma3_4b]:https://huggingface.co/google/gemma-3-4b-it

主にCPUで動かす方法について解説するが、少しの修正でGPUで動かすこともでき、その方法も付記してある。

動作確認は以下の環境で行ったが、以下で解説する手順はLinux (Ubuntu)を主なターゲットとして記述する。
ただしWindowsであっても大きくは変わらない。

*   CPU: Core i9-9990K
*   RAM: 64GB (32GB程度で動くと考えられる)
*   OS:  Windows 11 25H2
*   (GPU: RTX4070 VRAM 12GB)

## 構築手順

### Python 環境のセットアップ

まずPythonの venv 環境をセットアップする。可能なら uv などのよりモダンな手段を用いても構わない。
Pythonのバージョンは3.13で確認したが、3.11以降であれば問題無く動作すると思われる。

```console
# 新しい venv 環境を作成する
$ python -m venv venv

# 作成した venv 環境を有効化する
$ source ./venv/bin/activate

# パッケージマネージャー pip を更新する
$ python -m pip install -U pip
```

### Python モジュールのインストール

次に必要なPythonモジュールをインストールする。

```console
$ python -m pip install -U \
    gradio \
    llama-index \
    llama-index-embeddings-huggingface \
    llama-index-vector-stores-duckdb \
    llama_index.llms.huggingface \
    hf_xet \
    torch
```

### HuggingFace 関連の設定

今回のローカルRAGシステムでは、EmbeddingとLLMのモデルを [HuggingFace][hf] から取得するので、
HuggingFaceのアカウントを作成し、コマンドライン上でログインしておく必要がある。
また Gated Model を利用するので、事前にモデルページでアクセス許可を得ておく必要もある。

Gated Modelの詳細については HuggingFace の
[Gated models](https://huggingface.co/docs/hub/models-gated)
を参照すること。

[hf]:https://huggingface.co/

これらの手順は既に実行しているのであれば、適宜スキップすることができる。

アカウント作成からコマンドライン上でのログインについての詳細は
[HuggingFace の公式ドキュメント](https://huggingface.co/docs/huggingface_hub/quick-start)
に記載されているので、本文章では以下でその手順に触れるにとどめる。

1. <https://huggingface.co/join> にアクセスしてアカウントを作成する
2. ログインした状態で <https://huggingface.co/settings/tokens> にアクセスしてトークンを作成する

    パーミッションは Repositories 下の以下の2つを設定しておくこと。
    他のパーミッションについては任意。

    *   Read access to contents of all repos under your personal namespace
    *   Read access to contents of all public gated repos you can access

3. ログインした状態で以下のモデルのページを開き、 `Agree and send request repo` をクリックする

    *   [google/embeddinggemma-300m][embeddinggemma]
    *   [google/gemma-3-4b-it][gemma3_4b]

    Gated modelsへのアクセス方法については
    [公式ドキュメント](https://huggingface.co/docs/hub/models-gated#access-gated-models-as-a-user)
    の記載も参照すること。

    もしかしたら1つのモデルでアクセスリクエストが受理されれば、google orgの他のモデルについてもアクセスできるようになっているかも。

4. コマンドラインで `hf auth login` を実行し、2で作成したトークン (`hf_` で始まる文字列)を登録する

    `hf` コマンドは前節でPythonモジュールと一緒にインストールされている

### データの準備

RAGの対象となるデータ≒プレインテキストファイルを準備する。
今回はテスト用としてテキストエディタVimのプレインテキストで書かれたマニュアル、約150ファイルで合計サイズ約10MBを利用する。
ファイルは ./var/data ディレクトリへ配置する。

以下はそのコマンド:

```console
# Vim (v9.1) のソースコードを取得
$ curl -fsLO https://github.com/vim/vim/archive/refs/tags/v9.1.0000.tar.gz

# 展開すると ./vim-9.1.0000 ディレクトリに必要なものが格納されている
$ tar xf v9.1.0000.tar.gz

# 必要なファイルだけを ./var/data ディレクトリへコピー
$ mkdir -p ./var/data
$ cp ./vim-9.1.0000/runtime/doc/*.txt ./var/data

# 不要になったファイルとディレクトリを消す
$ rm -rf v9.1.0000.tar.gz vim-9.1.0000

# 取得したファイル名を確認する
$ ls -CF ./var/data
arabic.txt      ft_sql.txt    map.txt       pi_getscript.txt  sponsor.txt   usr_09.txt  usr_45.txt
autocmd.txt     gui.txt       mbyte.txt     pi_gzip.txt       starting.txt  usr_10.txt  usr_50.txt
builtin.txt     gui_w32.txt   message.txt   pi_logipat.txt    syntax.txt    usr_11.txt  usr_51.txt
change.txt      gui_x11.txt   mlang.txt     pi_netrw.txt      tabpage.txt   usr_12.txt  usr_52.txt
channel.txt     hangulin.txt  motion.txt    pi_paren.txt      tagsrch.txt   usr_20.txt  usr_90.txt
cmdline.txt     hebrew.txt    netbeans.txt  pi_spec.txt       term.txt      usr_21.txt  usr_toc.txt
debug.txt       help.txt      options.txt   pi_tar.txt        terminal.txt  usr_22.txt  various.txt
debugger.txt    helphelp.txt  os_390.txt    pi_vimball.txt    testing.txt   usr_23.txt  version4.txt
develop.txt     howto.txt     os_amiga.txt  pi_zip.txt        textprop.txt  usr_24.txt  version5.txt
diff.txt        if_cscop.txt  os_beos.txt   popup.txt         tips.txt      usr_25.txt  version6.txt
digraph.txt     if_lua.txt    os_dos.txt    print.txt         todo.txt      usr_26.txt  version7.txt
editing.txt     if_mzsch.txt  os_haiku.txt  quickfix.txt      uganda.txt    usr_27.txt  version8.txt
eval.txt        if_ole.txt    os_mac.txt    quickref.txt      undo.txt      usr_28.txt  version9.txt
farsi.txt       if_perl.txt   os_mint.txt   quotes.txt        userfunc.txt  usr_29.txt  vi_diff.txt
filetype.txt    if_pyth.txt   os_msdos.txt  recover.txt       usr_01.txt    usr_30.txt  vim9.txt
fold.txt        if_ruby.txt   os_os2.txt    remote.txt        usr_02.txt    usr_31.txt  visual.txt
ft_ada.txt      if_sniff.txt  os_qnx.txt    repeat.txt        usr_03.txt    usr_32.txt  windows.txt
ft_context.txt  if_tcl.txt    os_risc.txt   rileft.txt        usr_04.txt    usr_40.txt  workshop.txt
ft_mp.txt       indent.txt    os_unix.txt   russian.txt       usr_05.txt    usr_41.txt
ft_ps1.txt      index.txt     os_vms.txt    scroll.txt        usr_06.txt    usr_42.txt
ft_raku.txt     insert.txt    os_win32.txt  sign.txt          usr_07.txt    usr_43.txt
ft_rust.txt     intro.txt     pattern.txt   spell.txt         usr_08.txt    usr_44.txt
```

### インデックス構築

[indexing.py スクリプト](./indexing.py) を実行してインデックスを構築する。

初回実行時のみEmbeddingモデルをネットワークからダウンロードするために余分な時間がかかる。
2回目以降はローカルキャッシュから読みこむ。

```console
$ ./indexing.py
```

インデックスの構築には、テスト用の約150ファイルの合計約10MBで約1時間程かかるが、
GPUを用いれば数分に短縮される。

ここでのインデックス構築には `SimpleDirectoryReader` を用いている。
この `SimpleDirectoryReader` はプレインテキストのほか Microsoft Word や PDF などにも対応している。
詳しくは <https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/> を参照のこと。

### クエリー用のWeb UIを起動

[querywebui.py スクリプト](./querywebui.py) を実行してWeb UIサーバーを起動する。

```conosle
$ ./querywebui.py
```

しばらくすると `Running on local URL:  http://127.0.0.1:7860` というメッセージが表示されるので、
Webブラウザで <http://127.0.0.1:7860> へアクセスするとクエリー用のUIが表示される。

初回実行時のみLLMモデルをネットワークからダウンロードするために余分な時間がかかる。
2回目以降はローカルキャッシュから読みこむ。

UIでは `query` 入力エリアに質問を入力し `Submit` ボタンを押すと RAG が実行される。
CPUで実行している場合はおよそ30秒～1分ほどで結果が `output` テキストエリアに出力される。

以下に動作確認用の質問と回答の例を示す:

*   Who did create the Vim?

    A. Most of Vim was created by Bram Moolenaar \<Bram@vim.org\>. Parts of the documentation come from several Vi manuals, written by: W.N. Joy, Alan P.W. Hewett, and Mark Horton. The Vim editor is based on Stevie and includes (ideas from) other software, worked on by the people mentioned here.

*   Vimにはいくつのモードがある?

    7つの基本的なモードと、それらに派生した7つの追加モードがあります。

### 構築手順のまとめ

ここまででローカルRAGシステムの構築と、そのクエリー実行方法を解説した。
最初のPythonの実行環境作成と、HuggingFace上の準備、加えてデータの準備が手間ではあるが、
あとは短いスクリプトを2つ実行するだけとお手軽になっている。

インデックス構築までは、1度実行してしまえば再び実行する必要は基本的に無い。
あるとすればドキュメントが更新されてそれを反映するくらい。

[indexing.py](./indexing.py) と [querywebui.py](./querywebui.py) は短くシンプルなので
EmbeddingやLLMのモデルを変更したり、ベクトルストアを変更するのも容易であろう。
その際は [公式ドキュメント: Introduction to RAG](https://developers.llamaindex.ai/python/framework/understanding/rag/) が参考になる。

## 付録

### GPUで使うには

既に見てきたとおり、Python + CPUだけでRAGを利用することはできるが、快適とは言い難い。
そのため本節では torch ライブラリを GPU 対応のモノに入れ替える方法を紹介する。

入れ替えは簡単で、プラットフォームに応じてコマンドを1つ実行するだけ。
たとえばWindowsで CUDA 12.8 を利用するなら、以下のコマンドを実行する。

```console
$ python -m pip install -U torch --index-url https://download.pytorch.org/whl/cu128
```

LinuxでCUDA 13.0を使うならば以下のようになる。

```console
$ python -m pip install -U torch --index-url https://download.pytorch.org/whl/cu130
```

プラットフォーム毎の実行すべきコマンドは
[PyTorchの公式サイトの Start Locally](https://pytorch.org/get-started/locally/)
で確認できる。
リンク先にかかれたインストールコマンドに含まれたtorchvisionは、
今回のローカルRAGシステムでは不要なので適宜読み替えること。

これにより特にインデックス構築(Embedding)が約3分で終わるほどに速くなる。
LLMのほうは回答によっては約15秒くらいに速くなるが、遅いままの場合もありうる。

### LLLMにllama.cppのllama-serverを使う

RAGで使うLLMには性能の良いモデルが望ましい。
そのようなモデルはPython + GPUベースでも実行が遅くなる。
そこでLLM部分だけをC++で書かれた llama.cpp に置き換える。
llama.cpp には llama-server というLLM推論のためのHTTPサーバーがあり、
これを LlamaIndex と組み合わせて使う。

#### llama.cpp を利用するための準備

**llama.cpp のインストール**

llama.cpp には一部のプラットフォーム向けにコンパイル済みバイナリが用意されている。
このバイナリはほぼ毎日ビルドされており
<https://github.com/ggml-org/llama.cpp/releases/latest>
から最新のプラットフォーム向けの最新バイナリをダウンロードできる。

特にWindows版はアクセラレーションの種類も幅広く用意されており、展開して即使える。

Linux版は Ubuntu 用に CPU と Vulkan しか提供されていない。
Ubuntu 以外のディストリで動くかは不明。
[公式のビルドガイド](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) に従えば、
自身でビルドするのも比較的容易なので、そちらを参照のこと。

**HuggingFace トークンを設定**

[HuggingFace 関連の設定](#huggingface-関連の設定) で取得したトークンを
llama.cpp 用に環境変数 `HF_TOKEN` で設定する。

設定例:

```console
$ export HF_TOKEN=hf_xxxxxxxxx
```

`hf_xxxxxxxxx` 部分は自身のトークンで置き換える。

**Python モジュールの追加インストール**

llama.cppをLLMとして使うために、追加のPythonモジュールをインストールする

```console
$ python -m pip install llama-index-llms-llamafile
```

**Gated models のアクセス許可を得る**

llama.cpp では以下のモデルを利用する。あらかじめ Gated models のアクセス許可を得る。

*   [google/gemma-3-4b-it-qat-q4\_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf)

#### llama-server の起動

llama-server (llama.cpp) の起動は以下のようなコマンドで行う。
初回だけモデルのダウンロードが発生する。
2回目以降はローカルキャッシュのモノが利用される。

```console
$ llama-server -hf google/gemma-3-4b-it-qat-q4_0-gguf -c 0 -fa on
```

LLM推論HTTPサーバーは <http://127.0.0.1:8080> で起動するので、以下のコマンドで動作確認ができる。

```console
$ curl -s http://127.0.0.1:8080/completion -d '{"prompt": "Whats your model name?", "temperature": 0}'  | jq .content
"\n\nI am a large language model, trained by Google.\n\nI don't have a \"model name\" in the traditional sense. I'm a product of Google AI, and I'm based on the PaLM 2 architecture.\n\nWould you like to know more about how I was trained or my capabilities?"
```

#### クエリー用のWeb UIスクリプトの変更点

LLMにllama-serverを利用するためにはスクリプトに若干の変更が必要。
実際に変更したものが [querywebui2.py](querywebui2.py) になる。

```conosle
$ ./querywebui2.py
```

HuggingFaceLLMを使った querywebui.py とllama-serverを使った querywebui2.pyの差分は以下の通り

```diff
--- ./querywebui.py
+++ ./querywebui2.py
@@ -11,14 +11,12 @@
 from llama_index.vector_stores.duckdb import DuckDBVectorStore
 from llama_index.core import VectorStoreIndex
 from llama_index.embeddings.huggingface import HuggingFaceEmbedding
-from llama_index.llms.huggingface import HuggingFaceLLM
+from llama_index.llms.llamafile import Llamafile
 import gradio as gr
 logger.info("imported packages")

 # Embedding には HuggingFace 上の google/embeddinggemma-300m を利用する
 embedding_model_name = "google/embeddinggemma-300m"
-# LLM には HuggingFace 上の google/gemma-3-4b-it を利用する
-llm_model_name = "google/gemma-3-4b-it"

 # Embedding モデルを読みこむ
 Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
@@ -31,11 +29,7 @@
 logger.info("index ready")

 # LLM モデルを読みこむ
-Settings.llm = HuggingFaceLLM(
-        model_name=llm_model_name,
-        tokenizer_name=llm_model_name,
-        context_window=131072,
-)
+Settings.llm = Llamafile(base_url="http://localhost:8080", temperature=0, seed=0)
 logger.info("LLM model ready")

 # 検索エンジンを作成
#llama-index-llms-llamafile
```
