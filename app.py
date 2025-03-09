{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "627e38df",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pypdf in c:\\users\\hp\\anaconda3\\lib\\site-packages (5.3.1)\n",
      "Requirement already satisfied: pandas in c:\\users\\hp\\anaconda3\\lib\\site-packages (1.5.3)\n",
      "Requirement already satisfied: scikit-learn in c:\\users\\hp\\anaconda3\\lib\\site-packages (1.2.1)\n",
      "Requirement already satisfied: spacy in c:\\users\\hp\\anaconda3\\lib\\site-packages (3.8.4)\n",
      "Requirement already satisfied: sentence-transformers in c:\\users\\hp\\anaconda3\\lib\\site-packages (3.4.1)\n",
      "Requirement already satisfied: typing_extensions>=4.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pypdf) (4.12.2)\n",
      "Requirement already satisfied: python-dateutil>=2.8.1 in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from pandas) (2.8.2)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pandas) (2022.7)\n",
      "Requirement already satisfied: numpy>=1.21.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pandas) (1.26.4)\n",
      "Requirement already satisfied: threadpoolctl>=2.0.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from scikit-learn) (2.2.0)\n",
      "Requirement already satisfied: scipy>=1.3.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from scikit-learn) (1.10.1)\n",
      "Requirement already satisfied: joblib>=1.1.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from scikit-learn) (1.1.1)\n",
      "Requirement already satisfied: thinc<8.4.0,>=8.3.4 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (8.3.4)\n",
      "Requirement already satisfied: weasel<0.5.0,>=0.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (0.4.1)\n",
      "Requirement already satisfied: spacy-legacy<3.1.0,>=3.0.11 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (3.0.12)\n",
      "Requirement already satisfied: catalogue<2.1.0,>=2.0.6 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (2.0.10)\n",
      "Requirement already satisfied: requests<3.0.0,>=2.13.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (2.28.1)\n",
      "Requirement already satisfied: spacy-loggers<2.0.0,>=1.0.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (1.0.5)\n",
      "Requirement already satisfied: srsly<3.0.0,>=2.4.3 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (2.5.1)\n",
      "Requirement already satisfied: typer<1.0.0,>=0.3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (0.15.2)\n",
      "Requirement already satisfied: pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (2.10.6)\n",
      "Requirement already satisfied: cymem<2.1.0,>=2.0.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (2.0.11)\n",
      "Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (4.64.1)\n",
      "Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (1.0.12)\n",
      "Requirement already satisfied: setuptools in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (65.6.3)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (3.1.2)\n",
      "Requirement already satisfied: wasabi<1.2.0,>=0.9.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (1.1.3)\n",
      "Requirement already satisfied: preshed<3.1.0,>=3.0.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (3.0.9)\n",
      "Requirement already satisfied: langcodes<4.0.0,>=3.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from spacy) (3.5.0)\n",
      "Requirement already satisfied: packaging>=20.0 in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from spacy) (23.1)\n",
      "Requirement already satisfied: transformers<5.0.0,>=4.41.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from sentence-transformers) (4.49.0)\n",
      "Requirement already satisfied: torch>=1.11.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from sentence-transformers) (2.6.0)\n",
      "Requirement already satisfied: huggingface-hub>=0.20.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from sentence-transformers) (0.29.2)\n",
      "Requirement already satisfied: Pillow in c:\\users\\hp\\anaconda3\\lib\\site-packages (from sentence-transformers) (9.4.0)\n",
      "Requirement already satisfied: filelock in c:\\users\\hp\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (3.9.0)\n",
      "Requirement already satisfied: pyyaml>=5.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (6.0)\n",
      "Requirement already satisfied: fsspec>=2023.5.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from huggingface-hub>=0.20.0->sentence-transformers) (2025.3.0)\n",
      "Requirement already satisfied: language-data>=1.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from langcodes<4.0.0,>=3.2.0->spacy) (1.3.0)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (0.7.0)\n",
      "Requirement already satisfied: pydantic-core==2.27.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from pydantic!=1.8,!=1.8.1,<3.0.0,>=1.7.4->spacy) (2.27.2)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from python-dateutil>=2.8.1->pandas) (1.16.0)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (3.4)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (1.26.14)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2023.5.7)\n",
      "Requirement already satisfied: charset-normalizer<3,>=2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3.0.0,>=2.13.0->spacy) (2.0.4)\n",
      "Requirement already satisfied: confection<1.0.0,>=0.0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from thinc<8.4.0,>=8.3.4->spacy) (0.1.5)\n",
      "Requirement already satisfied: blis<1.3.0,>=1.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from thinc<8.4.0,>=8.3.4->spacy) (1.2.0)\n",
      "Requirement already satisfied: networkx in c:\\users\\hp\\anaconda3\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (2.8.4)\n",
      "Requirement already satisfied: sympy==1.13.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from torch>=1.11.0->sentence-transformers) (1.13.1)\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from sympy==1.13.1->torch>=1.11.0->sentence-transformers) (1.2.1)\n",
      "Requirement already satisfied: colorama in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from tqdm<5.0.0,>=4.38.0->spacy) (0.4.6)\n",
      "Requirement already satisfied: safetensors>=0.4.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.5.3)\n",
      "Requirement already satisfied: regex!=2019.12.17 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (2022.7.9)\n",
      "Requirement already satisfied: tokenizers<0.22,>=0.21 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from transformers<5.0.0,>=4.41.0->sentence-transformers) (0.21.0)\n",
      "Requirement already satisfied: shellingham>=1.3.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from typer<1.0.0,>=0.3.0->spacy) (1.5.4)\n",
      "Requirement already satisfied: click>=8.0.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from typer<1.0.0,>=0.3.0->spacy) (8.0.4)\n",
      "Requirement already satisfied: rich>=10.11.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from typer<1.0.0,>=0.3.0->spacy) (13.9.4)\n",
      "Requirement already satisfied: cloudpathlib<1.0.0,>=0.7.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from weasel<0.5.0,>=0.1.0->spacy) (0.21.0)\n",
      "Requirement already satisfied: smart-open<8.0.0,>=5.2.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from weasel<0.5.0,>=0.1.0->spacy) (5.2.1)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from jinja2->spacy) (2.1.1)\n",
      "Requirement already satisfied: marisa-trie>=1.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from language-data>=1.2->langcodes<4.0.0,>=3.2.0->spacy) (1.2.1)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich>=10.11.0->typer<1.0.0,>=0.3.0->spacy) (0.1.2)\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting en-core-web-sm==3.8.0\n",
      "  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl (12.8 MB)\n",
      "     -------------------------------------- 12.8/12.8 MB 808.9 kB/s eta 0:00:00\n",
      "\u001b[38;5;2m[+] Download and installation successful\u001b[0m\n",
      "You can now load the package via spacy.load('en_core_web_sm')\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n"
     ]
    }
   ],
   "source": [
    "!pip install pypdf pandas scikit-learn spacy sentence-transformers\n",
    "!python -m spacy download en_core_web_sm\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "99c17f52",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: tf-keras in c:\\users\\hp\\anaconda3\\lib\\site-packages (2.18.0)\n",
      "Requirement already satisfied: tensorflow<2.19,>=2.18 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tf-keras) (2.18.0)\n",
      "Requirement already satisfied: tensorflow-intel==2.18.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow<2.19,>=2.18->tf-keras) (2.18.0)\n",
      "Requirement already satisfied: numpy<2.1.0,>=1.26.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (1.26.4)\n",
      "Requirement already satisfied: setuptools in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (65.6.3)\n",
      "Requirement already satisfied: protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<6.0.0dev,>=3.20.3 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (3.20.3)\n",
      "Requirement already satisfied: libclang>=13.0.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (18.1.1)\n",
      "Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.6.0)\n",
      "Requirement already satisfied: wrapt>=1.11.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (1.14.1)\n",
      "Requirement already satisfied: six>=1.12.0 in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (1.16.0)\n",
      "Requirement already satisfied: opt-einsum>=2.3.2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (3.4.0)\n",
      "Requirement already satisfied: flatbuffers>=24.3.25 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (25.2.10)\n",
      "Requirement already satisfied: google-pasta>=0.1.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.2.0)\n",
      "Collecting keras>=3.5.0\n",
      "  Using cached keras-3.9.0-py3-none-any.whl (1.3 MB)\n",
      "Requirement already satisfied: termcolor>=1.1.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.5.0)\n",
      "Requirement already satisfied: packaging in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (23.1)\n",
      "Requirement already satisfied: requests<3,>=2.21.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.28.1)\n",
      "Requirement already satisfied: typing-extensions>=3.6.6 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (4.12.2)\n",
      "Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.31.0)\n",
      "Requirement already satisfied: astunparse>=1.6.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (1.6.3)\n",
      "Requirement already satisfied: h5py>=3.11.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (3.12.1)\n",
      "Requirement already satisfied: ml-dtypes<0.5.0,>=0.4.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.4.1)\n",
      "Requirement already satisfied: absl-py>=1.0.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.1.0)\n",
      "Requirement already satisfied: tensorboard<2.19,>=2.18 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.18.0)\n",
      "Requirement already satisfied: grpcio<2.0,>=1.24.3 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (1.70.0)\n",
      "Requirement already satisfied: wheel<1.0,>=0.23.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from astunparse>=1.6.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.38.4)\n",
      "Requirement already satisfied: namex in c:\\users\\hp\\anaconda3\\lib\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.0.8)\n",
      "Requirement already satisfied: optree in c:\\users\\hp\\anaconda3\\lib\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.14.0)\n",
      "Requirement already satisfied: rich in c:\\users\\hp\\anaconda3\\lib\\site-packages (from keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (13.9.4)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (3.4)\n",
      "Requirement already satisfied: charset-normalizer<3,>=2 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.0.4)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2023.5.7)\n",
      "Requirement already satisfied: urllib3<1.27,>=1.21.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from requests<3,>=2.21.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (1.26.14)\n",
      "Requirement already satisfied: markdown>=2.6.8 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (3.4.1)\n",
      "Requirement already satisfied: werkzeug>=1.0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.2.2)\n",
      "Requirement already satisfied: tensorboard-data-server<0.8.0,>=0.7.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.7.2)\n",
      "Requirement already satisfied: MarkupSafe>=2.1.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from werkzeug>=1.0.1->tensorboard<2.19,>=2.18->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.1.1)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from rich->keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (3.0.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\hp\\appdata\\roaming\\python\\python310\\site-packages (from rich->keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (2.15.1)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\hp\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.5.0->tensorflow-intel==2.18.0->tensorflow<2.19,>=2.18->tf-keras) (0.1.2)\n",
      "Installing collected packages: keras\n",
      "  Attempting uninstall: keras\n",
      "    Found existing installation: keras 2.11.0\n",
      "    Uninstalling keras-2.11.0:\n",
      "      Successfully uninstalled keras-2.11.0\n",
      "Successfully installed keras-3.9.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "    WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting keras==2.11.0\n",
      "  Using cached keras-2.11.0-py2.py3-none-any.whl (1.7 MB)\n",
      "Installing collected packages: keras\n",
      "  Attempting uninstall: keras\n",
      "    Found existing installation: keras 3.9.0\n",
      "    Uninstalling keras-3.9.0:\n",
      "      Successfully uninstalled keras-3.9.0\n",
      "Successfully installed keras-2.11.0\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "    WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "tensorflow-intel 2.18.0 requires keras>=3.5.0, but you have keras 2.11.0 which is incompatible.\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n"
     ]
    }
   ],
   "source": [
    "!pip install tf-keras\n",
    "!pip install keras==2.11.0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "761e635a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Found existing installation: protobuf 3.20.3\n",
      "Uninstalling protobuf-3.20.3:\n",
      "  Successfully uninstalled protobuf-3.20.3\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "ERROR: Exception:\n",
      "Traceback (most recent call last):\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_internal\\cli\\base_command.py\", line 160, in exc_logging_wrapper\n",
      "    status = run_func(*args)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_internal\\commands\\uninstall.py\", line 103, in run\n",
      "    uninstall_pathset.commit()\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_internal\\req\\req_uninstall.py\", line 424, in commit\n",
      "    self._moved_paths.commit()\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_internal\\req\\req_uninstall.py\", line 277, in commit\n",
      "    save_dir.cleanup()\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_internal\\utils\\temp_dir.py\", line 173, in cleanup\n",
      "    rmtree(self._path)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_vendor\\tenacity\\__init__.py\", line 328, in wrapped_f\n",
      "    return self(f, *args, **kw)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_vendor\\tenacity\\__init__.py\", line 408, in __call__\n",
      "    do = self.iter(retry_state=retry_state)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_vendor\\tenacity\\__init__.py\", line 364, in iter\n",
      "    raise retry_exc.reraise()\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_vendor\\tenacity\\__init__.py\", line 197, in reraise\n",
      "    raise self.last_attempt.result()\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\concurrent\\futures\\_base.py\", line 451, in result\n",
      "    return self.__get_result()\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\concurrent\\futures\\_base.py\", line 403, in __get_result\n",
      "    raise self._exception\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_vendor\\tenacity\\__init__.py\", line 411, in __call__\n",
      "    result = fn(*args, **kwargs)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\site-packages\\pip\\_internal\\utils\\misc.py\", line 128, in rmtree\n",
      "    shutil.rmtree(dir, ignore_errors=ignore_errors, onerror=rmtree_errorhandler)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\shutil.py\", line 750, in rmtree\n",
      "    return _rmtree_unsafe(path, onerror)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\shutil.py\", line 615, in _rmtree_unsafe\n",
      "    _rmtree_unsafe(fullname, onerror)\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\shutil.py\", line 620, in _rmtree_unsafe\n",
      "    onerror(os.unlink, fullname, sys.exc_info())\n",
      "  File \"C:\\Users\\hp\\anaconda3\\lib\\shutil.py\", line 618, in _rmtree_unsafe\n",
      "    os.unlink(fullname)\n",
      "PermissionError: [WinError 5] Access is denied: 'C:\\\\Users\\\\hp\\\\anaconda3\\\\Lib\\\\site-packages\\\\google\\\\~-otobuf\\\\internal\\\\_api_implementation.cp310-win_amd64.pyd'\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting protobuf==3.20.3\n",
      "  Using cached protobuf-3.20.3-cp310-cp310-win_amd64.whl (904 kB)\n",
      "Installing collected packages: protobuf\n",
      "Successfully installed protobuf-3.20.3\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
      "tensorflow-intel 2.18.0 requires keras>=3.5.0, but you have keras 2.11.0 which is incompatible.\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n",
      "WARNING: Ignoring invalid distribution -rotobuf (c:\\users\\hp\\anaconda3\\lib\\site-packages)\n"
     ]
    }
   ],
   "source": [
    "# Uninstall existing protobuf and reinstall a compatible version\n",
    "!pip uninstall -y protobuf\n",
    "!pip install protobuf==3.20.3\n",
    "\n",
    "# Restart the kernel after running this\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "68a74eb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Running in Jupyter Notebook mode...\n",
      "ATS Match Score (Sample Data): 66.32%\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import PyPDF2\n",
    "import spacy\n",
    "import torch\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "# Load NLP and BERT models\n",
    "nlp = spacy.load(\"en_core_web_sm\")\n",
    "bert_model = SentenceTransformer(\"all-MiniLM-L6-v2\")\n",
    "\n",
    "# Function to extract text from PDF\n",
    "def extract_text_from_pdf(uploaded_file):\n",
    "    reader = PyPDF2.PdfReader(uploaded_file)\n",
    "    text = \"\"\n",
    "    for page in reader.pages:\n",
    "        text += page.extract_text() + \"\\n\"\n",
    "    return text\n",
    "\n",
    "# Function to preprocess text\n",
    "def preprocess_text(text):\n",
    "    doc = nlp(text.lower())\n",
    "    return \" \".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])\n",
    "\n",
    "# Function to calculate similarity\n",
    "def get_similarity_score(job_desc, resume_text):\n",
    "    job_embedding = bert_model.encode([job_desc], convert_to_tensor=True)\n",
    "    resume_embedding = bert_model.encode([resume_text], convert_to_tensor=True)\n",
    "    similarity = cosine_similarity(job_embedding.cpu().detach().numpy(), \n",
    "                                   resume_embedding.cpu().detach().numpy())\n",
    "    return round(similarity[0][0] * 100, 2)\n",
    "\n",
    "# Streamlit UI\n",
    "def run_streamlit():\n",
    "    st.title(\"AI-Powered ATS Resume Screening System\")\n",
    "    st.write(\"Upload your resume and paste the job description to check ATS match score.\")\n",
    "\n",
    "    job_desc = st.text_area(\"Paste the Job Description Here\", height=150)\n",
    "    uploaded_file = st.file_uploader(\"Upload Your Resume (PDF)\", type=[\"pdf\"])\n",
    "\n",
    "    if uploaded_file is not None and job_desc:\n",
    "        with st.spinner(\"Processing...\"):\n",
    "            resume_text = extract_text_from_pdf(uploaded_file)\n",
    "            cleaned_resume = preprocess_text(resume_text)\n",
    "            cleaned_job_desc = preprocess_text(job_desc)\n",
    "            score = get_similarity_score(cleaned_job_desc, cleaned_resume)\n",
    "\n",
    "        st.success(f\"✅ ATS Match Score: {score}%\")\n",
    "        st.write(\"📌 The higher the score, the better your resume matches the job description.\")\n",
    "\n",
    "# Check if running in a Jupyter Notebook\n",
    "def is_notebook():\n",
    "    try:\n",
    "        get_ipython\n",
    "        return True\n",
    "    except NameError:\n",
    "        return False\n",
    "\n",
    "# Run in Jupyter Notebook\n",
    "if is_notebook():\n",
    "    print(\"Running in Jupyter Notebook mode...\")\n",
    "    \n",
    "    # Test with sample text\n",
    "    sample_job_desc = \"Looking for a Data Scientist with experience in Python and Machine Learning.\"\n",
    "    sample_resume_text = \"I am a Data Scientist skilled in Python, Machine Learning, and Deep Learning.\"\n",
    "    \n",
    "    cleaned_job_desc = preprocess_text(sample_job_desc)\n",
    "    cleaned_resume = preprocess_text(sample_resume_text)\n",
    "    \n",
    "    match_score = get_similarity_score(cleaned_job_desc, cleaned_resume)\n",
    "    print(f\"ATS Match Score (Sample Data): {match_score}%\")\n",
    "    \n",
    "# Run Streamlit App\n",
    "else:\n",
    "    run_streamlit()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "798a4143",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7c7206f",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
