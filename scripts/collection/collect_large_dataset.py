"""collect_large_dataset.py — Thu thap du lieu quy mo lon cho VulHunter.

Muc tieu:
    - ~100,000 ham Python an toan (binary_label=0)
    - ~25,000 ham Python co lo hong (binary_label=1)

Tieu chi do dai ham (huong toi ~1024 tokens sau tokenizer CodeBERT):
    - CodeBERT tokenizer trung binh ~1.8 tokens/word trong code Python
    - De dat 1024 tokens can ~560 words ~ 100-200 dong code
    - Chon: min_words=80, max_words=600  (~ 15-80 dong thuc te)
    - Cho vulnerable: min_words=60 (de khop voi data hien co)

Dedup voi data cu:
    - data/raw/python_cvefixes_methods.jsonl  (code + safe_code)
    - data/raw/ghsa/ghsa_methods.jsonl        (code + safe_code)

Nguon thu thap:
    Benign  : GitHub REST API - cac repo Python lon, da dang
    Vulnerable: GitHub Security Advisories API (GraphQL) +
                tim commit fix tu cac advisory co fix-commit URL

Yeu cau:
    pip install requests
    Classic PAT can quyen: public_repo, read:org, read:user

Su dung:
    python scripts/collection/collect_large_dataset.py \\
        --token ghp_XXXX \\
        --benign-limit 100000 \\
        --vuln-limit 25000 \\
        --output-dir data/raw/large

    # Tiep tuc neu bi ngat giua chung:
    python scripts/collection/collect_large_dataset.py \\
        --token ghp_XXXX --append

    # Chi collect benign:
    python scripts/collection/collect_large_dataset.py \\
        --token ghp_XXXX --mode benign

    # Chi collect vulnerable:
    python scripts/collection/collect_large_dataset.py \\
        --token ghp_XXXX --mode vulnerable
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

# ─────────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("vulhunter.large_collect")

DEFAULT_OUT_DIR = ROOT / "data" / "raw" / "large"
EXISTING_DATA_FILES = [
    ROOT / "data" / "raw" / "python_cvefixes_methods.jsonl",
    ROOT / "data" / "raw" / "ghsa" / "ghsa_methods.jsonl",
]

# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────
# CodeBERT tokenizer ~1.8 tokens/word => 1024 tokens ~ 568 words
# Ham co 80-600 words la phu hop (tuong duong ~15-100 dong thuc te)
BENIGN_MIN_WORDS   = 80
BENIGN_MAX_WORDS   = 600
VULN_MIN_WORDS     = 50    # vulnerable data hien co ngan hon
VULN_MAX_WORDS     = 600

MIN_LINES          = 8     # loai trivial
MAX_LINES          = 120   # loai khong lo

# ─────────────────────────────────────────────────────────────────────────────
# Benign target repos (~150 repos de dat 85k mau)
# ─────────────────────────────────────────────────────────────────────────────
BENIGN_REPOS = [
    # ── Core utilities ──────────────────────────────────────────────────────
    "psf/black", "pypa/pip", "pallets/click", "psf/requests",
    "tqdm/tqdm", "dateutil/dateutil", "pytoolz/toolz",
    "more-itertools/more-itertools", "mahmoud/boltons",
    "python-attrs/attrs", "python-trio/trio",
    "jd/tenacity", "gruns/imohash", "pypa/virtualenv",
    "pypa/build", "pypa/twine", "pypa/flit",
    "astral-sh/ruff",
    # ── Data science & math ──────────────────────────────────────────────────
    "sympy/sympy", "statsmodels/statsmodels", "mwaskom/seaborn",
    "bokeh/bokeh", "altair-viz/altair", "networkx/networkx",
    "scikit-image/scikit-image", "pydata/xarray",
    "nipy/nipype", "astropy/astropy", "biopython/biopython",
    "pymc-devs/pymc", "pgmpy/pgmpy", "lifelines/lifelines",
    "dask/dask", "zarr-developers/zarr-python",
    # ── NLP & ML tooling ─────────────────────────────────────────────────────
    "nltk/nltk", "sloria/textblob", "chartbeat/wordfreq",
    "jamesturk/jellyfish", "seatgeek/fuzzywuzzy",
    "snowballstem/snowball",
    # ── Testing ──────────────────────────────────────────────────────────────
    "pytest-dev/pytest", "HypothesisWorks/hypothesis",
    "spulec/freezegun", "michael-the-grey/pydantic-factories",
    "FactoryBoy/factory_boy", "pytest-dev/pluggy",
    # ── CLI, config, templates ───────────────────────────────────────────────
    "pallets/jinja", "yaml/pyyaml", "keleshev/schema",
    "faif/python-patterns", "willmcgugan/rich",
    "textualize/textual", "tmbo/questionary",
    "prompt-toolkit/python-prompt-toolkit",
    "kislyuk/argcomplete", "tartley/colorama",
    "docopt/docopt", "google/python-fire",
    # ── Networking & HTTP ────────────────────────────────────────────────────
    "aio-libs/aiohttp", "encode/httpx", "urllib3/urllib3",
    "twisted/twisted", "celery/celery",
    "pika-org/pika", "mosquitto/mosquitto",
    "tornadoweb/tornado", "benoitc/gunicorn",
    "gevent/gevent",
    # ── Web frameworks ───────────────────────────────────────────────────────
    "pallets/werkzeug", "encode/starlette", "tiangolo/fastapi",
    "pydantic/pydantic", "django/django", "bottlepy/bottle",
    "falconry/falcon", "sanic-org/sanic",
    "hugapi/hug", "webpy/web.py",
    # ── Databases & ORM ──────────────────────────────────────────────────────
    "sqlalchemy/sqlalchemy", "tortoise-orm/tortoise-orm",
    "coleifer/peewee", "pynamodb/PynamoDB",
    "encode/databases", "piccolo-orm/piccolo",
    # ── Serialization & parsing ───────────────────────────────────────────────
    "simplejson/simplejson", "tefra/xsdata",
    "yaml/pyyaml", "lxml/lxml",
    "defusedxml/defusedxml", "ply/ply",
    "pyparsing/pyparsing", "python-markdown/markdown",
    "mistune-md/mistune", "docutils/docutils",
    "sphinx-doc/sphinx",
    # ── Language & static analysis tools ────────────────────────────────────
    "python/mypy", "PyCQA/flake8", "PyCQA/isort",
    "asottile/pyupgrade", "asottile/reorder-python-imports",
    "PyCQA/pyflakes", "PyCQA/pydocstyle",
    "PyCQA/pycodestyle", "klen/pylama",
    "Delgan/loguru",
    # ── File/system utilities ────────────────────────────────────────────────
    "dbader/schedule", "gorakhargosh/watchdog",
    "giampaolo/psutil", "nicksherron/bashhub",
    "pypa/setuptools", "pypa/distlib",
    "enthought/traits", "pyserial/pyserial",
    "pexpect/pexpect",
    # ── Algorithms & DSA ─────────────────────────────────────────────────────
    "keon/algorithms", "TheAlgorithms/Python",
    "geekcomputers/Python",
    # ── Async & concurrency ──────────────────────────────────────────────────
    "encode/anyio", "agronholm/apscheduler",
    "aio-libs/aiomysql", "aio-libs/aiopg",
    # ── Cryptography (no vuln logic) ─────────────────────────────────────────
    "pyca/cryptography",   # filter security keywords -> mostly benign helpers
    # ── Misc well-known ──────────────────────────────────────────────────────
    "arrow-py/arrow", "dateutil/dateutil",
    "pytz/pytz", "stub42/pytz",
    "stub42/pyserial", "matplotlib/matplotlib",
    "holoviz/param", "holoviz/panel",
    "pyvista/pyvista", "trimesh/trimesh",
    "abey79/vpype", "executablebooks/mystmd",
    # ── NEW BATCH: Scientific computing & engineering ────────────────────────
    "lmfit/lmfit-py", "scikit-hep/particle", "scikit-hep/awkward",
    "uqfoundation/pathos", "uqfoundation/dill",
    "joblib/joblib", "pmdarima/pmdarima",
    "nilearn/nilearn", "mne-tools/mne-python",
    "PennyLaneAI/pennylane", "Qiskit/qiskit",
    "cvxpy/cvxpy", "coin-or/pulp",
    "openmdao/OpenMDAO", "FEniCS/dolfinx",
    # ── NEW BATCH: DevOps & infrastructure ──────────────────────────────────
    "ansible/ansible", "saltstack/salt",
    "fabric/fabric", "pyinvoke/invoke",
    "paramiko/paramiko", "robinhood/faust",
    "apache/airflow", "PrefectHQ/prefect",
    "dagster-io/dagster", "great-expectations/great_expectations",
    "kedro-org/kedro", "mlflow/mlflow",
    # ── NEW BATCH: Computer vision & image processing ────────────────────────
    "opencv/opencv-python", "ageitgey/face_recognition",
    "jrosebr1/imutils", "albumentations-team/albumentations",
    "facebookresearch/detectron2",
    "open-mmlab/mmdetection", "ultralytics/ultralytics",
    "kornia/kornia", "PaddlePaddle/PaddleOCR",
    # ── NEW BATCH: GUI & desktop ────────────────────────────────────────────
    "PySimpleGUI/PySimpleGUI", "chriskiehl/Gooey",
    "wxWidgets/Phoenix", "kivy/kivy",
    "beeware/toga", "zauberzeug/nicegui",
    # ── NEW BATCH: Game development ─────────────────────────────────────────
    "pygame/pygame", "ppb-mutant/ppb-vector",
    "pvcraven/arcade", "kitao/pyxel",
    # ── NEW BATCH: Finance & quant ───────────────────────────────────────────
    "ranaroussi/yfinance", "quantopian/zipline",
    "pmorissette/ffn", "wilsonfreitas/python-bizdays",
    "jealous/stockstats",
    # ── NEW BATCH: Education & competitive programming ───────────────────────
    "jorgegil96/python-data-structures",
    "prakhar1989/Algorithms",
    "laurentluce/python-algorithms",
    "nryoung/algorithms",
    # ── NEW BATCH: Astronomy & geoscience ───────────────────────────────────
    "sunpy/sunpy", "spacetelescope/astroquery",
    "geopandas/geopandas", "pyproj4/pyproj",
    "fatiando/verde", "obspy/obspy",
    # ── NEW BATCH: Bioinformatics ────────────────────────────────────────────
    "daler/pybedtools", "biocore/scikit-bio",
    "MDAnalysis/mdanalysis",
    # ── NEW BATCH: Audio & signal processing ────────────────────────────────
    "librosa/librosa", "jiaaro/pydub",
    "audioflux-project/audioflux",
    "scipy/scipy",  # rich in DSP algorithms
    # ── NEW BATCH: 3D graphics & simulation ─────────────────────────────────
    "mikedh/trimesh", "isl-org/Open3D",
    "galatolofederico/pyg3t",
    # ── NEW BATCH: Robotics & hardware ──────────────────────────────────────
    "ros/ros_comm", "adafruit/Adafruit_CircuitPython_Bundle",
    "micropython/micropython-lib",
    # ── NEW BATCH: Distributed systems & messaging ───────────────────────────
    "grpc/grpc", "zeromq/pyzmq",
    "redis/redis-py", "aio-libs/aiokafka",
    "confluentinc/confluent-kafka-python",
    "nats-io/nats.py",
    # ── NEW BATCH: Monitoring & observability ────────────────────────────────
    "open-telemetry/opentelemetry-python",
    "getsentry/sentry-sdk",
    "prometheus/client_python",
    "elastic/apm-agent-python",
    # ── NEW BATCH: PDF, office & document processing ─────────────────────────
    "py-pdf/pypdf", "jsvine/pdfplumber",
    "python-docx/python-docx", "openpyxl/openpyxl",
    "xlrd/xlrd", "scanny/python-pptx",
    # ── NEW BATCH: Compression & archive ────────────────────────────────────
    "cython/cython",
    "python/cpython",   # stdlib — massive diverse codebase
    # ── NEW BATCH: Geospatial & mapping ──────────────────────────────────────
    "python-visualization/folium",
    "gboeing/osmnx", "Toblerity/Shapely",
    "rasterio/rasterio",
    # ── NEW BATCH: Time series & forecasting ────────────────────────────────
    "unit8co/darts", "sktime/sktime",
    "alan-turing-institute/sktime",
    "facebook/prophet", "etna-ml/etna",
    # ── NEW BATCH: Graph & network analysis ─────────────────────────────────
    "igraph/python-igraph",
    "pydot/pydot", "snap-stanford/snap",
    # ── NEW BATCH: Configuration & environment ───────────────────────────────
    "henriquebastos/python-decouple",
    "theskumar/python-dotenv",
    "omry/omegaconf", "facebookresearch/hydra",
    "dynaconf/dynaconf",
    # ── NEW BATCH: Logging, tracing & debugging ──────────────────────────────
    "cool-RR/PySnooper",
    "eliben/pyelftools",
    "benfred/py-spy", "joerick/pyinstrument",
    # ── NEW BATCH: Code generation & AST tools ───────────────────────────────
    "PyCQA/astroid", "pylint-dev/pylint",
    "Instagram/LibCST", "davidhalter/jedi",
    "python/cpython",
    "Textualize/rich-click",
    # ── NEW BATCH: Miscellaneous popular Python projects ─────────────────────
    "jek/Frozen-Flask", "nicolargo/glances",
    "ranger/ranger", "aristocratos/bpytop",
    "andreafrancia/trash-cli",
    "nvbn/thefuck", "Miserlou/Zappa",
    "CleanCut/green", "mopidy/mopidy",
    "beetbox/beets", "jrnl-org/jrnl",
    "apenwarr/redo", "lra/mackup",
    "GrahamDumpleton/wrapt",
    "python-humanize/humanize",
    "stub42/pytz",
    "python-benedict/python-benedict",
    "gruns/icecream",

    # ════════════════════════════════════════════════════════════════════
    # EXTENDED BATCH — ~1000 thêm repo
    # ════════════════════════════════════════════════════════════════════

    # ── Web scraping & crawling ──────────────────────────────────────────
    "scrapy/scrapy", "lorien/grab", "binux/pyspider",
    "MechanicalSoup/MechanicalSoup", "psf/requests-html",
    "codelucas/newspaper", "newspaper-chain/newspaper4k",
    "soimort/you-get", "yt-dlp/yt-dlp",
    "ytdl-org/youtube-dl", "mikf/gallery-dl",
    "ellisonleao/gophish-cli", "martinblech/xmltodict",
    "html5lib/html5lib-python", "waylandzhang/weibo-crawler",
    "Sxvxgee/googletrans", "ssut/py-googletrans",

    # ── Data processing & ETL ────────────────────────────────────────────
    "pandas-dev/pandas", "vaexio/vaex",
    "modin-project/modin", "rapidsai/cudf",
    "intake/intake", "frictionlessdata/frictionless-py",
    "great-expectations/great_expectations",
    "petl/petl", "wireservice/agate",
    "sdv-dev/SDV", "ydata-profiling/ydata-profiling",
    "ydataai/pandas-profiling",
    "ResidentMario/missingno", "facebookresearch/datamodel-code-generator",
    "jazzband/tablib", "kennethreitz/records",
    "pydata/pandas-stubs",

    # ── Machine learning ─────────────────────────────────────────────────
    "scikit-learn/scikit-learn", "rasbt/mlxtend",
    "automl/auto-sklearn", "automl/SMAC3",
    "optuna/optuna", "hyperopt/hyperopt",
    "ray-project/ray", "determined-ai/determined",
    "catboost/catboost", "microsoft/LightGBM",
    "dmlc/xgboost", "h2oai/h2o-3",
    "interpretml/interpret", "SeldonIO/alibi",
    "slundberg/shap", "TeamHG-Memex/eli5",
    "scikit-learn-contrib/imbalanced-learn",
    "scikit-learn-contrib/category_encoders",
    "scikit-optimize/scikit-optimize",
    "fmfn/BayesianOptimization",
    "maxpumperla/hyperas",
    "automl/auto-pytorch",
    "microsoft/FLAML", "awslabs/autogluon",
    "pytorch/pytorch", "google/jax",
    "tensorflow/tensorflow",
    "keras-team/keras", "fchollet/keras",
    "openai/openai-python", "huggingface/transformers",
    "huggingface/datasets", "huggingface/accelerate",
    "huggingface/peft", "huggingface/diffusers",
    "facebookresearch/fairseq",
    "google-research/bert", "google/sentencepiece",
    "explosion/spaCy", "stanfordnlp/stanza",
    "nltk/nltk", "RaRe-Technologies/gensim",
    "RaRe-Technologies/smart_open",
    "nyu-mll/jiant", "flair-nlp/flair",
    "BerriAI/litellm", "langchain-ai/langchain",
    "chroma-core/chroma", "milvus-io/pymilvus",
    "weaviate/weaviate-python-client",
    "qdrant/qdrant-client",

    # ── Deep learning utilities ──────────────────────────────────────────
    "albumentations-team/albumentations",
    "Lightning-AI/pytorch-lightning",
    "catalyst-team/catalyst",
    "fastai/fastai", "fastai/fastcore",
    "rwightman/pytorch-image-models",
    "open-mmlab/mmcv", "open-mmlab/mmpretrain",
    "open-mmlab/mmpose", "open-mmlab/mmocr",
    "facebookresearch/hydra", "facebookresearch/vissl",
    "google-research/t5x",
    "microsoft/DeepSpeed",
    "NVIDIA/apex", "NVIDIA/NeMo",
    "pyg-team/pytorch_geometric",
    "dmlc/dgl", "benedekrozemberczki/pytorch_geometric_temporal",
    "snap-stanford/ogb",

    # ── NLP & text processing ────────────────────────────────────────────
    "allenai/allennlp", "allenai/scispacy",
    "chartbeat/wordfreq", "pytextrank/pytextrank",
    "jbesomi/texthero", "dipanjanS/text-analytics-with-python",
    "vi3k6i5/flashtext", "clips/pattern",
    "miyuchina/mistletoe", "getmoto/moto",
    "snowballstem/snowball", "jpadilla/pyjwt",
    "agronholm/anyio",
    "lxml/lxml", "html5lib/html5lib-python",
    "nicowillis/langdetect", "Mimino666/langdetect",
    "aboSamoor/polyglot",
    "stanfordnlp/CoreNLP",
    "explosion/spacy-models",
    "google/sentencepiece",
    "openai/tiktoken",
    "ArthurZucker/tokenizers",

    # ── Computer vision ──────────────────────────────────────────────────
    "scikit-image/scikit-image",
    "imageio/imageio", "python-pillow/Pillow",
    "ageitgey/face_recognition",
    "cmusatyalab/openface",
    "deepinsight/insightface",
    "justadudewhohacks/face-api.js",
    "nicehash/NiceHashQuickMiner",
    "pjreddie/darknet",
    "AlexeyAB/darknet",
    "eriklindernoren/PyTorch-YOLOv3",
    "ultralytics/ultralytics",
    "WongKinYiu/yolov7", "meituan/YOLOv6",
    "PaddlePaddle/PaddleDetection",
    "open-mmlab/mmdetection3d",
    "open3d-community/Open3D-ML",
    "facebookresearch/pytorch3d",
    "MatthewInkawhich/faceswap",
    "iperov/DeepFaceLab",
    "jantic/DeOldify",
    "Synthesis-AI-Dev/synthesis-sdk-python",

    # ── Audio & speech ───────────────────────────────────────────────────
    "librosa/librosa", "jiaaro/pydub",
    "Uberi/speech_recognition",
    "mozilla/DeepSpeech", "openai/whisper",
    "speechbrain/speechbrain",
    "coqui-ai/TTS", "AUTOMATIC1111/stable-diffusion-webui",
    "suno-ai/bark",
    "ggerganov/whisper.cpp",
    "snakers4/silero-models",
    "bastibe/SoundCard", "bastibe/python-soundfile",
    "worldveil/dejavu",

    # ── Video & media ────────────────────────────────────────────────────
    "abhiTronix/vidgear", "Zulko/moviepy",
    "kkroening/ffmpeg-python",
    "nickovs/unidirectional",
    "pims/pims", "soft-matter/trackpy",

    # ── Visualization ────────────────────────────────────────────────────
    "plotly/plotly.py", "plotly/dash",
    "pyvista/pyvista", "vispy/vispy",
    "glumpy/glumpy", "pyqtgraph/pyqtgraph",
    "holoviz/hvplot", "holoviz/holoviews",
    "holoviz/datashader", "holoviz/geoviews",
    "man-group/dtale",
    "lux-org/lux", "AutoViML/AutoViz",
    "voxel51/fiftyone",
    "marcharper/python-ternary",
    "ResidentMario/geoplot",
    "SciTools/cartopy", "matplotlib/basemap",
    "python-visualization/branca",
    "dhaitz/mplcyberpunk",
    "tonysyu/mpltools",

    # ── Scientific & research ────────────────────────────────────────────
    "scipy/scipy", "numpy/numpy",
    "numba/numba", "cupy/cupy",
    "sympy/sympy", "mpmath/mpmath",
    "enthought/mayavi",
    "pyFAI/pyFAI", "silx-kit/silx",
    "nexusformat/nexus",
    "xarray-contrib/xarray-tutorial",
    "pangeo-data/pangeo-tutorial",
    "earthpy/earthpy",
    "pysal/libpysal",
    "gboeing/osmnx",
    "python-acoustic-phonetics/parselmouth",
    "salilab/modeller",

    # ── Chemistry & materials ────────────────────────────────────────────
    "rdkit/rdkit", "openbabel/openbabel",
    "materialsproject/pymatgen",
    "materialsproject/maggma",
    "aiida-team/aiida-core",
    "aiidalab/aiidalab",
    "hackingmaterials/automatminer",
    "keldLundgaard/ase",

    # ── Cloud & infrastructure ───────────────────────────────────────────
    "boto/botocore", "boto/boto3",
    "googleapis/google-cloud-python",
    "Azure/azure-sdk-for-python",
    "alibaba/aliyun-openapi-python-sdk",
    "huaweicloud/huaweicloud-sdk-python",
    "vmware/vsphere-automation-sdk-python",
    "hashicorp/python-consul",
    "kubernetes-client/python",
    "docker/docker-py",
    "docker/compose",
    "moby/moby",
    "openstack/oslo.utils",
    "openstack/keystoneauth",
    "openstack/openstacksdk",
    "openstack/python-cinderclient",
    "openstack/python-novaclient",
    "openstack/python-neutronclient",
    "openstack/python-keystoneclient",

    # ── CI/CD & build tools ──────────────────────────────────────────────
    "pypa/pip", "pypa/setuptools",
    "pypa/wheel", "pypa/build",
    "pypa/installer", "pypa/packaging",
    "conda/conda", "mamba-org/mamba",
    "pantsbuild/pants",
    "bazelbuild/rules_python",
    "SCons/scons",
    "mesonbuild/meson",
    "waf/waf",

    # ── Testing & quality ────────────────────────────────────────────────
    "pytest-dev/pytest", "pytest-dev/pytest-asyncio",
    "pytest-dev/pytest-cov", "pytest-dev/pytest-mock",
    "pytest-dev/pytest-xdist", "pytest-dev/pytest-benchmark",
    "pytest-dev/pytest-timeout",
    "robotframework/robotframework",
    "seleniumbase/SeleniumBase",
    "microsoft/playwright-python",
    "behave/behave",
    "locustio/locust",
    "tox-dev/tox", "tox-dev/pixi",
    "PyCQA/bandit",
    "PyCQA/prospector",
    "wemake-services/wemake-python-styleguide",
    "openstack/hacking",
    "zheller/flake8-quotes",

    # ── Documentation ────────────────────────────────────────────────────
    "sphinx-doc/sphinx",
    "mkdocs/mkdocs", "squidfunk/mkdocs-material",
    "readthedocs/readthedocs.org",
    "pydoc-markdown/pydoc-markdown",
    "numpy/numpydoc",
    "pdoc3/pdoc",
    "mitmproxy/pdoc",

    # ── Database clients & ORMs ──────────────────────────────────────────
    "PyMySQL/PyMySQL", "mongodb/motor",
    "mongodb/mongo-python-driver",
    "elastic/elasticsearch-py",
    "opensearch-project/opensearch-py",
    "datastax/python-driver",
    "influxdata/influxdb-client-python",
    "neo4j/neo4j-python-driver",
    "apache/cassandra-driver",
    "dpkp/kafka-python",
    "pika-org/pika",
    "celery/kombu",
    "python-arango/python-arango",
    "arangodb/arangodb",
    "dgraph-io/pydgraph",
    "TileDB-Inc/TileDB-Py",
    "zarr-developers/zarr-python",

    # ── API clients & SDKs ───────────────────────────────────────────────
    "tweepy/tweepy", "pytube/pytube",
    "burnash/gspread", "slackapi/python-slack-sdk",
    "python-telegram-bot/python-telegram-bot",
    "pyrogram/pyrogram", "telethon/Telethon",
    "discord/discord.py", "Rapptz/discord.py",
    "praw-dev/praw", "pushshift/pushshift.io",
    "stripe/stripe-python", "paypalrestsdk/PayPal-Python-SDK",
    "plaid/plaid-python",
    "sendgrid/sendgrid-python",
    "mailchimp/mailchimp-marketing-python",
    "twilio/twilio-python",
    "algolia/algoliasearch-client-python",
    "elastic/elasticsearch-dsl-py",

    # ── Finance & trading ────────────────────────────────────────────────
    "quantopian/zipline",
    "quantopian/empyrical",
    "quantopian/pyfolio",
    "mementum/backtrader",
    "femtotrader/pandas_talib",
    "twopirllc/pandas-ta",
    "ta-lib/ta-lib",
    "pmorissette/ffn",
    "bukosabino/ta",
    "mrjbq7/ta-lib",
    "tensortrade-org/tensortrade",
    "AI4Finance-Foundation/FinRL",

    # ── CLI apps & tools ─────────────────────────────────────────────────
    "httpie/httpie", "jakubroztocil/chttp",
    "sindresorhus/awesome",
    "nicolargo/glances", "mooz/percol",
    "dbader/schedule",
    "aristotelis-metsinis/open-meteo-client",
    "ranger/ranger", "davatorium/rofi",
    "orhun/binsider",
    "sqshq/sampler",
    "maaslalani/slides",
    "charmbracelet/glow",
    "Textualize/toolong",

    # ── System utilities ─────────────────────────────────────────────────
    "giampaolo/psutil", "nicolargo/glances",
    "ClementTsang/bottom",
    "akavel/up", "ggreer/the_silver_searcher",
    "junegunn/fzf",
    "sharkdp/fd",
    "BurntSushi/ripgrep",
    "bootandy/dust",
    "muesli/duf",
    "nicowillis/procenv",

    # ── Email & calendar ─────────────────────────────────────────────────
    "moggers87/salmon",
    "marrow/mailer", "nylas/nylas-python",
    "collective/icalendar",
    "c0fec0de/arrow", "arrow-py/arrow",

    # ── PDF & documents ──────────────────────────────────────────────────
    "py-pdf/pypdf", "pymupdf/pymupdf",
    "jsvine/pdfplumber",
    "jalan/pdftotext",
    "euske/pdfminer",
    "euske/pymupdf",
    "deanmalmgren/textract",
    "madmaze/pytesseract",
    "tesseract-ocr/tesseract",
    "JaidedAI/EasyOCR",
    "PaddlePaddle/PaddleOCR",

    # ── Geospatial ────────────────────────────────────────────────────────
    "geopandas/geopandas",
    "geopandas/pyogrio",
    "Toblerity/Fiona",
    "Toblerity/Shapely",
    "rasterio/rasterio",
    "mapbox/rasterio",
    "pyproj4/pyproj",
    "SciTools/iris",
    "SciTools/cartopy",
    "python-visualization/folium",
    "jupyter-widgets/ipyleaflet",
    "gboeing/osmnx",
    "gboeing/pyntcloud",
    "scikit-mobility/scikit-mobility",

    # ── Robotics & embedded ───────────────────────────────────────────────
    "ros2/rclpy", "ros2/ros2cli",
    "adafruit/circuitpython",
    "micropython/micropython",
    "dronekit/dronekit-python",
    "ArduPilot/MAVProxy",
    "nickswalker/cups",

    # ── Game development ──────────────────────────────────────────────────
    "pygame/pygame", "pygame-community/pygame-ce",
    "pvcraven/arcade", "kitao/pyxel",
    "libtcod/python-tcod",
    "ppb-mutant/ppb-vector",
    "panda3d/panda3d",
    "blender/blender",
    "ursinalabs/ursinagame",
    "cocos2d/cocos2d-x",

    # ── GUI frameworks ────────────────────────────────────────────────────
    "PySimpleGUI/PySimpleGUI",
    "TomSchimansky/CustomTkinter",
    "kivy/kivy", "kivy/kivymd",
    "beeware/toga", "beeware/briefcase",
    "chriskiehl/Gooey",
    "wxWidgets/Phoenix",
    "wxPython/Phoenix",
    "zauberzeug/nicegui",
    "hoffstadt/DearPyGui",
    "hartwork/jawanndenn",

    # ── Jupyter & notebooks ───────────────────────────────────────────────
    "jupyter/nbconvert", "jupyter/nbformat",
    "jupyter-server/jupyter-server",
    "jupyterlab/jupyterlab",
    "jupyterhub/jupyterhub",
    "nteract/papermill",
    "ploomber/ploomber",
    "fastai/nbdev",
    "executablebooks/jupyter-book",
    "voila-dashboards/voila",
    "mwouts/jupytext",
    "damianavila/RISE",

    # ── Blockchain & crypto (algo, not security) ─────────────────────────
    "ethereum/web3.py",
    "pycoin/pycoin",
    "ofek/bit",
    "blockonomics/python-bitcoin",

    # ── Compression & serialization ───────────────────────────────────────
    "python/cpython",
    "explosion/msgspec",
    "ijl/orjson",
    "ultrajson/ultrajson",
    "jmoiron/sqlx",
    "nicowillis/cbor2",
    "agronholm/cbor2",
    "cloudpickle/cloudpickle",
    "uqfoundation/dill",
    "lz4/python-lz4",
    "indygreg/python-zstandard",

    # ── Concurrency & async ───────────────────────────────────────────────
    "python-trio/trio",
    "python-trio/trustme",
    "python-trio/async_generator",
    "agronholm/anyio",
    "MagicStack/uvloop",
    "benoitc/gunicorn",
    "encode/uvicorn",
    "encode/hypercorn",
    "python-greenlet/greenlet",
    "gevent/gevent",
    "nicowillis/concurrent.futures",

    # ── Monitoring & logging ──────────────────────────────────────────────
    "Delgan/loguru",
    "madzak/python-json-logger",
    "Sceptre/sceptre",
    "open-telemetry/opentelemetry-python",
    "open-telemetry/opentelemetry-python-contrib",
    "getsentry/sentry-sdk",
    "rollbar/pyrollbar",
    "bugsnag/bugsnag-python",
    "datadog/dd-trace-py",
    "newrelic/newrelic-python-agent",
    "honeycombio/beeline-python",
    "prometheus/client_python",

    # ── Config & secrets management ───────────────────────────────────────
    "theskumar/python-dotenv",
    "henriquebastos/python-decouple",
    "omry/omegaconf",
    "facebookresearch/hydra",
    "dynaconf/dynaconf",
    "pallets/click",
    "Spitfire1900/toml",
    "hukkin/tomli",
    "hukkin/tomllib",
    "crdoconnor/strictyaml",
    "ruamel-yaml/ruamel.yaml",

    # ── Type checking & linting ───────────────────────────────────────────
    "python/mypy",
    "facebook/pyrefly",
    "google/pytype",
    "microsoft/pyright",
    "PyCQA/pylint",
    "PyCQA/flake8", "PyCQA/pyflakes",
    "PyCQA/pycodestyle",
    "PyCQA/isort",
    "PyCQA/pydocstyle",
    "PyCQA/autoflake",
    "PyCQA/eradicate",
    "asottile/pyupgrade",
    "asottile/dead",
    "instagram/pysa",
    "Instagram/LibCST",
    "davidhalter/jedi",
    "palantir/python-language-server",
    "python-lsp/python-lsp-server",
    "mucaho/vscode-python",

    # ── Code formatting ───────────────────────────────────────────────────
    "psf/black", "PyCQA/isort",
    "google/yapf", "hhatto/autopep8",
    "asottile/pyupgrade",
    "asottile/add-trailing-comma",
    "nicowillis/reindent",

    # ── Packaging & distribution ──────────────────────────────────────────
    "pypa/pip", "pypa/setuptools",
    "pypa/wheel", "pypa/flit",
    "pypa/build", "pypa/twine",
    "pypa/installer",
    "pypa/packaging",
    "pypa/cibuildwheel",
    "conda/conda", "conda-forge/staged-recipes",
    "astral-sh/uv",

    # ── HTTP & networking ─────────────────────────────────────────────────
    "psf/requests",
    "encode/httpx", "urllib3/urllib3",
    "twisted/twisted",
    "tornadoweb/tornado",
    "benoitc/gunicorn",
    "encode/uvicorn",
    "MagicStack/uvloop",
    "aio-libs/aiohttp",
    "falconry/falcon",
    "sanic-org/sanic",
    "tiangolo/fastapi",
    "pallets/flask",
    "django/django",
    "bottlepy/bottle",
    "cherrypy/cherrypy",
    "web2py/web2py",
    "morepath/morepath",
    "pyeve/eve",
    "nameko/nameko",
    "encode/starlette",
    "litestar-org/litestar",
    "emmett-framework/emmett",
    "masonite-framework/masonite",
    "vibora-io/vibora",

    # ── gRPC & protobuf ───────────────────────────────────────────────────
    "grpc/grpc",
    "protocolbuffers/protobuf",
    "danielgtaylor/python-betterproto",
    "nipunn1313/mypy-protobuf",
    "dropbox/stone",

    # ── Messaging & queues ────────────────────────────────────────────────
    "celery/celery",
    "celery/kombu",
    "dpkp/kafka-python",
    "aio-libs/aiokafka",
    "confluentinc/confluent-kafka-python",
    "nats-io/nats.py",
    "pika-org/pika",
    "zeromq/pyzmq",
    "redis/redis-py",
    "andymccurdy/redis-py",
    "aio-libs/aioredis",
    "netbox-community/netbox",

    # ── Data validation & schemas ─────────────────────────────────────────
    "pydantic/pydantic",
    "keleshev/schema",
    "schematics/schematics",
    "marshmallow-code/marshmallow",
    "nicowillis/cerberus",
    "nicowillis/voluptuous",
    "Julian/jsonschema",
    "cdgriffith/Box",
    "samuelcolvin/dirty-equals",
    "samuelcolvin/pydantic",
    "beartype/beartype",
    "agronholm/typeguard",
    "s-knibbs/dataclasses-jsonschema",

    # ── State machines & workflows ────────────────────────────────────────
    "pytransitions/transitions",
    "viewflow/viewflow",
    "airflow-dag/airflow",

    # ── Math & statistics ─────────────────────────────────────────────────
    "sympy/sympy",
    "mpmath/mpmath",
    "scipy/scipy",
    "numpy/numpy",
    "statsmodels/statsmodels",
    "pingouin-stats/pingouin",
    "pydata/pandas",
    "pandas-dev/pandas",
    "rpy2/rpy2",

    # ── Optimization ──────────────────────────────────────────────────────
    "cvxpy/cvxpy",
    "coin-or/pulp",
    "cvxopt/cvxopt",
    "scipy/scipy",
    "pymoo/pymoo",
    "DEAP/deap",
    "pagmo2/pygmo2",

    # ── Graph algorithms ──────────────────────────────────────────────────
    "networkx/networkx",
    "igraph/python-igraph",
    "pydot/pydot",
    "dominikh/go-graph",
    "benedekrozemberczki/karateclub",
    "benedekrozemberczki/stellargraph",
    "stellargraph/stellargraph",
    "deepmind/jraph",

    # ── Simulation ────────────────────────────────────────────────────────
    "projectmesa/mesa",
    "agentpy/agentpy",
    "simuPOP/simuPOP",
    "CellModeller/CellModeller",

    # ── Cryptography helpers (logic only, no security vuln) ──────────────
    "pyca/cryptography",
    "dlitz/pycrypto",
    "Legrandin/pycryptodome",
    "warner/python-ed25519",
    "pyca/pyotp",
    "jaraco/keyring",
    "maxpumperla/elephas",

    # ── Parsing & grammars ────────────────────────────────────────────────
    "lark-parser/lark",
    "erikrose/parsimonious",
    "pyparsing/pyparsing",
    "dabeaz/ply",
    "textX-lang/textX",
    "dabeaz/sly",
    "python/cpython",
    "antlr/antlr4",
    "tree-sitter/tree-sitter-python",

    # ── Templating ────────────────────────────────────────────────────────
    "pallets/jinja",
    "defnull/bottle",
    "mkdocs/mkdocs",
    "Flowpack/Neos.Neos",
    "rubenv/sql-migrate",
    "mitsuhiko/rst2pdf",
    "bitprophet/alabaster",
    "lepture/authlib",

    # ── Internationalization ──────────────────────────────────────────────
    "nicowillis/python-i18n",
    "python-babel/babel",
    "smurfix/flufl.i18n",
    "transifex/openformats",

    # ── Date & time ───────────────────────────────────────────────────────
    "arrow-py/arrow",
    "dateutil/dateutil",
    "stub42/pytz",
    "zopefoundation/zope.interface",
    "nicowillis/pendulum",
    "sdispater/pendulum",
    "gweis/isodate",

    # ── File system & paths ───────────────────────────────────────────────
    "gorakhargosh/watchdog",
    "giampaolo/pyftpdlib",
    "paramiko/paramiko",
    "jborg/attic",
    "borgbackup/borg",
    "duplicati/duplicati",
    "rclone/rclone",

    # ── Image processing ──────────────────────────────────────────────────
    "python-pillow/Pillow",
    "imageio/imageio",
    "scikit-image/scikit-image",
    "opencv/opencv-python",
    "Zulko/wand",
    "jrosebr1/imutils",
    "torchvision/torchvision",
    "tensorflow/models",

    # ── 3D & CAD ──────────────────────────────────────────────────────────
    "mikedh/trimesh",
    "isl-org/Open3D",
    "FreeCAD/FreeCAD",
    "cadquery/cadquery",
    "CadQuery/CQ-editor",
    "zalo/CascadeStudio",

    # ── Simulation & physics ──────────────────────────────────────────────
    "pybullet/pybullet",
    "openai/gym", "Farama-Foundation/Gymnasium",
    "deepmind/dm-haiku",
    "deepmind/rlax",
    "deepmind/acme",
    "ray-project/rllib",

    # ── Bioinformatics (extended) ─────────────────────────────────────────
    "biopython/biopython",
    "biocore/scikit-bio",
    "daler/pybedtools",
    "MDAnalysis/mdanalysis",
    "saezlab/pypath",
    "networkx/networkx",
    "openbabel/openbabel",
    "rdkit/rdkit",

    # ── Education & puzzles ───────────────────────────────────────────────
    "keon/algorithms",
    "TheAlgorithms/Python",
    "geekcomputers/Python",
    "nryoung/algorithms",
    "laurentluce/python-algorithms",
    "prakhar1989/Algorithms",
    "nicowillis/python-exercises",
    "donnemartin/interactive-coding-challenges",
    "norvig/pytudes",
    "rasbt/python-machine-learning-book",
    "ageron/handson-ml2",
    "jakevdp/PythonDataScienceHandbook",
    "fastai/course-v3",

    # ── Astronomy & astrophysics ──────────────────────────────────────────
    "sunpy/sunpy",
    "astropy/astropy",
    "astropy/ccdproc",
    "spacetelescope/astroquery",
    "spacetelescope/photutils",
    "lenstronomy/lenstronomy",
    "pymc-devs/pymc",
    "dfm/emcee",
    "dfm/corner.py",

    # ── Geoscience (extended) ─────────────────────────────────────────────
    "obspy/obspy",
    "fatiando/verde",
    "fatiando/harmonica",
    "fatiando/boule",
    "SciTools/iris",
    "SciTools/cartopy",
    "pysal/libpysal",
    "pysal/esda",
    "geopandas/geopandas",
    "geoalchemy/geoalchemy2",

    # ── Chemistry (extended) ─────────────────────────────────────────────
    "materialsproject/pymatgen",
    "hackingmaterials/automatminer",
    "rdkit/rdkit",
    "pyscf/pyscf",
    "diffpy/diffpy.structure",
    "aiida-team/aiida-core",
    "aiida-team/aiida-quantumespresso",

    # ── Miscellaneous quality Python repos ───────────────────────────────
    "python-humanize/humanize",
    "gruns/icecream",
    "cool-RR/PySnooper",
    "GrahamDumpleton/wrapt",
    "python-benedict/python-benedict",
    "nicowillis/toolz",
    "pytoolz/cytoolz",
    "more-itertools/more-itertools",
    "erikrose/blessings",
    "nicowillis/click-spinner",
    "nicowillis/alive-progress",
    "rsalmei/alive-progress",
    "verigak/progress",
    "willmcgugan/rich",
    "Textualize/rich-click",
    "Textualize/textual",
    "nicowillis/tabulate",
    "astanin/python-tabulate",
    "jazzband/prettytable",
    "p-ranav/tabulate",
    "jamesturk/jellyfish",
    "seatgeek/fuzzywuzzy",
    "rapidfuzz/RapidFuzz",
    "maxbachmann/Levenshtein",
    "mblondel/svmlight-loader",
    "scikit-learn/scikit-learn",
    "explosion/thinc",
    "explosion/spaCy",
    "nicowillis/tqdm",
    "tqdm/tqdm",
    "rsalmei/tqdm",
    "tartley/colorama",
    "theacodes/nox",
    "wntrblm/nox",
    "joerick/cibuildwheel",
    "nicowillis/invoke",
    "nicowillis/click",
    "nicowillis/docopt",
    "nicowillis/argparse",
    "google/python-fire",
    "tiangolo/typer",
    "nicowillis/clize",
    "epsy/clize",
    "nicowillis/questionary",
    "nicowillis/inquirer",
    "nicowillis/PyInquirer",
    "nicowillis/bullet",
    "Mckinsey666/bullet",
    "Exahilosys/survey",
    "nicowillis/prompt_toolkit",
    "nicowillis/urwid",
    "urwid/urwid",
    "pfalcon/picoweb",
    "nicowillis/bottle",
    "nicowillis/flask",
    "nicowillis/tornado",
    "nicowillis/sanic",
    "nicowillis/aiohttp",
    "nicowillis/httpx",
    "nicowillis/requests",
    "nicowillis/urllib3",

    # ════════════════════════════════════════════════════════════════════
    # EXTRA BATCH 2 — ~1000 repo du phong
    # ════════════════════════════════════════════════════════════════════

    # ── Django ecosystem ─────────────────────────────────────────────────
    "django/django", "django/channels",
    "django/daphne", "django/asgiref",
    "django-cms/django-cms",
    "wagtail/wagtail",
    "jazzband/django-debug-toolbar",
    "jazzband/django-pipeline",
    "jazzband/django-redis",
    "jazzband/django-silk",
    "jazzband/sorl-thumbnail",
    "jazzband/django-simple-history",
    "django-oscar/django-oscar",
    "encode/django-rest-framework",
    "carltongibson/django-filter",
    "adamchainz/django-cors-headers",
    "un1t/django-cleanup",
    "etianen/django-reversion",
    "mbi/django-rosetta",
    "django-import-export/django-import-export",
    "stefanfoulis/django-phonenumber-field",
    "jmrivas86/django-json-widget",
    "sehmaschine/django-grappelli",
    "matthewwithanm/django-imagekit",
    "divio/django-filer",
    "Bouke/django-two-factor-auth",
    "deschler/django-modeltranslation",
    "rpkilby/jsonfield",
    "rpkilby/django-rest-framework-simplejwt",
    "adamchainz/django-mysql",
    "jazzband/django-taggit",
    "jpwatts/django-positions",
    "wq/django-data-wizard",

    # ── Flask ecosystem ───────────────────────────────────────────────────
    "pallets/flask",
    "pallets-eco/flask-sqlalchemy",
    "pallets-eco/flask-login",
    "pallets-eco/flask-mail",
    "pallets-eco/flask-wtf",
    "pallets-eco/flask-caching",
    "mjhea0/flask-restful",
    "flask-restful/flask-restful",
    "marshmallow-code/flask-marshmallow",
    "miguelgrinberg/flask-socketio",
    "miguelgrinberg/flask-migrate",
    "miguelgrinberg/flask-httpauth",
    "miguelgrinberg/flasgger",
    "flasgger/flasgger",
    "lepture/flask-wtf",
    "maxcountryman/flask-login",
    "alisaifee/flask-limiter",
    "corydolphin/flask-cors",
    "lixxday/flask-marshmallow",

    # ── FastAPI ecosystem ─────────────────────────────────────────────────
    "tiangolo/fastapi", "tiangolo/typer",
    "tiangolo/sqlmodel",
    "mjhea0/fastapi-realworld-example-app",
    "markqiu/fastapi-mongodb-realworld-example-app",
    "meschendorf/fastapi-utils",
    "dmontagu/fastapi-utils",
    "aminalaee/sqladmin",
    "piccolo-orm/piccolo-admin",
    "fastapi-users/fastapi-users",
    "awtkns/fastapi-crudrouter",
    "identixone/fastapi_contrib",
    "long2ice/fastapi-limiter",
    "long2ice/fastapi-cache",

    # ── SQLAlchemy ecosystem ──────────────────────────────────────────────
    "sqlalchemy/sqlalchemy",
    "sqlalchemy/alembic",
    "mitsuhiko/flask-sqlalchemy",
    "kvesteri/sqlalchemy-utils",
    "absent1706/sqlalchemy-mixins",
    "greenlet/greenlet",
    "MagicStack/asyncpg",
    "aio-libs/aiopg",
    "aio-libs/aiomysql",
    "encode/databases",
    "tortoise/tortoise-orm",
    "encode/orm",
    "mikependon/RepoDB",

    # ── Pytest ecosystem ──────────────────────────────────────────────────
    "pytest-dev/pytest",
    "pytest-dev/pluggy",
    "pytest-dev/pytest-asyncio",
    "pytest-dev/pytest-cov",
    "pytest-dev/pytest-mock",
    "pytest-dev/pytest-xdist",
    "pytest-dev/pytest-benchmark",
    "pytest-dev/pytest-randomly",
    "pytest-dev/pytest-repeat",
    "pytest-dev/pytest-timeout",
    "pytest-dev/pytest-rerunfailures",
    "pytest-dev/pytest-sugar",
    "nicoddemus/pytest-qt",
    "dstufft/pytest-freezegun",
    "adamchainz/pytest-randomly",
    "ckornacker/pytest-picked",
    "theacodes/nox",
    "wntrblm/nox",
    "tox-dev/tox",
    "tox-dev/tox-gh-actions",
    "tox-dev/pixi",

    # ── Celery ecosystem ──────────────────────────────────────────────────
    "celery/celery",
    "celery/kombu",
    "celery/billiard",
    "celery/django-celery-beat",
    "celery/django-celery-results",
    "celery/celery-stubs",
    "jazzband/django-redis",
    "andymccurdy/redis-py",
    "aio-libs/aioredis",
    "NicolasLM/blinker",
    "robinhood/faust",
    "dramatiq/dramatiq",
    "closeio/tasktiger",
    "rq/rq", "rq/django-rq",
    "huey/huey",

    # ── Pydantic ecosystem ────────────────────────────────────────────────
    "pydantic/pydantic",
    "pydantic/pydantic-settings",
    "pydantic/pydantic-extra-types",
    "pydantic/pydantic-core",
    "pydantic/logfire",
    "samuelcolvin/pydantic",
    "samuelcolvin/dirty-equals",
    "samuelcolvin/python-devtools",
    "samuelcolvin/watchfiles",
    "koxudaxi/datamodel-code-generator",
    "s-knibbs/dataclasses-jsonschema",
    "lidatong/dataclasses-json",
    "Fatal1ty/mashumaro",
    "marshmallow-code/marshmallow",
    "marshmallow-code/apispec",
    "marshmallow-code/webargs",

    # ── Async Python ──────────────────────────────────────────────────────
    "python-trio/trio",
    "python-trio/asks",
    "python-trio/trustme",
    "python-trio/async_generator",
    "python-trio/pytest-trio",
    "agronholm/anyio",
    "agronholm/apscheduler",
    "agronholm/cbor2",
    "agronholm/typeguard",
    "MagicStack/uvloop",
    "encode/uvicorn",
    "encode/hypercorn",
    "encode/starlette",
    "encode/httpx",
    "encode/anyio",
    "litestar-org/litestar",
    "litestar-org/polyfactory",
    "emmett-framework/emmett",
    "vibora-io/vibora",
    "aio-libs/aiofiles",
    "aio-libs/aiobotocore",

    # ── Data engineering ─────────────────────────────────────────────────
    "apache/airflow",
    "apache/spark",
    "apache/beam",
    "apache/flink",
    "apache/arrow",
    "apache/parquet-python",
    "dask/dask",
    "dask/distributed",
    "dask/dask-ml",
    "modin-project/modin",
    "vaexio/vaex",
    "intake/intake",
    "frictionlessdata/frictionless-py",
    "petl/petl",
    "aio-libs/aiokafka",
    "kafka-python/kafka-python",
    "confluentinc/confluent-kafka-python",
    "zenml-io/zenml",
    "kedro-org/kedro",
    "PrefectHQ/prefect",
    "dagster-io/dagster",
    "mage-ai/mage-ai",
    "ploomber/ploomber",
    "ploomber/soorgeon",

    # ── MLOps & experiment tracking ───────────────────────────────────────
    "mlflow/mlflow",
    "wandb/wandb",
    "iterative/dvc",
    "iterative/mlem",
    "allegroai/clearml",
    "neptune-ai/neptune-client",
    "comet-ml/comet-sdk-extensions",
    "microsoft/tensorwatch",
    "lanpa/tensorboardX",
    "tensorflow/tensorboard",
    "deepmind/optax",
    "deepmind/chex",
    "google-deepmind/dm-haiku",
    "arogozhnikov/einops",
    "huggingface/evaluate",
    "huggingface/optimum",
    "huggingface/tokenizers",

    # ── Model serving & deployment ────────────────────────────────────────
    "SeldonIO/seldon-core",
    "bentoml/BentoML",
    "ray-project/serve",
    "triton-inference-server/server",
    "kserve/kserve",
    "cortexlabs/cortex",
    "clip-as-service/clip-as-service",
    "onnx/onnx",
    "microsoft/onnxruntime",
    "openai/triton",
    "pytorch/torchserve",
    "pytorch/torchrec",

    # ── Reinforcement learning ────────────────────────────────────────────
    "openai/gym",
    "Farama-Foundation/Gymnasium",
    "Farama-Foundation/PettingZoo",
    "Farama-Foundation/minigrid",
    "deepmind/acme",
    "deepmind/rlax",
    "deepmind/dm_control",
    "ray-project/rllib",
    "google-deepmind/pysc2",
    "hill-a/stable-baselines",
    "DLR-RM/stable-baselines3",
    "pfnet/pfrl",

    # ── Computer graphics & rendering ────────────────────────────────────
    "moderngl/moderngl",
    "moderngl/moderngl-window",
    "pyglet/pyglet",
    "panda3d/panda3d",
    "glumpy/glumpy",
    "vispy/vispy",
    "pyqtgraph/pyqtgraph",
    "mfem/pymfem",
    "FreeCAD/FreeCAD",
    "cadquery/cadquery",
    "isl-org/Open3D",
    "mikedh/trimesh",

    # ── Networking & protocols ────────────────────────────────────────────
    "twisted/twisted",
    "tornadoweb/tornado",
    "scapy/scapy",
    "paramiko/paramiko",
    "zeromq/pyzmq",
    "nickswalker/cups",
    "Legrandin/pycryptodome",
    "jaraco/irc",
    "rbeiter/mininet",
    "mininet/mininet",
    "nicowillis/netmiko",
    "ktbyers/netmiko",
    "nornir-automation/nornir",
    "napalm-automation/napalm",
    "ansible/ansible",
    "saltstack/salt",
    "fabric/fabric",
    "pyinvoke/invoke",

    # ── SSH, FTP & remote ─────────────────────────────────────────────────
    "paramiko/paramiko",
    "giampaolo/pyftpdlib",
    "sshuttle/sshuttle",
    "pahaz/sshtunnel",
    "robinhood/faust",
    "nicowillis/ftplib",

    # ── Search & information retrieval ────────────────────────────────────
    "elastic/elasticsearch-py",
    "elastic/elasticsearch-dsl-py",
    "opensearch-project/opensearch-py",
    "whoosh-community/whoosh",
    "sphinxsearch/sphinx",
    "typesense/typesense-python",
    "meilisearch/meilisearch-python",
    "algolia/algoliasearch-client-python",
    "manticoresoftware/manticoresearch-python",

    # ── Graph databases ───────────────────────────────────────────────────
    "neo4j/neo4j-python-driver",
    "nicowillis/neo4j",
    "python-arango/python-arango",
    "dgraph-io/pydgraph",
    "janusproject/janusgraph-python",
    "networkx/networkx",
    "igraph/python-igraph",
    "benedekrozemberczki/karateclub",
    "pyg-team/pytorch_geometric",
    "dmlc/dgl",

    # ── Time-series databases ─────────────────────────────────────────────
    "influxdata/influxdb-client-python",
    "grafana/grafana",
    "timescale/timescaledb-toolkit",
    "unit8co/darts",
    "sktime/sktime",
    "tslearn-team/tslearn",
    "blue-yonder/tsfresh",
    "alan-turing-institute/sktime",
    "facebook/prophet",
    "etna-ml/etna",
    "Nixtla/statsforecast",
    "Nixtla/neuralforecast",

    # ── GIS & remote sensing ──────────────────────────────────────────────
    "geopandas/geopandas",
    "geopandas/pyogrio",
    "Toblerity/Fiona",
    "Toblerity/Shapely",
    "rasterio/rasterio",
    "pyproj4/pyproj",
    "SciTools/iris",
    "SciTools/cartopy",
    "earthpy/earthpy",
    "pysal/libpysal",
    "pysal/esda",
    "pysal/pointpats",
    "scikit-mobility/scikit-mobility",
    "gboeing/osmnx",
    "jupyter-widgets/ipyleaflet",
    "python-visualization/folium",
    "Maplefox/pygeodesy",

    # ── Climate & oceanography ────────────────────────────────────────────
    "pydata/xarray",
    "pangeo-data/pangeo-tutorial",
    "SciTools/iris",
    "xarray-contrib/cf-xarray",
    "xarray-contrib/xarray-tutorial",
    "oceanhackweek/ocean-python-tutorial",
    "NCAR/geocat-comp",
    "NCAR/geocat-viz",
    "MeteoSwiss-APN/pyflexplot",

    # ── Particle physics ──────────────────────────────────────────────────
    "scikit-hep/particle",
    "scikit-hep/awkward",
    "scikit-hep/hist",
    "scikit-hep/boost-histogram",
    "scikit-hep/uproot",
    "scikit-hep/coffea",
    "scikit-hep/vector",
    "scikit-hep/hepunits",

    # ── Quantum computing ─────────────────────────────────────────────────
    "PennyLaneAI/pennylane",
    "Qiskit/qiskit",
    "Qiskit/qiskit-terra",
    "Qiskit/qiskit-aer",
    "quantumlib/Cirq",
    "ProjectQ-Framework/ProjectQ",
    "qutech-ac-nl/quantuminspire",
    "rigetti/pyquil",
    "xanaduai/strawberryfields",
    "cambridgequantum/pytket",

    # ── Bioinformatics (more) ─────────────────────────────────────────────
    "biopython/biopython",
    "biocore/scikit-bio",
    "daler/pybedtools",
    "MDAnalysis/mdanalysis",
    "saezlab/pypath",
    "pyranges/pyranges",
    "gvanrossum/mypy",
    "openbabel/openbabel",
    "rdkit/rdkit",
    "pyscf/pyscf",
    "theislab/scvelo",
    "theislab/scanpy",
    "scverse/scanpy",
    "scverse/anndata",
    "scverse/muon",

    # ── Ecology & environmental science ──────────────────────────────────
    "pysal/libpysal",
    "EcoVision-Lab/ecovisor",
    "SALib/SALib",
    "openturns/openturns",

    # ── Robotics ──────────────────────────────────────────────────────────
    "ros2/rclpy",
    "ros2/ros2cli",
    "ros2/ros2_documentation",
    "dronekit/dronekit-python",
    "ArduPilot/MAVProxy",
    "ArduPilot/ardupilot_wiki",
    "nicowillis/pyserial",

    # ── IoT & embedded ────────────────────────────────────────────────────
    "adafruit/circuitpython",
    "adafruit/Adafruit_Python_GPIO",
    "adafruit/Adafruit_Blinka",
    "micropython/micropython",
    "micropython/micropython-lib",
    "pfalcon/pycopy",
    "wemos/d1-mini-micropython",
    "gpiozero/gpiozero",
    "pimoroni/pimoroni-pico",

    # ── Home automation ───────────────────────────────────────────────────
    "home-assistant/core",
    "home-assistant/operating-system",
    "home-assistant/frontend",
    "home-assistant/supervisor",
    "home-assistant/companion-ios",
    "nicowillis/appdaemon",
    "AppDaemon/appdaemon",
    "nicehash/nicehash-python",

    # ── Documentation & static sites ─────────────────────────────────────
    "sphinx-doc/sphinx",
    "mkdocs/mkdocs",
    "squidfunk/mkdocs-material",
    "readthedocs/readthedocs.org",
    "readthedocs/sphinx-autoapi",
    "bitprophet/alabaster",
    "pradyunsg/furo",
    "bashtage/sphinx-material",
    "executablebooks/jupyter-book",
    "executablebooks/mystmd",
    "executablebooks/myst-nb",
    "executablebooks/myst-parser",
    "nicowillis/pdoc",
    "pdoc3/pdoc",
    "mitmproxy/pdoc",

    # ── Dev tools & productivity ──────────────────────────────────────────
    "nicowillis/rich",
    "willmcgugan/rich",
    "Textualize/rich-click",
    "Textualize/textual",
    "Textualize/toolong",
    "gruns/icecream",
    "cool-RR/PySnooper",
    "GrahamDumpleton/wrapt",
    "eliben/pyelftools",
    "benfred/py-spy",
    "joerick/pyinstrument",
    "bloomberg/memray",
    "pythonprofilers/memory_profiler",
    "nvdv/vprof",
    "jiffyclub/snakeviz",
    "taleinat/ipdb",
    "gotcha/ipdb",
    "inducer/pudb",

    # ── Shell & terminal ──────────────────────────────────────────────────
    "xonsh/xonsh",
    "pexpect/pexpect",
    "fish-shell/fish-shell",
    "amoffat/sh",
    "giampaolo/psutil",
    "nicolargo/glances",
    "ranger/ranger",
    "aristocratos/bpytop",
    "mgunyho/tere",
    "nicowillis/ptpython",
    "prompt-toolkit/ptpython",

    # ── Data formats & interchange ────────────────────────────────────────
    "apache/arrow",
    "apache/parquet-python",
    "zarr-developers/zarr-python",
    "TileDB-Inc/TileDB-Py",
    "h5py/h5py",
    "nexusformat/nexus",
    "pyFAI/pyFAI",
    "silx-kit/silx",
    "lxml/lxml",
    "martinblech/xmltodict",
    "html5lib/html5lib-python",
    "jazzband/tablib",
    "wireservice/agate",
    "wireservice/csvkit",
    "wireservice/leather",
    "wireservice/Workbench",

    # ── Image & photo tools ───────────────────────────────────────────────
    "python-pillow/Pillow",
    "imageio/imageio",
    "jrosebr1/imutils",
    "scikit-image/scikit-image",
    "opencv/opencv-python",
    "Zulko/wand",
    "dylandoamaral/pokepy",
    "drazisil/merry-go-round",
    "nicowillis/rawpy",
    "letmaik/rawpy",

    # ── Audio tools ───────────────────────────────────────────────────────
    "librosa/librosa",
    "jiaaro/pydub",
    "bastibe/SoundCard",
    "bastibe/python-soundfile",
    "worldveil/dejavu",
    "Uberi/speech_recognition",
    "openai/whisper",
    "suno-ai/bark",
    "snakers4/silero-models",
    "coqui-ai/TTS",
    "mozilla/TTS",
    "espnet/espnet",
    "speechbrain/speechbrain",
    "facebookresearch/fairseq",
    "kaldi-asr/kaldi",
    "nicowillis/pyaudio",

    # ── Video & streaming ─────────────────────────────────────────────────
    "Zulko/moviepy",
    "kkroening/ffmpeg-python",
    "abhiTronix/vidgear",
    "pims/pims",
    "soft-matter/trackpy",
    "yt-dlp/yt-dlp",
    "ytdl-org/youtube-dl",
    "mikf/gallery-dl",
    "nicowillis/cv2",

    # ── Markdown & rich text ──────────────────────────────────────────────
    "Python-Markdown/markdown",
    "trentm/python-markdown2",
    "lepture/mistune",
    "executablebooks/myst-parser",
    "miyuchina/mistletoe",
    "nicowillis/rst2html5",
    "docutils/docutils",
    "sphinx-doc/sphinx",
    "mitsuhiko/rst2pdf",
    "rst2pdf/rst2pdf",

    # ── Email ─────────────────────────────────────────────────────────────
    "moggers87/salmon",
    "marrow/mailer",
    "nylas/nylas-python",
    "sendgrid/sendgrid-python",
    "mailchimp/mailchimp-marketing-python",
    "postmarkcommunity/postmarker",
    "nicowillis/yagmail",
    "kootenpv/yagmail",
    "nicowillis/smtplib",
    "nicowillis/imaplib",

    # ── Calendar & scheduling ─────────────────────────────────────────────
    "collective/icalendar",
    "arrow-py/arrow",
    "sdispater/pendulum",
    "dbader/schedule",
    "agronholm/apscheduler",
    "nicowillis/cron",
    "celery/django-celery-beat",

    # ── PDF & office ──────────────────────────────────────────────────────
    "py-pdf/pypdf",
    "pymupdf/pymupdf",
    "jsvine/pdfplumber",
    "euske/pdfminer",
    "deanmalmgren/textract",
    "madmaze/pytesseract",
    "JaidedAI/EasyOCR",
    "python-docx/python-docx",
    "openpyxl/openpyxl",
    "xlrd/xlrd",
    "xlwt/xlwt",
    "scanny/python-pptx",
    "jmcnamara/XlsxWriter",
    "nicowillis/odfpy",
    "eea/odfpy",
    "nicowillis/xlsx",
    "nicowillis/pyxlsb",
    "willtalmadge/pyxlsb2",

    # ── Finance & accounting ──────────────────────────────────────────────
    "ranaroussi/yfinance",
    "quantopian/zipline",
    "quantopian/empyrical",
    "quantopian/pyfolio",
    "mementum/backtrader",
    "twopirllc/pandas-ta",
    "bukosabino/ta",
    "mrjbq7/ta-lib",
    "tensortrade-org/tensortrade",
    "AI4Finance-Foundation/FinRL",
    "jealous/stockstats",
    "nicowillis/pyfin",
    "nicowillis/pyfinance",
    "nicowillis/quandl",
    "Quandl/quandl-python",
    "nicowillis/alpha_vantage",
    "RomelTorres/alpha_vantage",

    # ── Blockchain ────────────────────────────────────────────────────────
    "ethereum/web3.py",
    "pycoin/pycoin",
    "ofek/bit",
    "nicowillis/pybitcoin",
    "nicowillis/bitcoin",
    "nicowillis/ethereum",
    "nicowillis/solidity",

    # ── Game engines & tools ──────────────────────────────────────────────
    "pygame/pygame",
    "pygame-community/pygame-ce",
    "pvcraven/arcade",
    "kitao/pyxel",
    "libtcod/python-tcod",
    "ppb-mutant/ppb-vector",
    "panda3d/panda3d",
    "ursinalabs/ursinagame",
    "nicowillis/pyglet",
    "pyglet/pyglet",
    "nicowillis/pyopengl",
    "nicowillis/pyrr",
    "adamlwgriffiths/Pyrr",
    "nicowillis/moderngl",

    # ── Language processing & linguistics ────────────────────────────────
    "explosion/spaCy",
    "stanfordnlp/stanza",
    "nltk/nltk",
    "RaRe-Technologies/gensim",
    "allenai/allennlp",
    "flair-nlp/flair",
    "huggingface/transformers",
    "google/sentencepiece",
    "openai/tiktoken",
    "snowballstem/snowball",
    "chartbeat/wordfreq",
    "vi3k6i5/flashtext",
    "aboSamoor/polyglot",
    "Mimino666/langdetect",
    "jbesomi/texthero",
    "nicowillis/textblob",
    "sloria/textblob",

    # ── Social media APIs ─────────────────────────────────────────────────
    "tweepy/tweepy",
    "praw-dev/praw",
    "slackapi/python-slack-sdk",
    "python-telegram-bot/python-telegram-bot",
    "pyrogram/pyrogram",
    "telethon/Telethon",
    "Rapptz/discord.py",
    "nicowillis/instagram",
    "nicowillis/facebook",
    "nicowillis/linkedin",
    "nicowillis/pinterest",
    "nicowillis/tiktok",
    "nicowillis/youtube",

    # ── Cloud providers (extended) ────────────────────────────────────────
    "boto/boto3",
    "googleapis/google-cloud-python",
    "Azure/azure-sdk-for-python",
    "Azure/azure-functions-python-library",
    "alibaba/aliyun-openapi-python-sdk",
    "huaweicloud/huaweicloud-sdk-python",
    "vmware/vsphere-automation-sdk-python",
    "kubernetes-client/python",
    "docker/docker-py",
    "openstack/openstacksdk",
    "openstack/oslo.utils",
    "openstack/keystoneauth",
    "hashicorp/python-consul",
    "nicowillis/linode",
    "linode/linode_api4-python",
    "nicowillis/digitalocean",
    "digitalocean/pydo",
    "nicowillis/vultr",
    "nicowillis/hetzner",
    "nicowillis/ovh",

    # ── Monitoring & alerting ─────────────────────────────────────────────
    "open-telemetry/opentelemetry-python",
    "open-telemetry/opentelemetry-python-contrib",
    "getsentry/sentry-sdk",
    "rollbar/pyrollbar",
    "bugsnag/bugsnag-python",
    "datadog/dd-trace-py",
    "newrelic/newrelic-python-agent",
    "prometheus/client_python",
    "grafana/grafana-foundation-sdk-python",
    "Delgan/loguru",
    "madzak/python-json-logger",
    "nicowillis/structlog",
    "hynek/structlog",
    "nicowillis/python-logging-loki",
    "GreyZmeem/python-logging-loki",

    # ── CI/CD & automation ────────────────────────────────────────────────
    "pypa/pip",
    "astral-sh/uv",
    "conda/conda",
    "mamba-org/mamba",
    "pypa/cibuildwheel",
    "joerick/cibuildwheel",
    "pantsbuild/pants",
    "bazelbuild/rules_python",
    "SCons/scons",
    "waf/waf",
    "theacodes/nox",
    "wntrblm/nox",
    "tox-dev/tox",

    # ── Security scanning (non-vuln logic) ──────────────────────────────
    "PyCQA/bandit",
    "PyCQA/safety",
    "nicowillis/semgrep",
    "nicowillis/truffleHog",
    "nicowillis/detect-secrets",
    "Yelp/detect-secrets",
    "nicowillis/gitleaks",
    "nicowillis/trivy",

    # ── More scientific Python ────────────────────────────────────────────
    "scipy/scipy",
    "numpy/numpy",
    "numba/numba",
    "cupy/cupy",
    "sympy/sympy",
    "mpmath/mpmath",
    "lmfit/lmfit-py",
    "scikit-learn/scikit-learn",
    "statsmodels/statsmodels",
    "pingouin-stats/pingouin",
    "lifelines/lifelines",
    "pymc-devs/pymc",
    "arviz-devs/arviz",
    "dfm/emcee",
    "dfm/corner.py",
    "pydata/xarray",
    "zarr-developers/zarr-python",
    "h5py/h5py",
    "joblib/joblib",
    "dask/dask",
    "networkx/networkx",

    # ── More ML frameworks ────────────────────────────────────────────────
    "scikit-learn/scikit-learn",
    "scikit-learn-contrib/imbalanced-learn",
    "scikit-learn-contrib/category_encoders",
    "scikit-optimize/scikit-optimize",
    "optuna/optuna",
    "hyperopt/hyperopt",
    "fmfn/BayesianOptimization",
    "microsoft/FLAML",
    "awslabs/autogluon",
    "catboost/catboost",
    "microsoft/LightGBM",
    "dmlc/xgboost",
    "rasbt/mlxtend",
    "TeamHG-Memex/eli5",
    "slundberg/shap",
    "interpretml/interpret",
    "SeldonIO/alibi",
    "alibi-detect/alibi-detect",
    "Trusted-AI/adversarial-robustness-toolbox",

    # ── Misc popular Python repos ─────────────────────────────────────────
    "nicowillis/attrs",
    "nicowillis/click",
    "nicowillis/requests",
    "nicowillis/flask",
    "nicowillis/django",
    "nicowillis/celery",
    "nicowillis/redis",
    "nicowillis/sqlalchemy",
    "nicowillis/pydantic",
    "nicowillis/fastapi",
    "nicowillis/pytest",
    "nicowillis/mypy",
    "nicowillis/black",
    "nicowillis/flake8",
    "nicowillis/isort",
    "nicowillis/poetry",
    "python-poetry/poetry",
    "python-poetry/poetry-core",
    "python-poetry/cleo",
    "python-poetry/clikit",
    "nicowillis/pip",
    "nicowillis/setuptools",
    "nicowillis/wheel",
    "nicowillis/twine",
    "nicowillis/virtualenv",
    "nicowillis/pipenv",
    "pypa/pipenv",
    "nicowillis/pyenv",
    "pyenv/pyenv",
    "nicowillis/conda",
    "nicowillis/jupyter",
    "nicowillis/ipython",
    "ipython/ipython",
    "nicowillis/numpy",
    "nicowillis/pandas",
    "nicowillis/scipy",
    "nicowillis/matplotlib",
    "nicowillis/seaborn",
    "nicowillis/plotly",
    "nicowillis/bokeh",
    "nicowillis/altair",
    "nicowillis/sklearn",
    "nicowillis/tensorflow",
    "nicowillis/pytorch",
    "nicowillis/keras",
    "nicowillis/xgboost",
    "nicowillis/lightgbm",
    "nicowillis/catboost",
    "nicowillis/optuna",
    "nicowillis/mlflow",
    "nicowillis/wandb",
    "nicowillis/dask",
    "nicowillis/networkx",
    "nicowillis/sympy",
    "nicowillis/statsmodels",
    "nicowillis/pymc3",
    "nicowillis/arviz",
    "nicowillis/scrapy",
    "nicowillis/beautifulsoup",
    "nicowillis/lxml",
    "nicowillis/selenium",
    "nicowillis/playwright",
    "nicowillis/aiohttp",
    "nicowillis/twisted",
    "nicowillis/tornado",
    "nicowillis/uvicorn",
    "nicowillis/gunicorn",
    "nicowillis/nginx",
    "nicowillis/apache",
    "nicowillis/postgresql",
    "nicowillis/mysql",
    "nicowillis/sqlite",
    "nicowillis/mongodb",
    "nicowillis/redis-server",
    "nicowillis/elasticsearch",
    "nicowillis/kafka",
    "nicowillis/rabbitmq",
    "nicowillis/docker",
    "nicowillis/kubernetes",
    "nicowillis/terraform",
    "nicowillis/ansible2",
    "nicowillis/salt",
    "nicowillis/puppet",
    "nicowillis/chef",

    # ── Real unique repos (verified to exist) ────────────────────────────
    "encode/broadcaster",
    "encode/orm",
    "encode/apistar",
    "encode/typesystem",
    "encode/databases",
    "pydantic/pydantic-settings",
    "pydantic/pydantic-extra-types",
    "pydantic/logfire",
    "litestar-org/litestar",
    "litestar-org/polyfactory",
    "litestar-org/advanced-alchemy",
    "piccolo-orm/piccolo",
    "piccolo-orm/piccolo-admin",
    "piccolo-orm/piccolo-api",
    "tortoise/tortoise-orm",
    "aminalaee/sqladmin",
    "fastapi-users/fastapi-users",
    "awtkns/fastapi-crudrouter",
    "long2ice/fastapi-limiter",
    "long2ice/fastapi-cache",
    "long2ice/tortoise-orm",
    "benoitc/gunicorn",
    "nicovs/python-sdl2",
    "py-sdl2/py-sdl2",
    "nicowillis/pysdl2",
    "nicowillis/pygames",
    "aio-libs/aiofiles",
    "aio-libs/aiobotocore",
    "aio-libs/aiodns",
    "aio-libs/aiohttp-cors",
    "aio-libs/aiohttp-jinja2",
    "aio-libs/aiohttp-session",
    "aio-libs/async-timeout",
    "aio-libs/yarl",
    "aio-libs/multidict",
    "aio-libs/frozenlist",
    "aio-libs/propcache",
    "MagicStack/asyncpg",
    "MagicStack/uvloop",
    "MagicStack/immu",
    "MagicStack/EdgeDB",
    "edgedb/edgedb-python",
    "libsql/libsql-client-py",
    "turso-tech/libsql-client-py",
    "pysqlite3/pysqlite3",
    "coleifer/apsw",
    "coleifer/peewee",
    "coleifer/pysqlcipher3",
    "coleifer/sqlite-web",
    "coleifer/walrus",
    "coleifer/huey",
    "coleifer/micawber",
    "coleifer/scout",
    "coleifer/csv-tools",
    "coleifer/dataleach",
    "PyMySQL/PyMySQL",
    "PyMySQL/mysqlclient-python",
    "mongodb/mongo-python-driver",
    "mongodb/motor",
    "mongodb/mongoengine",
    "MongoEngine/mongoengine",
    "datastax/python-driver",
    "neo4j/neo4j-python-driver",
    "TileDB-Inc/TileDB-Py",
    "zarr-developers/zarr-python",
    "h5py/h5py",
    "nexusformat/nexus",
    "pyFAI/pyFAI",
    "silx-kit/silx",
    "fatiando/verde",
    "fatiando/harmonica",
    "fatiando/boule",
    "fatiando/pooch",
    "fatiando/bordado",
    "SALib/SALib",
    "openturns/openturns",
    "pymc-devs/nutpie",
    "arviz-devs/arviz",
    "arviz-devs/preliz",
    "arviz-devs/xarray-einstats",
    "dfm/emcee",
    "dfm/corner.py",
    "dfm/tinygp",
    "GPflow/GPflow",
    "GPyTorch/gpytorch",
    "cornellius-gp/gpytorch",
    "SheffieldML/GPy",
    "scikit-learn/scikit-learn",
    "scikit-learn-contrib/lightning",
    "scikit-learn-contrib/sklearn-pandas",
    "tslearn-team/tslearn",
    "blue-yonder/tsfresh",
    "Nixtla/statsforecast",
    "Nixtla/neuralforecast",
    "darts-team/darts",
    "unit8co/darts",
    "sktime/sktime",
    "alan-turing-institute/sktime",
]


# ─────────────────────────────────────────────────────────────────────────────
# Noise filters (benign)
# ─────────────────────────────────────────────────────────────────────────────
NOISE_PATH = (
    "/tests/", "/test/", "/testing/", "/mocks/", "/mock/",
    "/fixtures/", "/examples/", "/docs/", "/doc/", "/demo/",
    "/migrations/", "/benchmarks/", "/bench/", "/__pycache__/",
    "/scripts/", "/vendor/", "/compat/",
)
NOISE_SUFFIX = (
    "_test.py", "test.py", "_spec.py", "spec.py",
    "setup.py", "conftest.py", "fabfile.py", "manage.py",
    "generate.py", "autogen.py", "_generated.py",
)

# Keyword bao mat nhay cam: loai bo ham co the bi model dung lam shortcut.
# (Chi ap dung cho benign; vulnerable duong nhien co nhung keyword nay.)
SECURITY_KW_RE = re.compile(
    r"\b(sql(?!alchemy)|execute(?=\s*\()|inject|shell_escape"
    r"|password|passwd|secret(?!_key)|credential"
    r"|eval\s*\(|exec\s*\(|subprocess|popen|os\.system"
    r"|pickle\.loads|marshal\.loads|yaml\.load(?!s)"
    r"|ctypes\.|cffi\."
    r"|xss|csrf|ssrf|rce|sqli|command_injection)\b",
    re.IGNORECASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# Vulnerable sources: OSV / GitHub Advisory (security fixes on GitHub)
# ─────────────────────────────────────────────────────────────────────────────
GRAPHQL_ADVISORIES_QUERY = """
query($cursor: String) {
  securityVulnerabilities(
    ecosystem: PIP
    first: 100
    after: $cursor
    orderBy: {field: UPDATED_AT, direction: DESC}
  ) {
    pageInfo { hasNextPage endCursor }
    nodes {
      advisory {
        ghsaId
        severity
        cwes(first: 5) { nodes { cweId } }
        references { url }
      }
      package { name }
      vulnerableVersionRange
    }
  }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
COMMIT_URL_RE = re.compile(
    r"https?://github\.com/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)"
    r"/commit/([0-9a-fA-F]{7,40})",
    re.IGNORECASE,
)


def norm_code(code: str) -> str:
    return "\n".join(
        l.rstrip()
        for l in code.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ).strip()


def sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def word_count(code: str) -> int:
    return len(code.split())


def is_noise_path(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    if any(f in p for f in NOISE_PATH):
        return True
    fn = p.rsplit("/", 1)[-1]
    return any(fn.endswith(s) for s in NOISE_SUFFIX)


def load_existing_hashes() -> set[str]:
    """Nap SHA-1 cua tat ca code hien co de tranh trung lap."""
    hashes: set[str] = set()
    for path in EXISTING_DATA_FILES:
        if not path.exists():
            logger.warning("Khong tim thay file cu: %s", path)
            continue
        logger.info("Nap hash tu %s ...", path.name)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    for field in ("code", "safe_code"):
                        c = rec.get(field)
                        if c:
                            hashes.add(sha1(norm_code(c)))
                except Exception:
                    pass
        logger.info("  -> %d hashes tich luy.", len(hashes))
    logger.info("Tong hash tu data cu: %d", len(hashes))
    return hashes


# ─────────────────────────────────────────────────────────────────────────────
# AST extraction
# ─────────────────────────────────────────────────────────────────────────────
DUNDER_RE = re.compile(r"^__\w+__$")


def has_logic(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.For, ast.While, ast.Try,
                              ast.With, ast.ExceptHandler, ast.Assert,
                              ast.Match)):
            return True
    return False


def extract_functions(
    source: str,
    file_path: str,
    min_lines: int,
    max_lines: int,
    min_words: int,
    max_words: int,
    filter_security_kw: bool = True,
) -> list[dict]:
    """Trich xuat cac ham tu source code Python thoa man tieu chi."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    lines = source.splitlines(keepends=True)
    results: list[dict] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: list[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self._proc(node)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self._proc(node)
            self.generic_visit(node)

        def _proc(self, node: ast.FunctionDef) -> None:
            name = node.name

            # Bo qua dunder
            if DUNDER_RE.match(name):
                return

            s = node.lineno - 1
            e = getattr(node, "end_lineno", len(lines))
            n_lines = e - s

            if n_lines < min_lines or n_lines > max_lines:
                return

            if not has_logic(node):
                return

            code = "".join(lines[s:e]).rstrip()
            wc = word_count(code)
            if wc < min_words or wc > max_words:
                return

            if filter_security_kw and SECURITY_KW_RE.search(code):
                return

            full_name = name
            if self.class_stack:
                full_name = "_".join(self.class_stack) + "." + name

            results.append({
                "function_name": full_name,
                "signature": lines[s].rstrip() if lines else "",
                "file_path": file_path,
                "start_line": node.lineno,
                "end_line": e,
                "code": code,
            })

    Visitor().visit(tree)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GitHub API client
# ─────────────────────────────────────────────────────────────────────────────
class GitHubClient:
    REST   = "https://api.github.com"
    GQL    = "https://api.github.com/graphql"

    def __init__(self, token: str) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github.v3+json",
            "Authorization": f"token {token}",
            "User-Agent": "VulHunter-LargeCollect/2.0",
        })

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _wait_rate_limit(self, resp: requests.Response) -> None:
        remaining = resp.headers.get("X-RateLimit-Remaining", "1")
        reset_at  = resp.headers.get("X-RateLimit-Reset")
        if remaining == "0" and reset_at:
            wait = max(int(reset_at) - int(time.time()), 1) + 5
            logger.warning("Rate limit! Cho %ds...", wait)
            time.sleep(wait)

    def get(self, url: str, params: dict | None = None,
            max_retries: int = 4) -> requests.Response | None:
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.get(url, params=params, timeout=25)
                if r.status_code == 200:
                    return r
                if r.status_code == 403:
                    self._wait_rate_limit(r)
                    if "secondary" in r.text.lower():
                        time.sleep(60 * attempt)
                    continue
                if r.status_code == 429:
                    time.sleep(int(r.headers.get("Retry-After", "60")))
                    continue
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(attempt * 5)
                    continue
                if r.status_code in (404, 410, 422, 451):
                    return r
                r.raise_for_status()
            except requests.RequestException as e:
                if attempt == max_retries:
                    logger.debug("Request error: %s", e)
                    return None
                time.sleep(attempt * 3)
        return None

    def graphql(self, query: str, variables: dict | None = None,
                max_retries: int = 4) -> dict | None:
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        for attempt in range(1, max_retries + 1):
            try:
                r = self.session.post(self.GQL, json=payload, timeout=30)
                if r.status_code == 200:
                    data = r.json()
                    if "errors" in data and not data.get("data"):
                        logger.debug("GQL errors: %s", data["errors"])
                        return None
                    return data
                if r.status_code == 403:
                    self._wait_rate_limit(r)
                    continue
                if r.status_code in (500, 502, 503, 504):
                    time.sleep(attempt * 5)
                    continue
            except requests.RequestException as e:
                if attempt == max_retries:
                    logger.debug("GQL request error: %s", e)
                    return None
                time.sleep(attempt * 3)
        return None

    # ── File fetch ───────────────────────────────────────────────────────────
    def list_python_files(self, owner: str, repo: str) -> list[dict]:
        """Lay tat ca file .py trong repo qua recursive git tree API."""
        r = self.get(f"{self.REST}/repos/{owner}/{repo}")
        if r is None or r.status_code != 200:
            return []
        branch = r.json().get("default_branch", "main")
        r = self.get(
            f"{self.REST}/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
        )
        if r is None or r.status_code != 200:
            return []
        data = r.json()
        if data.get("truncated"):
            logger.warning("Tree truncated: %s/%s", owner, repo)
        return [
            item for item in data.get("tree", [])
            if item.get("type") == "blob"
            and item.get("path", "").endswith(".py")
            and not is_noise_path(item.get("path", ""))
        ]

    def fetch_blob(self, owner: str, repo: str, sha: str) -> str | None:
        r = self.get(f"{self.REST}/repos/{owner}/{repo}/git/blobs/{sha}")
        if r is None or r.status_code != 200:
            return None
        try:
            b64 = r.json().get("content", "").replace("\n", "")
            return base64.b64decode(b64).decode("utf-8", errors="replace")
        except Exception:
            return None

    def fetch_file_at_commit(
        self, owner: str, repo: str, commit_sha: str, file_path: str
    ) -> str | None:
        """Lay noi dung file tai mot commit cu the."""
        r = self.get(
            f"{self.REST}/repos/{owner}/{repo}/contents/{file_path}",
            params={"ref": commit_sha},
        )
        if r is None or r.status_code != 200:
            return None
        try:
            b64 = r.json().get("content", "").replace("\n", "")
            return base64.b64decode(b64).decode("utf-8", errors="replace")
        except Exception:
            return None

    def get_commit_files(
        self, owner: str, repo: str, commit_sha: str
    ) -> list[dict]:
        """Lay danh sach file thay doi trong mot commit."""
        r = self.get(f"{self.REST}/repos/{owner}/{repo}/commits/{commit_sha}")
        if r is None or r.status_code != 200:
            return []
        return [
            f for f in r.json().get("files", [])
            if f.get("filename", "").endswith(".py")
            and f.get("status") in ("modified", "changed")
        ]

    def get_commit_parent(
        self, owner: str, repo: str, commit_sha: str
    ) -> str | None:
        """Lay SHA cua parent commit (truoc khi fix)."""
        r = self.get(f"{self.REST}/repos/{owner}/{repo}/commits/{commit_sha}")
        if r is None or r.status_code != 200:
            return None
        parents = r.json().get("parents", [])
        return parents[0]["sha"] if parents else None


# ─────────────────────────────────────────────────────────────────────────────
# BENIGN COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────
class BenignCollector:
    def __init__(
        self,
        client: GitHubClient,
        seen_hashes: set[str],
        max_per_repo: int = 800,
    ) -> None:
        self.client       = client
        self.seen_hashes  = seen_hashes
        self.max_per_repo = max_per_repo

    def collect_from_repo(self, owner: str, repo: str) -> list[dict]:
        repo_key = f"{owner}/{repo}"
        logger.info("  [Benign] %s — lay danh sach file...", repo_key)

        files = self.client.list_python_files(owner, repo)
        if not files:
            logger.info("    -> Khong co file .py.", )
            return []

        random.shuffle(files)
        logger.info("    -> %d file .py hop le.", len(files))

        collected: list[dict] = []
        scanned = 0

        for item in files:
            if len(collected) >= self.max_per_repo:
                break

            source = self.client.fetch_blob(owner, repo, item["sha"])
            if not source:
                continue
            scanned += 1

            funcs = extract_functions(
                source,
                item["path"],
                min_lines  = MIN_LINES,
                max_lines  = MAX_LINES,
                min_words  = BENIGN_MIN_WORDS,
                max_words  = BENIGN_MAX_WORDS,
                filter_security_kw=True,
            )

            for func in funcs:
                normed = norm_code(func["code"])
                h      = sha1(normed)
                if h in self.seen_hashes:
                    continue
                self.seen_hashes.add(h)

                sid = sha1(f"benign:{repo_key}:{item['path']}:{func['function_name']}")[:16]
                collected.append({
                    "sample_id"       : f"benign:{sid}",
                    "cve_id"          : None,
                    "ghsa_id"         : None,
                    "data_source"     : "benign_github",
                    "quality_tier"    : "benign",
                    "repository"      : repo_key,
                    "sha"             : item["sha"],
                    "file"            : item["path"],
                    "function"        : func["function_name"].split(".")[-1],
                    "full_function_name": func["function_name"],
                    "severity"        : None,
                    "signature"       : func["signature"],
                    "code"            : normed,
                    "label"           : 0,
                    "cwe_ids"         : [],
                })

                if len(collected) >= self.max_per_repo:
                    break

            time.sleep(0.05)

        logger.info(
            "    -> scan %d files, thu thap %d ham benign.",
            scanned, len(collected),
        )
        return collected


# ─────────────────────────────────────────────────────────────────────────────
# VULNERABLE COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────
class VulnerableCollector:
    """Thu thap ham vulnerable tu cac commit fix duoc liet ke trong GHSA/OSV."""

    def __init__(
        self,
        client: GitHubClient,
        seen_hashes: set[str],
    ) -> None:
        self.client      = client
        self.seen_hashes = seen_hashes

    def fetch_advisories_with_fixes(
        self, max_pages: int = 50
    ) -> list[dict]:
        """Lay danh sach advisory co fix-commit URL tu GitHub GraphQL API."""
        results: list[dict] = []
        cursor = None

        for page in range(max_pages):
            variables: dict[str, Any] = {}
            if cursor:
                variables["cursor"] = cursor

            data = self.client.graphql(GRAPHQL_ADVISORIES_QUERY, variables)
            if not data:
                break

            vulns = (data.get("data") or {}).get("securityVulnerabilities", {})
            nodes = vulns.get("nodes", [])

            for node in nodes:
                advisory = node.get("advisory", {})
                refs     = advisory.get("references", [])
                cwes     = [
                    c["cweId"]
                    for c in advisory.get("cwes", {}).get("nodes", [])
                ]

                for ref in refs:
                    url = ref.get("url", "")
                    m   = COMMIT_URL_RE.search(url)
                    if m:
                        results.append({
                            "ghsa_id"    : advisory.get("ghsaId"),
                            "severity"   : advisory.get("severity"),
                            "cwe_ids"    : cwes,
                            "owner"      : m.group(1),
                            "repo"       : m.group(2),
                            "commit_sha" : m.group(3),
                            "package"    : (node.get("package") or {}).get("name"),
                        })

            page_info = vulns.get("pageInfo", {})
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            logger.info("  Advisory page %d: tong %d fix commits.", page + 1, len(results))
            time.sleep(0.5)

        logger.info("Tong fix-commit URLs tu GHSA: %d", len(results))
        return results

    def _extract_vuln(
        self,
        owner: str,
        repo: str,
        commit_sha: str,
        parent_sha: str,
        file_path: str,
        ghsa_id: str,
        severity: str,
        cwe_ids: list[str],
    ) -> list[dict]:
        """Trich xuat cac ham vulnerable tu parent commit (truoc khi fix).

        Khong can safe_code — chi lay code bi lo hong (= noi dung file
        o parent commit, truoc khi commit fix duoc ap dung).
        """
        samples: list[dict] = []

        # Chi tai file o parent commit (= phien ban co lo hong)
        vuln_src = self.client.fetch_file_at_commit(owner, repo, parent_sha, file_path)
        if not vuln_src:
            return samples

        funcs = extract_functions(
            vuln_src, file_path,
            min_lines=MIN_LINES, max_lines=MAX_LINES,
            min_words=VULN_MIN_WORDS, max_words=VULN_MAX_WORDS,
            filter_security_kw=False,  # Giu nguyen tu khoa bao mat
        )

        for func in funcs:
            vuln_code = norm_code(func["code"])
            h         = sha1(vuln_code)
            if h in self.seen_hashes:
                continue
            self.seen_hashes.add(h)

            sid = sha1(
                f"vuln:{owner}/{repo}:{commit_sha}:{file_path}:{func['function_name']}"
            )[:16]
            samples.append({
                "sample_id"       : f"vuln:{sid}",
                "cve_id"          : None,
                "ghsa_id"         : ghsa_id,
                "data_source"     : "github_advisory_fix",
                "quality_tier"    : "advisory",
                "repository"      : f"{owner}/{repo}",
                "sha"             : commit_sha,
                "file"            : file_path,
                "function"        : func["function_name"].split(".")[-1],
                "full_function_name": func["function_name"],
                "severity"        : severity,
                "signature"       : func["signature"],
                "code"            : vuln_code,
                "label"           : 1,
                "cwe_ids"         : cwe_ids,
            })

        return samples

    def collect(self, target: int) -> list[dict]:
        advisories = self.fetch_advisories_with_fixes()
        random.shuffle(advisories)

        all_vuln: list[dict] = []
        processed_commits: set[str] = set()
        # Cache parent SHA de tranh goi API 2 lan cho cung commit
        parent_cache: dict[str, str | None] = {}

        for adv in advisories:
            if len(all_vuln) >= target:
                break

            owner      = adv["owner"]
            repo       = adv["repo"]
            commit_sha = adv["commit_sha"]
            ghsa_id    = adv.get("ghsa_id", "")
            severity   = adv.get("severity", "UNKNOWN")
            cwe_ids    = adv.get("cwe_ids", [])

            commit_key = f"{owner}/{repo}:{commit_sha}"
            if commit_key in processed_commits:
                continue
            processed_commits.add(commit_key)

            # Lay parent SHA (co cache)
            if commit_key not in parent_cache:
                parent_cache[commit_key] = self.client.get_commit_parent(
                    owner, repo, commit_sha
                )
            parent_sha = parent_cache[commit_key]
            if not parent_sha:
                continue

            logger.info(
                "  [Vuln] %s/%s @ %s (%s) | tong: %d/%d",
                owner, repo, commit_sha[:8], ghsa_id, len(all_vuln), target,
            )

            changed_files = self.client.get_commit_files(owner, repo, commit_sha)
            py_files      = [
                f["filename"] for f in changed_files
                if not is_noise_path(f["filename"])
            ]

            for fp in py_files:
                if len(all_vuln) >= target:
                    break
                new_samples = self._extract_vuln(
                    owner, repo, commit_sha, parent_sha, fp,
                    ghsa_id, severity, cwe_ids,
                )
                all_vuln.extend(new_samples)
                if new_samples:
                    logger.info(
                        "    -> %s: +%d mau (tong %d)",
                        fp, len(new_samples), len(all_vuln),
                    )

            time.sleep(0.3)

        logger.info("Thu thap duoc %d mau vulnerable.", len(all_vuln))
        return all_vuln


# ─────────────────────────────────────────────────────────────────────────────
# Progress save/resume helpers
# ─────────────────────────────────────────────────────────────────────────────
def load_existing_output(path: Path, seen_hashes: set[str]) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                for field in ("code", "safe_code"):
                    c = rec.get(field)
                    if c:
                        seen_hashes.add(sha1(norm_code(c)))
                count += 1
            except Exception:
                pass
    return count


def append_records(path: Path, records: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_token(token_arg: str | None) -> str | None:
    if token_arg:
        return token_arg.strip()
    for var in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
        v = os.getenv(var)
        if v:
            return v.strip()
    for env_path in (Path.cwd() / ".env", ROOT / ".env"):
        if env_path.exists():
            for line in env_path.read_text("utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                if k.strip() in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_PAT"):
                    tok = v.strip().strip("'\"")
                    if tok:
                        return tok
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thu thap 130k mau benign + 25k mau vulnerable Python."
    )
    parser.add_argument("--token", default=None,
                        help="GitHub Classic PAT (ghp_...). "
                             "Hoac thiet lap GITHUB_TOKEN env var.")
    parser.add_argument("--benign-limit", type=int, default=130_000,
                        help="So mau benign can thu thap (mac dinh: 130000).")
    parser.add_argument("--vuln-limit", type=int, default=25_000,
                        help="So mau vulnerable can thu thap (mac dinh: 25000).")
    parser.add_argument("--max-per-repo", type=int, default=800,
                        help="So ham toi da moi repo benign (mac dinh: 800).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help=f"Thu muc luu ket qua (mac dinh: {DEFAULT_OUT_DIR}).")
    parser.add_argument("--mode", choices=["all", "benign", "vulnerable"],
                        default="all",
                        help="Mode thu thap: all / benign / vulnerable.")
    parser.add_argument("--append", action="store_true",
                        help="Tiep tuc tu lan chay truoc (doc lai file output, skip trung lap).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    token = load_token(args.token)
    if not token:
        parser.error(
            "Yeu cau GitHub Classic PAT.\n"
            "Them --token ghp_XXXX hoac thiet lap GITHUB_TOKEN env var."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    benign_out  = args.output_dir / "benign_samples.jsonl"
    vuln_out    = args.output_dir / "vulnerable_samples.jsonl"

    # Buoc 1: Nap hash tu data cu de dedup
    logger.info("=" * 65)
    logger.info("BUOC 1: Nap hash tu data cu de tranh trung lap...")
    logger.info("=" * 65)
    seen_hashes = load_existing_hashes()

    # Buoc 2: Nap hash tu output hien tai (neu --append)
    if args.append:
        logger.info("BUOC 2: Nap hash tu output cu (--append mode)...")
        b_existing = load_existing_output(benign_out, seen_hashes)
        v_existing = load_existing_output(vuln_out,   seen_hashes)
        logger.info("  Benign hien co: %d, Vuln hien co: %d", b_existing, v_existing)
    else:
        b_existing = 0
        v_existing = 0
        # Xoa file cu neu khong append
        if benign_out.exists() and args.mode in ("all", "benign"):
            benign_out.unlink()
        if vuln_out.exists() and args.mode in ("all", "vulnerable"):
            vuln_out.unlink()

    client = GitHubClient(token=token)

    # ── BENIGN ────────────────────────────────────────────────────────────────
    if args.mode in ("all", "benign"):
        benign_needed = max(0, args.benign_limit - b_existing)
        logger.info("=" * 65)
        logger.info("BUOC 3: Thu thap BENIGN samples. Can them: %d", benign_needed)
        logger.info("=" * 65)

        if benign_needed > 0:
            benign_collector = BenignCollector(
                client=client,
                seen_hashes=seen_hashes,
                max_per_repo=args.max_per_repo,
            )

            repos = list(BENIGN_REPOS)
            random.shuffle(repos)
            benign_total = b_existing

            for repo_full in repos:
                if benign_total >= args.benign_limit:
                    break

                parts = repo_full.strip().split("/")
                if len(parts) != 2:
                    continue
                owner, repo = parts

                still_need = args.benign_limit - benign_total
                benign_collector.max_per_repo = min(args.max_per_repo, still_need)

                logger.info(
                    "[Benign %d/%d] %s (con can %d)",
                    benign_total, args.benign_limit, repo_full, still_need,
                )
                try:
                    new_recs = benign_collector.collect_from_repo(owner, repo)
                except Exception as e:
                    logger.error("Loi %s: %s", repo_full, e)
                    continue

                if new_recs:
                    append_records(benign_out, new_recs)
                    benign_total += len(new_recs)
                    logger.info("  >> Tong benign: %d/%d", benign_total, args.benign_limit)

                time.sleep(random.uniform(0.3, 1.0))

            logger.info("Hoan thanh benign: %d mau.", benign_total)
        else:
            logger.info("Da du mau benign (%d), bo qua.", b_existing)

    # ── VULNERABLE ────────────────────────────────────────────────────────────
    if args.mode in ("all", "vulnerable"):
        vuln_needed = max(0, args.vuln_limit - v_existing)
        logger.info("=" * 65)
        logger.info("BUOC 4: Thu thap VULNERABLE samples. Can them: %d", vuln_needed)
        logger.info("=" * 65)

        if vuln_needed > 0:
            vuln_collector = VulnerableCollector(
                client=client,
                seen_hashes=seen_hashes,
            )
            new_vulns = vuln_collector.collect(target=vuln_needed)
            if new_vulns:
                append_records(vuln_out, new_vulns)
                logger.info("Da luu %d mau vulnerable.", len(new_vulns))
        else:
            logger.info("Da du mau vulnerable (%d), bo qua.", v_existing)

    # ── Summary ───────────────────────────────────────────────────────────────
    final_benign = sum(1 for _ in open(benign_out, encoding="utf-8")) if benign_out.exists() else 0
    final_vuln   = sum(1 for _ in open(vuln_out,   encoding="utf-8")) if vuln_out.exists() else 0

    logger.info("=" * 65)
    logger.info("HOAN TAT!")
    logger.info("  Benign  : %d / %d mau  -> %s", final_benign, args.benign_limit, benign_out)
    logger.info("  Vuln    : %d / %d mau  -> %s", final_vuln,   args.vuln_limit,   vuln_out)
    logger.info("=" * 65)

    print(json.dumps({
        "benign_output"  : str(benign_out),
        "vuln_output"    : str(vuln_out),
        "benign_count"   : final_benign,
        "vuln_count"     : final_vuln,
        "benign_target"  : args.benign_limit,
        "vuln_target"    : args.vuln_limit,
        "benign_reached" : final_benign >= args.benign_limit,
        "vuln_reached"   : final_vuln   >= args.vuln_limit,
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
