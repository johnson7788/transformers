#读取setup.py中的_deps

_deps = [
    "black>=20.8b1",
    "cookiecutter==1.7.2",
    "dataclasses",
    "datasets",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flake8>=3.8.3",
    "flax==0.2.2",
    "fugashi>=1.0",
    "ipadic>=1.0.0,<2.0",
    "isort>=5.5.4",
    "jax>=0.2.0",
    "jaxlib==0.1.55",
    "keras2onnx",
    "numpy",
    "onnxconverter-common",
    "onnxruntime-tools>=1.4.2",
    "onnxruntime>=1.4.0",
    "packaging",
    "parameterized",
    "protobuf",
    "psutil",
    "pydantic",
    "pytest",
    "pytest-xdist",
    "python>=3.6.0",
    "recommonmark",
    "regex!=2019.12.17",
    "requests",
    "sacremoses",
    "scikit-learn",
    "sentencepiece==0.1.91",
    "sphinx-copybutton",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme==0.4.3",  # sphinx-rtd-theme==0.5.0 introduced big changes in the style.
    "sphinx==3.2.1",
    "starlette",
    "tensorflow-cpu>=2.0",
    "tensorflow>=2.0",
    "timeout-decorator",
    "tokenizers==0.9.4",
    "torch>=1.0",
    "tqdm>=4.27",
    "unidic>=1.0.2",
    "unidic_lite>=1.0.7",
    "uvicorn",
]

with open("requirements.txt", "w") as f:
    for dep in _deps:
        if dep == "python>=3.6.0":
            continue
        f.write(dep+"\n")