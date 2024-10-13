from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm_adaptive_router",
    version="0.1.11",
    author="Emin Genc",
    author_email="emingench@gmail.com",
    description="An adaptive router for LLM model selection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emingenc/llm_adaptive_router",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "langchain==0.3.1",
        "langchain-community==0.3.1",
        "langchain-core==0.3.8",
        "langchain-openai==0.2.1",
        "langchain-text-splitters==0.3.0",
    ],
    keywords=["llm", "router", "adaptive", "llm-adaptive-router", "llm-router"],
    project_urls={
        "Homepage": "https://github.com/emingenc/llm_adaptive_router",
        "Source": "https://github.com/emingenc/llm_adaptive_router",
        "Issue Tracker": "https://github.com/emingenc/llm_adaptive_router/issues",
    },
)