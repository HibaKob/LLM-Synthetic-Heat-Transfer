# LLM-Based Synthetic Heat-Transfer Problem Generation

[![python](https://img.shields.io/badge/python-3.11.16-blue.svg)](https://www.python.org/) 
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/license/mit)

[![coverage_test](https://github.com/HibaKob/LLM-Synthetic-Heat-Transfer/actions/workflows/coverage_test.yml/badge.svg?branch=main)](https://github.com/HibaKob/LLM-Synthetic-Heat-Transfer/actions/workflows/coverage_test.yml)
[![codecov](https://codecov.io/gh/HibaKob/LLM-Synthetic-Heat-Transfer/graph/badge.svg?token=SXCVUX5MT7)](https://codecov.io/gh/HibaKob/LLM-Synthetic-Heat-Transfer)


An open-source pipeline that uses large language models to generate synthetic
undergraduate engineering heat-transfer problem-solving data, paired with **deterministic
quality verification** for numeric-answer checking and code execution.

> **Status: work in progress.** Seed set, verification, and a working generation
> stage are implemented and runnable today. LLM-as-judge scoring, deduplication,
> diversity balancing, and scale-up are on the roadmap below. See
> [Project status](#project-status) for exactly what is and isn't built yet.


## Table of Contents
* [Motivation](#motivation)
* [Pipeline Description](#pipeline)
* [Quickstart](#quickstart)
* [Project status](#project-status)
* [Contact Information](#contact)
* [Resources](#resources)

## Motivation <a name="motivation"></a>
Synthetic data is increasingly used to train and post-train language models, but a central challenge is **quality control**: how do we know the generated examples are correct? Most pipelines rely on an LLM-as-judge, which is useful but inherits the model's own errors and biases.

This project explores a complementary approach for a domain where correctness is **checkable**: engineering problems with numeric answers or executable code. Each generated example, whether of **Type A (analytical)** or **Type B (coding)** (see [Pipeline Description](#pipeline) for a description of the different problem types) is verified deterministically:

- **Type A (analytical):** the stated answer is checked against an independent recomputation.
- **Type B (coding):** the generated solution is executed against test cases whose answers are known analytically.

The correctness signal therefore comes from computation, not from trusting the generating model, a concept that transfers to any domain with a verifiable ground truth.

## Pipeline Description <a name="pipeline"></a>
The released pipeline so far consists of 2 stages to achieve verified data generation.
* STAGE 1: **BUILD SEEDS** human authored problems ──▶ build_seeds.py ──▶ verification.py checks the humnan work ──▶ verified seeds.json 

* STAGE 2: **GENERATE DATASET** seeds.json ──▶ LLM generates new problems ──▶ verification.py checks the LLM's work (each with answer + check) ──▶ keep passing, discard failing ──▶ final dataset

1. **Seeds.** A set of author-written, verified heat-transfer problems spanning conduction, convection, radiation, transient analysis, and heat exchangers, in two formats: analytical (Type A) and coding (Type B).
2. **Generation.** An LLM expands the seeds into new problems using a given prompt (increasing complexity, varying parameters and setup).
3. **Verification.** Every generated example is checked deterministically; only examples that pass are kept.
4. **Provenance.** Every example carries a `provenance` field recording its source (authored seed vs. model-generated) and, for generated items, the seed and model it came from, such that data lineage is explicit.

The data consisted of 45 typical undergraduate heat transfer problems. The coverage across heat transfer subfields is as follows:

* Conduction (8 Type A + 2 Type B): plane wall, composite/series resistance, cylindrical shell, resistance networks with convection, R-value, finite-difference solvers (with and without heat generation)
• Convection (11 Type A): Newton's law of cooling, solving for h, Nusselt/Reynolds numbers, fin temperature profile
* Radiation (5 Type A): Stefan-Boltzmann, gray-surface net exchange
* Extended surfaces/fins (4 Type A + 1 Type B): fin parameter, infinite-fin heat rate
* Transient (5 Type A + 2 Type B): Biot/Fourier numbers, lumped capacitance, thermal diffusivity, explicit FTCS transient solver
* Heat exchangers (4 Type A + 1 Type B): LMTD (counter/parallel flow), heat duty, energy balance
* Energy balance (2 Type A): sensible heat, heater timing

In total, there are 39 Type A problems and 6 Type B problems. Type A problems are analytical problems that ask for computing a numerical solution for a single unknown. For each Type A generated seed, a Python check_expression computes the answer independently, to maintain robustness of the code and diminish possible human erros. Type B problems on the other hand, are problems that expect a numerical expression as an output. Each Type B seed carries a reference_solution and a verification field describing test cases with known analytical answers (linear profile for source-free conduction, parabola for uniform generation, exponential decay for lumped cooling, steady-state limit for transient).

Whether the data is generated by a human author or an LLM, only the problems that pass the verification process are saved into the .json files. 

In this work, the LLM used is llama3.1.

### Data provenance

Every example records its origin:

- `human_authored` — written by the author, then passed deterministic verification.
- `llm_generated_verified` — produced by a model, then passed deterministic verification.

Generated examples also record the originating seed id and the generator model, so the composition and lineage of any dataset built from this pipeline are fully transparent.

### Design notes

- **Verifiable domain by design.** Heat transfer was chosen because answers are numeric or executable, enabling deterministic verification. The verification approach generalizes to any domain with a computable ground truth.
- **Judge independence (planned).** When the LLM-as-judge stage is added, the judge model will differ from the generator to avoid self-preference bias.
- **Single-teacher bias (planned mitigation).** Multi-teacher generation is on the oadmap to reduce a dataset mimicking one model's quirks.

## Quickstart <a name="quickstart"></a>
### Get a copy of the repository on your local machine
The best way to do this is to create a GitHub account and ``clone`` the repository. However, you can also download the repository by clicking the green ``Code`` button and selecting ``Download ZIP``. Download and unzip the ``LLM-Synthetic-Heat-Transfer-main`` folder and place it in a convenient location on your computer.

Alternatively, you can run the following command in a ``Terminal`` session:
```bash
git clone https://github.com/HibaKob/LLM-Synthetic-Heat-Transfer.git
```
Following this step, ``LLM-Synthetic-Heat-Transfer`` folder will be downloaded in your ``Terminal`` directory. 

### Create and activate a conda virtual environment
1. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) on your local machine.
2. Open a ``Terminal`` session (or equivalent) -- note that Mac computers come with ``Terminal`` pre-installed (type ``⌘-space`` and then search for ``Terminal``).
3. Type in the terminal to create a virtual environment with conda:
```bash
conda create --name LLM-Synthetic-Heat-Transfer-env python=3.11.16
```
4. Type in the terminal to activate your virtual environment:
```bash
conda activate LLM-Synthetic-Heat-Transfer-env
```
5. Check to make sure that the correct version of python is running (should be ``3.11.16``)
```bash
python --version
```
6. Update some base modules (just in case)
```bash
pip install --upgrade pip setuptools wheel
```

Note that once you have created this virtual environment you can ``activate`` and ``deactivate`` it in the future -- it is not necessary to create a new virtual environment each time you want to run this code, you can simply type ``conda activate LLM-Synthetic-Heat-Transfer-env`` and then pick up where you left off (see also: [conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)).


### Install LLM-Synthetic-Heat-Transfer
1. Use a ``Terminal`` session to navigate to the ``LLM-Synthetic-Heat-Transfer-main`` folder or ``LLM-Synthetic-Heat-Transfer`` folder (depending on the method you followed to download the github repository). The command ``cd`` will allow you to do this (see: [terminal cheat sheet](https://terminalcheatsheet.com/))
2. Type the command ``ls`` and make sure that the file ``pyproject.toml`` is in the current directory.
3. Now, create an editable install of microbundle compute:
```bash
pip install -e .
```
4. If you would like to see what packages were installed, you can type ``pip list``
5. You can test that the code is working with pytest (all tests should pass):
```bash
pytest -v --cov=syntheticheatransfer  --cov-report term-missing
```

### Using LLM-Synthetic-Heat-Transfer
Using ``LLM-Synthetic-Heat-Transfer`` is very simple. First, the human-authored problems should be put in a format that is machine readable, and should be verified to avoid calculation mistakes. This is achieved by modifying and running the ``build_seeds.py`` script file contained in the ``scripts`` folder. 

In a terminal running the ``LLM-Synthetic-Heat-Transfer-env`` environment, run the following python command.
```bash
python python .../scripts/build_seeds.py
```

The generator and (planned) judge models are set via configuration so they can be swapped or mixed without code changes. Set the API credentials via environment variable before running generation.



## Project status <a name="project-status"></a>
**Implemented**
- [x] Authored, verified seed set (`data/seed`) with per-example provenance
- [x] Deterministic verification: numeric-answer checking (Type A) and code execution (Type B)
- [x] Working LLM generation stage (llama-based expansion of seeds)
- [x] Generated and verified output (`data/generated`)
- [x] Tests and CI

**Roadmap**
- [ ] Expand human-authored seed data to cover more topics across heat transfer subfields
- [ ] LLM-as-judge scoring for reasoning quality (using a judge model distinct from the generator)
- [ ] Deduplication (near-duplicate detection) and diversity balancing across topics/difficulty
- [ ] Multi-teacher generation (mixing models) to reduce single-teacher bias
- [ ] Scale-up to a larger released dataset and a Hugging Face dataset card
- [ ] Downstream evaluation: fine-tune a small model on filtered vs. unfiltered data

## Contact Information <a name="contact"></a>
For additional information, please contact Hiba Kobeissi ``hibakob@bu.edu``.

## Resources <a name="resources"></a>
[1] [https://ollama.com](https://ollama.com)