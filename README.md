# LLM-Based Synthetic Heat-Transfer Problem Generation
An open-source pipeline that uses large language models to generate synthetic
undergraduate engineering heat-transfer problem-solving data, paired with **deterministic
quality verification** for numeric-answer checking and code execution.

> **Status: work in progress.** Seed set, verification, and a working generation
> stage are implemented and runnable today. LLM-as-judge scoring, deduplication,
> diversity balancing, and scale-up are on the roadmap below. See
> [Project status](#project-status) for exactly what is and isn't built yet.

[![python](https://img.shields.io/badge/python-3.11.16-blue.svg)](https://www.python.org/) 
![os](https://img.shields.io/badge/os-ubuntu%20|%20macos%20|%20windows-blue.svg)
[![license](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/license/mit)

[![coverage_test](https://github.com/HibaKob/LLM-Synthetic-Heat-Transfer/actions/workflows/coverage_test.yml/badge.svg?branch=main)](https://github.com/HibaKob/LLM-Synthetic-Heat-Transfer/actions/workflows/coverage_test.yml)
[![codecov](https://codecov.io/gh/HibaKob/LLM-Synthetic-Heat-Transfer/graph/badge.svg?token=SXCVUX5MT7)](https://codecov.io/gh/HibaKob/LLM-Synthetic-Heat-Transfer)


## Table of Contents
* [Motivation](#motivation)
* [Roadmap](#roadmap)
* [Pipeline Description](#pipeline)
* [Tutorial](#tutorial)
* [To-Do List](#todo)
* [Contact Information](#contact)
* [Resources](#resources)

## Motivation <a name="motivation"></a>
Synthetic data is increasingly used to train and post-train language models, but a
central challenge is **quality control**: how do we know the generated examples are
correct? Most pipelines rely on an LLM-as-judge, which is useful but inherits the
model's own errors and biases.

This project explores a complementary approach for a domain where correctness is
**checkable**: engineering problems with numeric answers or executable code. Each
generated example, whether of **Type A (analytical)** or **Type B (coding)** (see [Pipeline Description](#pipeline) for a description of the different problem types) is verified deterministically:

- **Type A (analytical):** the stated answer is checked against an independent
  recomputation.
- **Type B (coding):** the generated solution is executed against test cases whose
  answers are known analytically.

The correctness signal therefore comes from computation, not from trusting the
generating model, a concept that transfers to any domain with a verifiable ground truth.

## Roadmap <a name="roadmap"></a>

## Pipeline Description <a name="pipeline"></a>

## Tutorial <a name="tutorial"></a>

## To-Do List <a name="todo"></a>

## Contact Information <a name="contact"></a>
For additional information, please contact Hiba Kobeissi ``hibakob@bu.edu``.

## Resources <a name="resources"></a>