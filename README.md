# AI Agents Deployed to Modal

This repository contains two AI agents deployed to Modal, a serverless platform. The agents are designed for marketing and judgment tasks, each with specific capabilities and integrations with external services like Google Ads, MongoDB, and MultiOn.

## Table of Contents

- [JudgeAgent](#judgeagent)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Functions](#functions)
- [MarketingAgent](#marketingagent)
  - [Overview](#overview-1)
  - [Installation](#installation-1)
  - [Usage](#usage-1)
  - [Functions](#functions-1)
- [MultiOnAgent](#multionagent)
  - [Overview](#overview-2)
  - [Installation](#installation-2)
  - [Usage](#usage-2)
  - [Functions](#functions-2)

## JudgeAgent

### Overview

`JudgeAgent` is an AI agent designed to assist in generating marketing strategies, validating ad text, and interacting with the Google Ads API. It integrates with MongoDB for data retrieval and storage and uses LangChain and OpenAI for natural language processing and decision-making.

### Installation

The `JudgeAgent` is deployed using the following dependencies:

- langchain
- langchain_openai
- pymongo
- google_auth_oauthlib
- google-ads

These dependencies are installed within a `debian_slim` Docker image.

### Usage

The `JudgeAgent` can be triggered via Modal's serverless functions. The main functionalities include generating keyword ideas, validating ad text, and interacting with the Google Ads API.

### Functions

- **authentication_google**: Handles authentication for the Google Ads API using OAuth2.
- **generate_keyword_ideas**: Generates keyword ideas based on seed keywords using the Google Ads Keyword Planner API.
- **validate_ad_length**: Validates the length of headlines and descriptions for advertisements.
- **run**: Executes the agent's main workflow, connecting to MongoDB, processing marketing data, and using OpenAI's GPT-4 model to generate marketing strategies.

## MarketingAgent

### Overview

`MarketingAgent` is an AI agent designed for generating marketing content tailored to specific user personas and businesses. It integrates with OpenAI's GPT-4 and can execute commands through the MultiOn agent for browsing and extracting information from the web.

### Installation

The `MarketingAgent` is deployed using the following dependencies:

- langchain
- langchain_openai

These dependencies are installed within a `debian_slim` Docker image.

### Usage

The `MarketingAgent` can be triggered via a POST request to the provided Modal web endpoint. It generates marketing content such as ad text and keywords based on the provided business name and user persona.

### Functions

- **multi_on_agent**: Interacts with the MultiOn agent to browse the web and extract information.
- **agent**: Executes the agent's main workflow, generating marketing strategies based on the input command, which includes business names and user personas.

## MultiOnAgent

### Overview

`MultiOnAgent` is a simple AI agent designed to interact with the MultiOn platform for browsing and extracting information from the web. It can initiate and manage browsing sessions and execute commands to gather information.

### Installation

The `MultiOnAgent` is deployed using the following dependency:

- multion

This dependency is installed within a `debian_slim` Docker image.

### Usage

The `MultiOnAgent` can be triggered via a POST request to the provided Modal web endpoint. It supports restarting sessions and executing browsing commands.

### Functions

- **run**: Executes a browsing command using the MultiOn platform, with optional session management.

## Additional Information

These agents are deployed on Modal's serverless platform and leverage the power of cloud computing to perform complex tasks without the need for dedicated infrastructure. For more details on using Modal, refer to the [Modal documentation](https://modal.com/docs).

