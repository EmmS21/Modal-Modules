
# Marketing Agent

This repository contains the code for a Marketing Agent application designed to generate marketing materials based on user personas and business information. The application utilizes LangChain, OpenAI's GPT-4, and the Modal platform to provide intelligent marketing solutions.

## Overview

The Marketing Agent generates keywords, advertisement text, and marketing strategies for businesses. The agent uses a combination of natural language processing and AI tools to understand the target user persona and create tailored marketing content.

## Components

### 1. **Models**
- **Command:** Defines the input format for user persona and business name.
- **AdText:** Contains headlines and descriptions for advertisements.
- **MarketingResult:** The output format containing keywords, ad texts, paths taken, business, and user persona information.
- **KeywordIdeasInput:** Handles input for generating keyword ideas.

### 2. **Functions**
- **load_prompt:** Loads the OpenAI prompt from a remote URL.
- **multi_on_agent:** Interacts with the MultiOn agent to browse and extract information from the web.

### 3. **Agent Execution**
The main function `agent` processes the input command, interacts with the OpenAI language model, and generates marketing results. The execution involves the use of LangChain's tools and memory management for conversational context.

## Deployment

The application is deployed using Modal, a platform for running AI applications. The `App` object initializes with the necessary image, secrets, and functions.

### Dependencies

- `langchain`
- `langchain_openai`
- `modal`
- `requests`
- `json`
- `os`

## Usage

1. Set up the environment with the required dependencies.
2. Deploy the application on Modal.
3. Use the `agent` function to generate marketing materials by providing the `Command` input.

## License

This project is licensed under the MIT License.

