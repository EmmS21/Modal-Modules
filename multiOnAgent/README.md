# MultiOnAgent Module Documentation

## Overview

This module provides a web endpoint for interacting with the MultiOn API to browse and extract information from Google. It uses Modal for deployment and Pydantic for request validation.

## Dependencies

- modal
- pydantic
- multion

## Setup

1. Ensure you have a Modal account and the Modal CLI installed.
2. Set up a secret named "MULTION" in Modal containing your MultiOn API key.

## Module Components

### Image Configuration

```
multion_image = modal.Image.debian_slim().pip_install("multion")
app = modal.App("MultiOnAgent", image=multion_image, secrets=[modal.Secret.from_name("MULTION")])
```

- Creates a Debian-based Docker image with the multion package installed.
- Initializes a Modal App named "MultiOnAgent" with the custom image and MULTION secret.

### Command Model

```
class Command(BaseModel):
    cmd: str
    restart_session: bool = False
```

Defines the structure for incoming commands using Pydantic.
- cmd: The command to be executed (required).
- restart_session: Whether to start a new session (optional, default is False).

### Web Endpoint
```
@app.function()
@modal.web_endpoint(method="POST")
def run(command: Command):
    # Function implementation
```
- Exposes a POST endpoint that accepts a Command object.
- Interacts with the MultiOn API to execute the command and return results.

## Usage
To use this module:

Deploy the app using Modal CLI.
`modal deploy runAgent.py`

and run the module using a CURL command:

`curl -X POST https://emms21--multionagent-run.modal.run -H "Content-Type: application/json" -d '{"cmd": "What city am I in", "restart_session": false}'`

The endpoint will return a JSON response with the message key containing the results from MultiOn.

Note
Ensure that the MULTION_API_KEY environment variable is properly set in your Modal deployment environment.
