import modal
from pydantic import BaseModel
import json
import os

multion_image = modal.Image.debian_slim().pip_install("multion")
app = modal.App("MultiOnAgent", image=multion_image, secrets=[modal.Secret.from_name("MULTION")])

class Command(BaseModel):
    cmd: str
    restart_session: bool = False

@app.function()
@modal.web_endpoint(method="POST")
def run(command: Command):
    from multion.client import MultiOn

    multion = MultiOn(api_key=os.environ["MULTION_API_KEY"])
    url = "https://google.com"
    if command.restart_session:
        session = multion.sessions.create(url=url)
        session_id = session.session_id
    browse = multion.browse(
        cmd=command.cmd,
        url=url,
        local=False,
        temperature=0
    )
    if command.restart_session:
        multion.sessions.close(session_id=session_id)
    response = json.dumps({"message": browse.message})
    return response
