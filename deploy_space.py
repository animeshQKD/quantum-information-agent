"""
deploy_space.py - publish the web app (app.py) as a Hugging Face Space.

    python deploy_space.py --repo YOUR_NAME/quantum-information-agent --dry-run   # show what would happen
    python deploy_space.py --repo YOUR_NAME/quantum-information-agent             # upload the app
    python deploy_space.py --repo YOUR_NAME/quantum-information-agent --settings  # + Space secrets/variables

Safety:
  * Only the files in APP_FILES / DATA_FILES are uploaded (a whitelist). .env, logs/, figures/,
    data/sessions/, simulated data and the tests can never reach the Space.
  * Every uploaded file is scanned for API keys first; if one is found, nothing is uploaded.
  * --settings sends the keys from your local .env to the Space as SECRETS (encrypted, never in a
    file, not visible to visitors). OPENAI_API_KEY is never sent: the public app uses free models only.
  * The README (Space card) and requirements.txt are generated from THIS Python environment, so the
    Space runs the same Gradio, Python and package versions that passed the tests here.

Uploading needs a Hugging Face token with WRITE access: run `hf auth login` once (or set
HF_DEPLOY_TOKEN). The inference token (HF_TOKEN) is NEVER used for uploading, even when it is set in the
shell: huggingface_hub would otherwise prefer it silently over `hf auth login`, and an inference token
usually cannot write (2026-11-04: "403 Forbidden ... pass create_pr=1" right after a fresh login).
"""
import argparse
import os
import re
import shutil
import sys
import tempfile
from importlib import metadata

APP_FILES = ["app.py", "agent_tools.py", "capabilities.py", "display.py", "graph.py", "llm.py", "mcp_server.py",
             "plots.py", "qi_tools.py", "routing.py", "session.py", "workbench.py", "workbench_ui.py"]
DATA_FILES = ["data/state_tomo_data.csv", "data/example_fringe_scan.csv"]

# packages the app imports, pinned to the versions tested here (gradio itself comes from the Space's
# sdk_version). NOT pinned: pydantic - the Space installs gradio[oauth,mcp], whose 'mcp' extra limits
# pydantic (e.g. <= 2.12.5 for gradio 6.27), so a pin copied from this computer can make the build fail.
PACKAGES = ["numpy", "scipy", "matplotlib", "python-dotenv", "langgraph", "langchain-core",
            "langchain-openai", "langchain-groq", "langchain-google-genai"]
UNPINNED = ["pydantic"]
# what the Space's build adds to requirements.txt (from its build log)
SPACE_EXTRAS = ['gradio[oauth,mcp]=={gradio}', "uvicorn>=0.14.0", "websockets>=10.4", "spaces"]

SECRET_KEYS = ["HF_TOKEN", "GROQ_API_KEY", "GOOGLE_API_KEY"]        # never OPENAI_API_KEY
VARIABLES = {                                                        # public settings (not secret)
    "USE_HF": "1",
    "HF_MODEL": None,                 # None: copied from your .env if set there
    "HF_FALLBACK_MODEL": None,
    "GROQ_MODEL": None,
    "GROQ_FALLBACK_MODEL": None,
    "GEMINI_MODEL": None,
    "PROVIDER_ORDER": "hf,hf_backup,groq,groq_backup,gemini",
    "DAILY_HF_CALLS": "500",
    "QUESTIONS_PER_HOUR": "40",
}

KEY_PATTERNS = [r"hf_[A-Za-z0-9]{30,}", r"sk-[A-Za-z0-9_-]{20,}", r"gsk_[A-Za-z0-9]{20,}", r"AIza[0-9A-Za-z_-]{30,}"]


def find_keys(paths: list[str]) -> list[str]:
    """Files that contain something that looks like an API key."""
    bad = []
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as f:
            text = f.read()
        if any(re.search(k, text) for k in KEY_PATTERNS):
            bad.append(p)
    return bad


def requirements() -> str:
    lines = []
    for pkg in PACKAGES:
        try:
            lines.append(f"{pkg}=={metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            lines.append(pkg)
    lines += UNPINNED
    return "\n".join(lines) + "\n"


def check_install(req_file: str) -> str | None:
    """Ask pip whether the Space's build would succeed: the Space's exact install command (requirements.txt
    plus gradio[oauth,mcp] and the extras its build adds), resolved WITHOUT installing anything. It resolves
    for this computer: version conflicts (like pydantic vs gradio's 'mcp' extra) do not depend on the
    platform, while resolving for Linux from Windows is unreliable (pip would apply this computer's
    'Windows only' package conditions). Returns None if it resolves, else pip's explanation."""
    import subprocess
    import gradio
    cmd = [sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--disable-pip-version-check",
           "-r", req_file] + [x.format(gradio=gradio.__version__) for x in SPACE_EXTRAS]
    try:
        out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"could not run pip: {e}"
    if out.returncode == 0:
        return None
    lines = [x for x in ((out.stdout or "") + "\n" + (out.stderr or "")).splitlines() if x.strip()]
    start = next((i for i, x in enumerate(lines) if x.startswith("ERROR")), max(0, len(lines) - 25))
    return "\n".join(lines[start:start + 30])


TAGS = ["quantum", "quantum-information", "quantum-optics", "quantum-computing", "physics", "photonics",
        "entanglement", "tomography", "agent", "langgraph", "tool-calling", "education", "mcp-server"]


def model_ids(env: dict) -> list[str]:
    """Hub model ids the Space uses (HF_MODEL, HF_FALLBACK_MODEL without the ':provider' suffix). Listing
    them puts the Space on each model's page under 'Spaces using this model'."""
    out = []
    for k in ("HF_MODEL", "HF_FALLBACK_MODEL"):
        m = (env.get(k) or "").split(":")[0].strip()
        if m.count("/") == 1 and m not in out:
            out.append(m)
    return out


def space_host(repo: str) -> str:
    """The Space's own web address: 'Animesh1986/my_app' -> 'animesh1986-my-app.hf.space'."""
    return re.sub(r"[^a-z0-9-]", "-", repo.lower().replace("/", "-")) + ".hf.space"


def readme(title: str, env: dict | None = None, thumbnail: str = "", github: str = "", repo: str = "") -> str:
    import gradio
    mcp_url = f"https://{space_host(repo)}/gradio_api/mcp/" if repo else "https://<this Space's host>/gradio_api/mcp/"
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    models = model_ids(env or {})
    yaml_list = lambda key, xs: (f"{key}:\n" + "".join(f"  - {x}\n" for x in xs)) if xs else ""   # noqa: E731
    return f"""---
title: {title}
emoji: 🔬
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {gradio.__version__}
python_version: "{py}"
app_file: app.py
pinned: true
short_description: Photon polarization states, noise, entanglement, tomography
{yaml_list("tags", TAGS)}{yaml_list("models", models)}{f"thumbnail: {thumbnail}" + chr(10) if thumbnail else ""}---

# Quantum Information Agent

Ask in plain English about single photons and polarization-entangled photon pairs. A language model
plans the work, but every number comes from tested Python code (NumPy/SciPy), and an automatic answer
check removes or corrects statements that the computed results do not support.

**What it computes:** states and their density matrices, Stokes parameters, entanglement measures
(concurrence, negativity, entanglement of formation, CHSH, entropies), local and global noise channels
with their Kraus operators, Lindblad time evolution, wave plates, tomography of your own counts with
error bars, certification, protocols (teleportation, QKD key rate), and plots.

**Three ways to use it:** the *Chat* tab (language model), the *Workbench* tab (the same tools from
panels, no language model, no limits), and an **MCP server** for your own agent (Claude Desktop, Cursor, ...):
the physics tools at `{mcp_url}` (setup for Claude Desktop and Cursor is in the app under *Use these
tools from your own agent*). No language model runs for MCP calls.

**Limits of this public demo:** one or two photons (qubits); {VARIABLES['QUESTIONS_PER_HOUR']} chat
questions per hour per visitor; the free model quotas can run out for the day; uploaded files and
states are deleted when you close the tab or after two hours. Do not upload confidential data.

Built by Dr. Animesh Sinha Roy with LangGraph and Gradio.{chr(10) + chr(10) + "**Source code, tests and design notes:** " + github if github else ""}
"""


def stage(folder: str, title: str, env: dict | None = None, thumbnail: str = "", github: str = "",
          repo: str = "") -> list[str]:
    """Copy the whitelisted files plus the generated README and requirements into `folder`."""
    missing = [p for p in APP_FILES + DATA_FILES if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Missing file(s): {', '.join(missing)}. Run this script in the project folder.")
    for p in APP_FILES + DATA_FILES:
        os.makedirs(os.path.join(folder, os.path.dirname(p)), exist_ok=True)
        shutil.copy(p, os.path.join(folder, p))
    with open(os.path.join(folder, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme(title, env, thumbnail, github, repo))
    with open(os.path.join(folder, "requirements.txt"), "w", encoding="utf-8") as f:
        f.write(requirements())
    return sorted(os.path.relpath(os.path.join(r, n), folder).replace("\\", "/")
                  for r, _, names in os.walk(folder) for n in names)


def local_env() -> dict:
    from dotenv import dotenv_values
    return {k: v for k, v in dotenv_values(".env").items() if v} if os.path.isfile(".env") else {}


def deploy_token() -> tuple[str | None, str]:
    """The token for uploading and where it comes from: HF_DEPLOY_TOKEN, else the `hf auth login` token.
    HF_TOKEN in the environment is ignored here (it is the inference token, see the docstring)."""
    if os.getenv("HF_DEPLOY_TOKEN"):
        return os.environ["HF_DEPLOY_TOKEN"], "HF_DEPLOY_TOKEN"
    from huggingface_hub import get_token
    hidden = {k: os.environ.pop(k) for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN") if k in os.environ}
    try:
        return get_token(), "hf auth login"
    finally:
        os.environ.update(hidden)


NOT_ALLOWED = ("Hugging Face refused the upload (403): the token from {src} cannot write to {repo}.\n"
               "Create a token of type Write (huggingface.co -> Settings -> Access Tokens), run\n"
               "`hf auth login`, choose 'paste a token', paste it at the hidden prompt, and run this again.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="Space id, e.g. YOUR_NAME/quantum-information-agent")
    ap.add_argument("--title", default="Quantum Information Agent")
    ap.add_argument("--private", action="store_true", help="create the Space as private (first deploy)")
    ap.add_argument("--settings", action="store_true", help="also set the Space secrets and variables from .env")
    ap.add_argument("--dry-run", action="store_true", help="show what would be uploaded and set; change nothing")
    ap.add_argument("--thumbnail", default="", help="URL of an image shown when the Space link is shared")
    ap.add_argument("--github", default="", help="URL of the GitHub repository, linked from the Space card")
    ap.add_argument("--no-check", action="store_true", help="skip the pip check of the Space's install")
    a = ap.parse_args(argv)

    folder = tempfile.mkdtemp(prefix="qi_space_")
    try:
        env = local_env()
        files = stage(folder, a.title, env, a.thumbnail, a.github, a.repo)
        bad = find_keys([os.path.join(folder, f) for f in files])
        if bad:
            raise SystemExit("STOP: these files contain something that looks like an API key; nothing was "
                             f"uploaded: {', '.join(os.path.relpath(b, folder) for b in bad)}")
        print(f"Files for {a.repo} ({len(files)}):")
        for f in files:
            print("  ", f)
        print("\nrequirements.txt:\n" + "".join("   " + x + "\n" for x in requirements().splitlines()))
        if not a.no_check:
            print("Checking that the Space can install these packages (pip resolves them, installs nothing; "
                  "1-3 min) ...")
            problem = check_install(os.path.join(folder, "requirements.txt"))
            if problem:
                raise SystemExit("STOP: the Space's build would fail; nothing was uploaded. pip says:\n" + problem)
            print("   OK: the Space's install resolves.\n")

        print("Space card (README.md) header:")
        with open(os.path.join(folder, "README.md"), encoding="utf-8") as f:
            print("".join("   " + x for x in f.read().split("\n---\n")[0].splitlines(True)) + "\n")
        secrets = {k: env[k] for k in SECRET_KEYS if env.get(k)}
        variables = {k: (v if v is not None else env.get(k)) for k, v in VARIABLES.items()}
        variables = {k: v for k, v in variables.items() if v}
        if a.settings:
            print("Space secrets (values hidden):", ", ".join(secrets) or "none found in .env")
            print("Space variables:", ", ".join(f"{k}={v}" for k, v in variables.items()))
            if "OPENAI_API_KEY" in env:
                print("OPENAI_API_KEY is NOT sent (the public app uses free models only).")
        if a.dry_run:
            print("\nDry run: nothing was uploaded.")
            return 0

        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
        token, src = deploy_token()
        if not token:
            raise SystemExit("No Hugging Face login found; nothing was uploaded. Run `hf auth login` "
                             "with a Write token (choose 'paste a token').")
        api = HfApi(token=token)
        me = api.whoami()
        role = me.get("auth", {}).get("accessToken", {}).get("role", "unknown")
        print(f"Uploading as {me.get('name')} (token from {src}, type: {role}).")
        owner = a.repo.split("/")[0]
        if me.get("name") != owner and owner not in [o.get("name") for o in me.get("orgs", [])]:
            raise SystemExit(f"STOP: you are logged in as {me.get('name')}, but the Space belongs to {owner}; "
                             "nothing was uploaded.")
        try:
            api.create_repo(a.repo, repo_type="space", space_sdk="gradio", private=a.private, exist_ok=True)
            api.upload_folder(folder_path=folder, repo_id=a.repo, repo_type="space",
                              commit_message="Deploy Quantum Information Agent",
                              delete_patterns=["*.py", "data/*.csv"])  # files removed from the whitelist go too
            if a.settings:
                for k, v in secrets.items():
                    api.add_space_secret(a.repo, k, v)
                for k, v in variables.items():
                    api.add_space_variable(a.repo, k, v)
        except HfHubHTTPError as e:
            if getattr(e.response, "status_code", None) == 403:
                raise SystemExit(NOT_ALLOWED.format(src=src, repo=a.repo)) from None
            raise
        print(f"\nDone: https://huggingface.co/spaces/{a.repo} (it builds for a few minutes).")
        return 0
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
