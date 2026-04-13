import sys
import os
import subprocess
from pathlib import Path
from typing import Optional, List
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from .logger import logger

ACTIVE_SKILL_PATH: Optional[Path] = None

def set_active_path(path: Optional[Path]):
    global ACTIVE_SKILL_PATH
    ACTIVE_SKILL_PATH = path
    if path:
        logger.info(f"Context switched to skill path: {path}")

class RunScriptInput(BaseModel):
    script_name: str = Field(..., description="Filename (e.g., 'magic.py')")
    args: List[str] = Field(default_factory=list, description="Arguments")

@tool(args_schema=RunScriptInput)
def run_skill_script(script_name: str, args: Optional[List[str]] = None) -> str:
    """Execute a Python script in 'scripts' folder."""
    
    logger.info(f"🛠️ Tool Call: run_skill_script | Script: {script_name} | Args: {args}")

    if not ACTIVE_SKILL_PATH:
        msg = "❌ Error: No active skill path set."
        logger.error(msg)
        return msg

    scripts_dir = (ACTIVE_SKILL_PATH / "scripts").resolve()
    script_file = (scripts_dir / script_name).resolve()
    
    if not scripts_dir.exists():
        msg = f"❌ Error: Directory not found: {scripts_dir}"
        logger.error(msg)
        return msg

    if not script_file.exists():
        existing = [f.name for f in scripts_dir.glob("*")]
        msg = f"❌ Error: Script not found: {script_file}. Existing files: {existing}"
        logger.error(msg)
        return msg

    try:
        env = os.environ.copy()
        python_exe = sys.executable
        safe_args = args if args is not None else []
        
        cmd = [python_exe, str(script_file)] + [str(a) for a in safe_args]
        
        logger.debug(f"⚡ Executing command: {' '.join(cmd)}")
        logger.debug(f"📂 Working Directory: {scripts_dir}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=60,
            cwd=str(scripts_dir),
            env=env
        )
        
        output = result.stdout.strip() if result.stdout else "[No Output]"
        
        if result.returncode == 0:
            logger.info(f"✅ Script Success. Output length: {len(output)}")
            logger.debug(f"Script Output:\n{output}")
            return f"Success:\n{output}"
        else:
            logger.error(f"❌ Script Failed (Code {result.returncode})")
            logger.error(f"Error Output:\n{output}")
            return f"Error (Code {result.returncode}):\n{output}"

    except Exception as e:
        logger.critical(f"💥 Exception in run_skill_script: {str(e)}", exc_info=True)
        return f"System Execution Error: {str(e)}"

class ReadRefInput(BaseModel):
    filename: str = Field(..., description="The name of the file to read (e.g., 'guidelines.md')")

@tool(args_schema=ReadRefInput)
def read_reference(filename: str) -> str:
    """
    Read the content of a reference file located in the 'references' folder of the active skill.
    Use this when you need to look up static information, documentation, or guidelines provided by the skill.
    """
    if not ACTIVE_SKILL_PATH:
        return "Error: No active skill context."

    ref_dir = ACTIVE_SKILL_PATH / "references"
    file_path = ref_dir / filename

    logger.info(f"📖 Tool Call: read_reference | File: {filename}")

    if not ref_dir.exists():
        return "Error: This skill has no 'references' folder."

    if not file_path.exists():
        existing = [f.name for f in ref_dir.glob("*") if f.is_file()]
        return f"Error: File '{filename}' not found. Available files: {existing}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            if len(content) > 10000:
                return f"{content[:10000]}\n\n[Content truncated because it is too long]"
            return content
    except Exception as e:
        return f"Error reading file: {str(e)}"

DEFAULT_TOOLS = [run_skill_script, read_reference]
