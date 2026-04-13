import os
import sqlite3
from typing import Annotated, List, Literal, TypedDict, Any, Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

from .loader import SkillRegistry
from .tools import DEFAULT_TOOLS, set_active_path
from .logger import logger

load_dotenv()
registry = SkillRegistry()

llm = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "gpt-4o"),
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    active_skill: Optional[str]
    router_reasoning: Optional[str]

def router_node(state: AgentState):
    logger.info("--- 🚦 Node: Router ---")
    messages = state["messages"]
    
    registry.refresh()
    skills = registry.list_skills()
    
    if not isinstance(messages[-1], HumanMessage) or not skills:
        return {"active_skill": None}

    skill_list_str = "\n".join([f"- {s['name']}: {s['description']}" for s in skills])
    user_input = messages[-1].content
    current_skill = state.get("active_skill")

    prompt = f"""
    Role: Expert Skill Router
    
    You have a set of specialist agents (Skills) available:
    {skill_list_str}
    
    Current Active Skill: {current_skill if current_skill else "None"}
    User Request: "{user_input}"
    
    TASK: Decide which skill to activate.
    
    IMPORTANT RULES:
    1. Analyze the user's intent CAREFULLY.
    2. If the user explicitly mentions a skill name (e.g., "use password generator"), SWITCH to that skill.
    3. If the current skill CANNOT handle the user's request, you MUST switch.
    4. Only switch if the new skill is BETTER SUITED than the current one.
    5. For general chat, greetings, or questions unrelated to specific tools, keep current or use "default".
    
    Examples of switching:
    - User: "I need a password" -> Switch to password_generator
    - User: "Draw a chart" -> Keep chart_maker (already active)
    - User: "What's the weather?" -> Switch to url_reader or web_searcher
    
    OUTPUT: Return ONLY the exact name of the skill to activate, or "default" if no specific skill is needed.
    Do NOT explain your reasoning.
    """
    
    response = llm.invoke(prompt).content.strip()
    
    target = response if registry.get_skill(response) else None
    
    if target != current_skill:
        logger.info(f"🔄 SWITCHING SKILL: {current_skill} -> {target}")
    else:
        logger.info(f"✅ KEEPING SKILL: {target}")
    
    return {
        "active_skill": target,
        "router_reasoning": f"Routed to: {target}"
    }

def agent_node(state: AgentState):
    logger.info("--- 🤖 Node: Agent ---")
    skill_name = state.get("active_skill")
    skill = registry.get_skill(skill_name)
    
    logger.info(f"Current Active Skill: {skill_name}")

    system_text = "You are a helpful AI assistant."
    if skill:
        set_active_path(skill.root_path)
        
        system_text += f"\n\n=== ACTIVE SKILL: {skill.name} ===\n{skill.instructions}"
        
        ref_dir = skill.root_path / "references"
        if ref_dir.exists():
            files = [f.name for f in ref_dir.glob("*") if f.is_file() and not f.name.startswith(".")]
            if files:
                system_text += "\n\n=== AVAILABLE REFERENCES (Knowledge Base) ===\n"
                system_text += "You have access to the following files in the 'references' folder:\n"
                for f in files:
                    system_text += f"- {f}\n"
                system_text += "Use the `read_reference` tool to read their content if needed.\n"
        
        logger.debug(f"Injecting instructions for {skill.name}")
    else:
        set_active_path(None)

    model = llm.bind_tools(DEFAULT_TOOLS)
    full_messages = [SystemMessage(content=system_text)] + state["messages"]
    
    logger.debug("Invoking LLM...")
    response = model.invoke(full_messages)
    
    if response.tool_calls:
        logger.info(f"🛠️ Agent requested tools: {response.tool_calls}")
    else:
        logger.info("🗣️ Agent responded with text.")
        
    return {"messages": [response]}

def tool_router(state: AgentState) -> Literal["tools", "end"]:
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "tools"
    return "end"

def build_graph(db_path="memory.sqlite"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(DEFAULT_TOOLS))
    
    workflow.set_entry_point("router")
    workflow.add_edge("router", "agent")
    workflow.add_conditional_edges("agent", tool_router, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile(checkpointer=memory)
