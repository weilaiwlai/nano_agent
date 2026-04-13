import frontmatter
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

SKILLS_ROOT = Path("skills")

class SkillMetadata(TypedDict):
    name: str
    description: str
    version: Optional[str]
    author: Optional[str]

class AgentSkill:
    def __init__(self, path: Path):
        self.root_path = path
        self.skill_file = path / "SKILL.md"
        self._load()

    def _load(self):
        if not self.skill_file.exists():
            raise FileNotFoundError(f"Missing SKILL.md in {self.root_path}")
        
        post = frontmatter.load(str(self.skill_file))
        
        self.metadata = SkillMetadata(
            name=post.metadata.get("name", self.root_path.name),
            description=post.metadata.get("description", "No description provided."),
            version=str(post.metadata.get("version", "1.0")),
            author=post.metadata.get("author", "Unknown")
        )
        self.instructions = post.content

    @property
    def name(self):
        return self.metadata["name"]

class SkillRegistry:
    def __init__(self):
        SKILLS_ROOT.mkdir(exist_ok=True)
        self.skills: Dict[str, AgentSkill] = {}
        self.refresh()

    def refresh(self):
        self.skills = {}
        for item in SKILLS_ROOT.iterdir():
            if item.is_dir() and (item / "SKILL.md").exists():
                try:
                    skill = AgentSkill(item)
                    self.skills[skill.name] = skill
                except Exception as e:
                    print(f"Error loading skill {item.name}: {e}")

    def get_skill(self, name: str) -> Optional[AgentSkill]:
        return self.skills.get(name)

    def list_skills(self) -> List[Dict]:
        return [s.metadata for s in self.skills.values()]
