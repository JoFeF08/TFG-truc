import os

replacements = [
    ("from joc.entorn", "from joc.entorn"),
    ("import joc.entorn", "import joc.entorn"),
    ("from joc.vista", "from joc.vista"),
    ("import joc.vista", "import joc.vista"),
    ("from joc.controlador", "from joc.controlador"),
    ("import joc.controlador", "import joc.controlador"),
    ("from RL.models", "from RL.models"),
    ("import RL.models", "import RL.models"),
    ("from RL.entrenament", "from RL.entrenament"),
    ("import RL.entrenament", "import RL.entrenament"),
]

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        if ".venv" in root or ".git" in root or "__pycache__" in root or "build" in root or "dist" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    new_content = content
                    for old, new in replacements:
                        # Only replace if it's not already joc.entorn or RL.models
                        if old in new_content and not new in new_content:
                            # Actually, a simple replace could work, but let's be safe.
                            pass
                            
                    # A safer replace
                    for old, new in replacements:
                        new_content = new_content.replace(old, new)
                    
                    # Fix potential double replaces if it was already updated
                    new_content = new_content.replace("joc.", "joc.")
                    new_content = new_content.replace("RL.", "RL.")
                    
                    if new_content != content:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"Updated {path}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    process_directory(r"c:\Users\ferri\Documents\ProjectesCodi\TFG-truc")

