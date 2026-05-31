import os
import re

JAVA_ROOT = r"d:\Project\NanoAgent\nanoagent-java"

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    original = content

    # Remove lombok imports
    content = re.sub(r'^import lombok\.\w+;\s*\n', '', content, flags=re.MULTILINE)

    # Extract class name
    class_match = re.search(r'public\s+class\s+(\w+)', content)
    if not class_match:
        class_match = re.search(r'class\s+(\w+)', content)
    class_name = class_match.group(1) if class_match else "Unknown"

    # Check if @Slf4j is present
    has_slf4j = '@Slf4j' in content
    has_req_args = '@RequiredArgsConstructor' in content
    has_data = '@Data' in content and '@Builder' not in content  # Simple @Data only
    has_builder = '@Builder' in content

    # Remove annotations
    content = re.sub(r'@Slf4j\s*\n', '', content)
    content = re.sub(r'@RequiredArgsConstructor\s*\n', '', content)

    # Add imports if needed
    imports_to_add = []
    if has_slf4j:
        if 'import org.slf4j.Logger;' not in content:
            imports_to_add.append('import org.slf4j.Logger;')
        if 'import org.slf4j.LoggerFactory;' not in content:
            imports_to_add.append('import org.slf4j.LoggerFactory;')

    if imports_to_add:
        # Find the last import or package line
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            if line.startswith('import '):
                insert_idx = i + 1
            elif line.startswith('package '):
                insert_idx = i + 1

        for imp in reversed(imports_to_add):
            lines.insert(insert_idx, imp)
        content = '\n'.join(lines)

    # Add Logger field if @Slf4j was present
    if has_slf4j:
        # Find the class declaration line
        class_pattern = r'(public\s+class\s+' + class_name + r'\s*\{)'
        content = re.sub(
            class_pattern,
            r'\1\n\n    private static final Logger log = LoggerFactory.getLogger(' + class_name + r'.class);',
            content
        )

    # Add constructor if @RequiredArgsConstructor was present
    if has_req_args:
        # Find all final fields
        final_fields = re.findall(r'private\s+final\s+(\w+(?:<\w+>)?)\s+(\w+)\s*;', content)

        if final_fields:
            params = ', '.join(f'{ftype} {fname}' for ftype, fname in final_fields)
            assignments = '\n'.join(f'        this.{fname} = {fname};' for _, fname in final_fields)

            constructor = f'''
    public {class_name}({params}) {{
{assignments}
    }}'''

            # Insert after the last final field
            last_field = final_fields[-1][1]
            pattern = r'(private\s+final\s+\S+\s+' + last_field + r'\s*;)'
            content = re.sub(pattern, r'\1' + constructor, content)

    if content != original:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filepath}")
        return True
    return False

count = 0
for root, dirs, files in os.walk(JAVA_ROOT):
    for f in files:
        if f.endswith('.java'):
            filepath = os.path.join(root, f)
            if process_file(filepath):
                count += 1

print(f"\nTotal files fixed: {count}")