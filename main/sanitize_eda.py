import os
import re

def sanitize_file(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Replace common problematic characters
    replacements = {
        '\u2013': '-', # en dash
        '\u2014': '-', # em dash
        '\u2026': '...', # ellipsis
        '\u2192': '->', # arrow
        '\u2713': '[DONE]', # checkmark
        '\u2212': '-', # mathematical minus
        '\u201d': '"', # right double quote
        '\u201c': '"', # left double quote
    }
    
    for char, rep in replacements.items():
        content = content.replace(char, rep)
    
    # Remove any remaining non-ASCII
    content = re.sub(r'[^\x00-\x7f]', '', content)
    
    with open(filepath, 'w', encoding='ascii') as f:
        f.write(content)
    print(f"Sanitized: {filepath}")

eda_dir = 'eda'
for filename in os.listdir(eda_dir):
    if filename.endswith('.py'):
        sanitize_file(os.path.join(eda_dir, filename))

# Also sanitize the main runner
sanitize_file('run_eda.py')
