import json
import sys

def find_urls(obj, urls):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == 'downloadUrl' and 'html' in v:
                urls.append(v)
            else:
                find_urls(v, urls)
    elif isinstance(obj, list):
        for item in obj:
            find_urls(item, urls)

files = [
    r"C:\Users\Vaibhav\.gemini\antigravity\brain\76691dcc-169c-427e-bcb7-0d6be2444b08\.system_generated\steps\113\output.txt",
    r"C:\Users\Vaibhav\.gemini\antigravity\brain\76691dcc-169c-427e-bcb7-0d6be2444b08\.system_generated\steps\114\output.txt"
]

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            data = json.load(file)
            urls = []
            find_urls(data, urls)
            print(f"{f}: {urls}")
    except Exception as e:
        print(f"Error reading {f}: {e}")
