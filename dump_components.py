import json
import sys

files = [
    r"C:\Users\Vaibhav\.gemini\antigravity\brain\76691dcc-169c-427e-bcb7-0d6be2444b08\.system_generated\steps\113\output.txt",
    r"C:\Users\Vaibhav\.gemini\antigravity\brain\76691dcc-169c-427e-bcb7-0d6be2444b08\.system_generated\steps\114\output.txt"
]

with open('output_dump.txt', 'w', encoding='utf-8') as out_f:
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                out_f.write(f"--- File: {f} ---\n")
                out_f.write(json.dumps(data['outputComponents'], indent=2))
                out_f.write("\n\n")
        except Exception as e:
            out_f.write(f"Error reading {f}: {e}\n")
