import json
import sys

files = [
    r"C:\Users\Vaibhav\.gemini\antigravity\brain\76691dcc-169c-427e-bcb7-0d6be2444b08\.system_generated\steps\112\output.txt",
    r"C:\Users\Vaibhav\.gemini\antigravity\brain\76691dcc-169c-427e-bcb7-0d6be2444b08\.system_generated\steps\113\output.txt",
    r"C:\Users\Vaibhav\.gemini\antigravity\brain\76691dcc-169c-427e-bcb7-0d6be2444b08\.system_generated\steps\114\output.txt"
]

with open('urls.txt', 'w') as out_f:
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                try:
                    url = data['outputComponents'][0]['design']['screens'][0]['htmlCode']['downloadUrl']
                except KeyError:
                    url = data['outputComponents'][0]['htmlCode']['downloadUrl']
                out_f.write(url + '\n')
        except Exception as e:
            out_f.write(f"Error: {e}\n")
