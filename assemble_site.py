import os
import re

design_dir = r"C:\Users\Vaibhav\OneDrive\Desktop\tech-slaves\.stitch\designs"
public_dir = r"C:\Users\Vaibhav\OneDrive\Desktop\tech-slaves\site\public"

os.makedirs(public_dir, exist_ok=True)

nav_mappings = {
    'Dashboard': 'index.html',
    'Bottleneck Explorer': 'bottleneck.html',
    'Data Streams': 'data-streams.html',
    'Analytics': 'analytics.html',
    'API Docs': 'api.html'
}

for filename in os.listdir(design_dir):
    if not filename.endswith('.html'):
        continue
        
    filepath = os.path.join(design_dir, filename)
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # We want to replace href="#" with href="page.html" for specific links
    # The structure is roughly:
    # <a ... href="#">
    #   <span ...>icon</span>
    #   <span>Menu Name</span>
    # </a>
    
    # We can use regex to find <a> tags and check their content
    
    def replace_link(match):
        a_tag_start = match.group(1)
        inner_content = match.group(2)
        
        for name, link in nav_mappings.items():
            if f">{name}<" in inner_content:
                # Replace href="#" with href="link"
                new_start = re.sub(r'href="#"', f'href="{link}"', a_tag_start)
                return new_start + inner_content + "</a>"
                
        return match.group(0)
    
    # Match <a ... href="#">...</a> non-greedily
    pattern = r'(<a[^>]*href="#"[^>]*>)(.*?)(</a>)'
    
    modified_content = re.sub(pattern, replace_link, content, flags=re.DOTALL)
    
    out_path = os.path.join(public_dir, filename)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(modified_content)
        
    print(f"Processed {filename}")
