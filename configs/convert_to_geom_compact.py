import json

# Read geom_specification.json, skip comments, parse each JSON object,
# and output as compact JSONL (one geom spec per line)

with open("geom_specification.json", "r") as s:
    with open("geom_specification_compact.json", "w") as t:
        content = s.read()
        
        # Remove comment lines
        lines = [line for line in content.split("\n") if line.strip() and not line.strip().startswith("#")]
        
        # Parse and output each JSON object on one line
        buffer = ""
        brace_count = 0
        
        for line in lines:
            buffer += line + "\n"
            brace_count += line.count("{") - line.count("}")
            
            # When we close a top-level object, write it as compact JSON on one line
            if brace_count == 0 and "{" in buffer:
                try:
                    obj = json.loads(buffer)
                    t.write(json.dumps(obj, separators=(",", ":")) + "\n")
                    buffer = ""
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON: {e}")
                    print(f"Buffer: {buffer[:100]}...")
