import os


def create_consolidated_markdown(root_dir=".", output_file="codebase.md"):
    """Generate consolidated markdown from all source files."""

    exclude_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', 'output', 'resources', '.idea', 'backup'}

    with open(output_file, 'w') as out:
        out.write("# PyPATools - Complete Codebase\n\n")

        for root, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in sorted(files):
                if any(file.endswith(ext) for ext in ['.py', '.yml', '.md', '.txt']):
                    filepath = os.path.join(root, file)
                    relpath = os.path.relpath(filepath, root_dir)

                    out.write(f"## {relpath}\n")
                    out.write(
                        "```" + ("python" if file.endswith('.py') else "yaml" if file.endswith('.yml') else "") + "\n")

                    try:
                        with open(filepath, 'r') as f:
                            out.write(f.read())
                    except Exception as e:
                        out.write(f"[Error reading file: {e}]")

                    out.write("\n```\n\n")

    print(f"Generated {output_file}")


# Run it
create_consolidated_markdown()
