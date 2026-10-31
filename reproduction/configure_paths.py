"""Configure paths in the published local-source copy; dry run by default."""
import argparse
import ast
import io
import json
import tokenize
from pathlib import Path


def path_mapping(root, data_root, spacy_model):
    return sorted([
        ('/home/peng/PythonProjects/momask-codes-main/models/en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0', spacy_model),
        ('/home/ubuntu/peng/momask-codes-main/models/en_core_web_sm-3.5.0/en_core_web_sm/en_core_web_sm-3.5.0', spacy_model),
        ('/home/peng/PythonProjects/T2M-GPT-main', str(root)),
        ('/home/peng/PythonProjects/Dataset/HumanML3D', str(data_root / 'HumanML3D')),
        ('/home/peng/PythonProjects/Human-Datasets/HumanML3D', str(data_root / 'HumanML3D')),
        ('../Dataset/BABEL-TEACH', str(data_root / 'BABEL_TEACH')),
        ('../Dataset', str(data_root)),
        ('../dataset/HumanML3D', str(data_root / 'HumanML3D')),
        ('./dataset/KIT-ML', str(data_root / 'KIT-ML')),
        ('./dataset/BABEL_TEACH', str(data_root / 'BABEL_TEACH')),
    ], key=lambda item: len(item[0]), reverse=True)


def relocate(value, mappings):
    if not isinstance(value, str):
        return value
    for old, new in mappings:
        if value == old or value.startswith(old + '/'):
            return new + value[len(old):]
    return value


def rewrite_python(text, mappings):
    offsets = [0]
    for line in text.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    edits = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type != tokenize.STRING:
            continue
        try:
            value = ast.literal_eval(token.string)
        except (ValueError, SyntaxError):
            continue
        replacement = relocate(value, mappings)
        if isinstance(value, str) and replacement != value:
            begin = offsets[token.start[0] - 1] + token.start[1]
            end = offsets[token.end[0] - 1] + token.end[1]
            edits.append((begin, end, repr(replacement)))
    updated = text
    for begin, end, replacement in reversed(edits):
        updated = updated[:begin] + replacement + updated[end:]
    if edits:
        ast.parse(updated)
    return updated


def rewrite_json(text, mappings):
    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v) for v in value]
        return relocate(value, mappings)
    original = json.loads(text)
    updated = walk(original)
    return text if updated == original else json.dumps(updated, indent=4, ensure_ascii=False) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parent / 'local')
    parser.add_argument('--data-root', type=Path, help='Parent containing HumanML3D, KIT-ML, and/or BABEL_TEACH')
    parser.add_argument('--spacy-model', default='en_core_web_sm')
    parser.add_argument('--write', action='store_true', help='Apply changes; omitted means dry run')
    args = parser.parse_args()
    root = args.source_root.expanduser().resolve()
    data_root = (args.data_root or root.parent / 'Dataset').expanduser().resolve()
    mappings = path_mapping(root, data_root, args.spacy_model)
    manifest = json.loads((Path(__file__).resolve().parent / 'source-manifest.json').read_text())
    prefix = 'reproduction/local/'
    changed = []
    for entry in manifest['files']:
        destination = entry['destination']
        if not destination.startswith(prefix):
            continue
        rel = Path(destination[len(prefix):])
        if rel.suffix not in {'.py', '.json'}:
            continue
        path = (root / rel).resolve()
        if root not in path.parents:
            raise ValueError('Source path escapes source root: ' + str(rel))
        text = path.read_text(encoding='utf-8')
        updated = rewrite_python(text, mappings) if path.suffix == '.py' else rewrite_json(text, mappings)
        if updated != text:
            changed.append(rel.as_posix())
            if args.write:
                path.write_text(updated, encoding='utf-8')
    print(json.dumps({'mode': 'write' if args.write else 'dry-run', 'source_root': str(root),
                      'data_root': str(data_root), 'changed_files': changed}, indent=2))


if __name__ == '__main__':
    main()
