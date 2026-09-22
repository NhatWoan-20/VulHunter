import ast
import json
import textwrap
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data" / "processed" / "master_semantic_samples.jsonl"
OUTPUT = ROOT / "data" / "processed" / "master_graph_samples.jsonl"


class DocstringStripper(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:]
        return node


def strip_docstrings(code: str) -> str:
    dedented = textwrap.dedent(code)
    tree = ast.parse(dedented)
    tree = DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    stripped = ast.unparse(tree)
    return stripped if stripped.strip() else "pass"


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    rows = skipped = 0
    with INPUT.open("r", encoding="utf-8") as fin, OUTPUT.open("w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            row = json.loads(raw)
            try:
                row["code"] = strip_docstrings(row.get("code", ""))
            except Exception:
                skipped += 1
                continue
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows += 1

    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "rows": rows, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


