from __future__ import annotations

from pathlib import Path


PATH = Path("tests/test_self_hosted_workflow_policy.py")
text = PATH.read_text(encoding="utf-8")

old = '''def _dispatch_only_workflow(text: str) -> bool:
    return _single_event_workflow(text, "workflow_dispatch")
'''
new = '''def _trigger_event_block(text: str, event: str) -> str:
    prefix = text.split("permissions:", maxsplit=1)[0]
    lines = prefix.splitlines(keepends=True)
    marker = f"  {event}:"
    try:
        start = next(
            index for index, line in enumerate(lines) if line.rstrip() == marker
        )
    except StopIteration:
        return ""
    collected = [lines[start]]
    for line in lines[start + 1 :]:
        if re.match(r"^  [A-Za-z0-9_-]+:\\s*$", line):
            break
        collected.append(line)
    return "".join(collected)


def _dispatch_only_workflow(text: str) -> bool:
    return _single_event_workflow(text, "workflow_dispatch")
'''
if text.count(old) != 1:
    raise SystemExit("dispatch helper insertion point changed")
text = text.replace(old, new, 1)

old = '''    if prefix.count(request_path) != 1:
        errors.append("reviewed request path must occur exactly once in trigger block")
'''
new = '''    push_block = _trigger_event_block(workflow_text, "push")
    if push_block.count(request_path) != 1:
        errors.append("reviewed request path must occur exactly once in push trigger")
'''
if text.count(old) != 1:
    raise SystemExit("reviewed-file trigger count check changed")
text = text.replace(old, new, 1)

PATH.write_text(text, encoding="utf-8")
