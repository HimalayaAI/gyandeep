from __future__ import annotations
import re

def extract_think_and_final(text: str) -> tuple[str, str]:
    if not text:
        return "", ""

    lower = text.lower()
    think_text = ""
    final_text = text

    # 1. Handle <tool_call> tags (treat as thinking)
    if "<tool_call>" in lower:
        tool_call_texts = re.findall(r"<tool_call>(.*?)</tool_call>", text, flags=re.IGNORECASE | re.DOTALL)
        if tool_call_texts:
            think_text = "\n\n".join(t.strip() for t in tool_call_texts if t.strip())
            final_text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # 2. Handle <think> tags (without 'ing')
    if "<think>" in lower:
        if "</think>" in lower:
            think_texts = re.findall(r"<think>(.*?)</think>", final_text, flags=re.IGNORECASE | re.DOTALL)
            if think_texts:
                extracted = "\n\n".join(t.strip() for t in think_texts if t.strip())
                think_text = (think_text + "\n\n" + extracted).strip() if think_text else extracted
            final_text = re.sub(r"<think>.*?</think>", "", final_text, flags=re.IGNORECASE | re.DOTALL)
        else:
            # Unclosed <think> tag – treat first paragraph after tag as thinking
            after = re.split(r"<think>", final_text, flags=re.IGNORECASE, maxsplit=1)[-1]
            parts = re.split(r"\n\s*\n", after, maxsplit=1)
            if len(parts) > 1:
                extracted = parts[0].strip()
                think_text = (think_text + "\n\n" + extracted).strip() if think_text else extracted
                final_text = parts[1].strip()
            else:
                final_text = after.strip()
                # no extra think_text from unclosed tag

    # 3. Handle <thinking> tags (with 'ing')
    if "<thinking>" in lower:
        if "</thinking>" in lower:
            think_texts = re.findall(r"<thinking>(.*?)</thinking>", final_text, flags=re.IGNORECASE | re.DOTALL)
            if think_texts:
                extracted = "\n\n".join(t.strip() for t in think_texts if t.strip())
                think_text = (think_text + "\n\n" + extracted).strip() if think_text else extracted
            final_text = re.sub(r"<thinking>.*?</thinking>", "", final_text, flags=re.IGNORECASE | re.DOTALL)
        else:
            after = re.split(r"<thinking>", final_text, flags=re.IGNORECASE, maxsplit=1)[-1]
            parts = re.split(r"\n\s*\n", after, maxsplit=1)
            if len(parts) > 1:
                extracted = parts[0].strip()
                think_text = (think_text + "\n\n" + extracted).strip() if think_text else extracted
                final_text = parts[1].strip()
            else:
                final_text = after.strip()
                # no extra think_text from unclosed tag

    # 4. Handle <final> or <answer> tags
    final_match = re.search(r"<final>(.*?)</final>", final_text, flags=re.IGNORECASE | re.DOTALL)
    if not final_match:
        final_match = re.search(r"<answer>(.*?)</answer>", final_text, flags=re.IGNORECASE | re.DOTALL)
    if final_match:
        final_text = final_match.group(1).strip()
    else:
        # Remove any orphaned </final> or </answer> tags
        final_text = re.sub(r"</?(final|answer)>", "", final_text, flags=re.IGNORECASE).strip()

    # 5. Final cleanup – remove any leftover tag names (just in case)
    final_text = final_text.replace("<think>", "").replace("</think>", "")
    final_text = final_text.replace("<thinking>", "").replace("</thinking>", "")
    final_text = final_text.strip()

    # 6. Remove markdown bold markers
    final_text = final_text.replace("**", "")
    think_text = think_text.replace("**", "")

    return final_text, think_text
