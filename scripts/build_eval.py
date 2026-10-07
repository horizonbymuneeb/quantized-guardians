"""Build the evaluation set: benchmarks + character-level evasion variants."""
import pandas as pd, random, json
random.seed(0)
D = "data/"
rows = []
def add(src, texts, labels):
    for t, l in zip(texts, labels):
        if isinstance(t, str) and t.strip():
            rows.append({"src": src, "text": t, "label": int(l)})

df = pd.read_parquet(D + "deepset_test.parquet"); add("deepset", df.text, df.label)
for i in (1, 2, 3):
    df = pd.read_parquet(D + f"notinject{i}.parquet")
    col = "prompt" if "prompt" in df.columns else df.columns[0]
    add("notinject", df[col], [0] * len(df))
df = pd.read_csv(D + "jb_test.csv"); add("jailbreak_cls", df.prompt, (df.type == "jailbreak").astype(int))
df = pd.read_parquet(D + "safeguard_test.parquet")
df = pd.concat([df[df.label==l].sample(500, random_state=0) for l in (0,1)])
add("safeguard", df.text, df.label)

# Evasion variants on 300 attack prompts from safeguard (character-level, Hackett et al. style)
attacks = [r for r in rows if r["src"] == "safeguard" and r["label"] == 1]
attacks = random.sample(attacks, 300)
HOMO = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "i": "\u0456", "x": "\u0445", "y": "\u0443"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
def homoglyph(t): return "".join(HOMO[c] if c in HOMO and random.random() < 0.5 else c for c in t)
def zerowidth(t): return "".join(c + ("\u200b" if c.isalpha() and random.random() < 0.3 else "") for c in t)
def leet(t): return "".join(LEET[c] if c in LEET and random.random() < 0.5 else c for c in t)
for name, f in [("evade_homoglyph", homoglyph), ("evade_zerowidth", zerowidth), ("evade_leet", leet)]:
    for r in attacks:
        rows.append({"src": name, "text": f(r["text"]), "label": 1})
out = pd.DataFrame(rows)
out.to_parquet(D + "evalset.parquet")
print(out.groupby(["src", "label"]).size())
print(len(out))
