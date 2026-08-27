"""Generate the player app's self-hosted utility layer.

SUPERSEDED. This was the one-shot migration that let web/index.html drop the Tailwind
CDN. The page has since been rebuilt on a semantic component stylesheet and no longer
carries Tailwind-shaped class names, so re-running this would inject a layer nothing
uses. Kept only as the record of how the CDN was removed.

Emits exactly the utilities web/index.html and web/app.js used at the time, so the page
could render correctly offline. Selectors are escaped programmatically because
hand-escaping `.bg-[#062b21]/85` is a typo farm.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WHITE = {
    "40": "0.40",
    "45": "0.45",
    "50": "0.50",
    "55": "0.55",
    "60": "0.60",
    "65": "0.65",
    "70": "0.70",
    "80": "0.80",
    "85": "0.85",
}

RULES: dict[str, str] = {
    # display / position
    "block": "display:block",
    "flex": "display:flex",
    "grid": "display:grid",
    "hidden": "display:none",
    "relative": "position:relative",
    "absolute": "position:absolute",
    "fixed": "position:fixed",
    "sticky": "position:sticky",
    "inset-0": "top:0;right:0;bottom:0;left:0",
    "inset-x-0": "left:0;right:0",
    "top-0": "top:0",
    "bottom-0": "bottom:0",
    "z-20": "z-index:20",
    "z-30": "z-index:30",
    "overflow-hidden": "overflow:hidden",
    "object-contain": "object-fit:contain",
    # flex / grid
    "flex-1": "flex:1 1 0%",
    "flex-col": "flex-direction:column",
    "flex-wrap": "flex-wrap:wrap",
    "items-center": "align-items:center",
    "justify-between": "justify-content:space-between",
    "justify-center": "justify-content:center",
    "place-items-center": "place-items:center",
    "grid-cols-4": "grid-template-columns:repeat(4,minmax(0,1fr))",
    "grid-cols-[auto_repeat(8,1fr)]": "grid-template-columns:auto repeat(8,minmax(0,1fr))",
    # gap
    "gap-1.5": "gap:0.375rem",
    "gap-2": "gap:0.5rem",
    "gap-2.5": "gap:0.625rem",
    "gap-3": "gap:0.75rem",
    "gap-5": "gap:1.25rem",
    # margin / padding
    "mx-auto": "margin-left:auto;margin-right:auto",
    "mb-1": "margin-bottom:0.25rem",
    "mb-1.5": "margin-bottom:0.375rem",
    "mb-3": "margin-bottom:0.75rem",
    "mt-3": "margin-top:0.75rem",
    "mt-4": "margin-top:1rem",
    "mt-5": "margin-top:1.25rem",
    "p-4": "padding:1rem",
    "pb-4": "padding-bottom:1rem",
    "pb-40": "padding-bottom:10rem",
    "pt-5": "padding-top:1.25rem",
    "px-4": "padding-left:1rem;padding-right:1rem",
    "py-1": "padding-top:0.25rem;padding-bottom:0.25rem",
    "py-3": "padding-top:0.75rem;padding-bottom:0.75rem",
    "py-5": "padding-top:1.25rem;padding-bottom:1.25rem",
    "py-7": "padding-top:1.75rem;padding-bottom:1.75rem",
    # sizing
    "h-8": "height:2rem",
    "h-9": "height:2.25rem",
    "h-14": "height:3.5rem",
    "w-6": "width:1.5rem",
    "w-8": "width:2rem",
    "w-9": "width:2.25rem",
    "w-full": "width:100%",
    "max-w-xs": "max-width:20rem",
    "max-w-3xl": "max-width:48rem",
    "max-h-[46vh]": "max-height:46vh",
    "min-h-screen": "min-height:100vh",
    # type
    "text-[11px]": "font-size:11px;line-height:1.45",
    "text-xs": "font-size:0.75rem;line-height:1rem",
    "text-sm": "font-size:0.875rem;line-height:1.25rem",
    "text-base": "font-size:1rem;line-height:1.5rem",
    "text-lg": "font-size:1.125rem;line-height:1.75rem",
    "text-xl": "font-size:1.25rem;line-height:1.75rem",
    "text-2xl": "font-size:1.5rem;line-height:2rem",
    "font-semibold": "font-weight:600",
    "font-bold": "font-weight:700",
    "font-extrabold": "font-weight:800",
    "font-black": "font-weight:900",
    "leading-none": "line-height:1",
    "leading-tight": "line-height:1.25",
    "leading-relaxed": "line-height:1.625",
    "tracking-wider": "letter-spacing:0.05em",
    "uppercase": "text-transform:uppercase",
    "tabular-nums": "font-variant-numeric:tabular-nums",
    "text-center": "text-align:center",
    "text-start": "text-align:start",
    "text-end": "text-align:end",
    # colour
    "text-[var(--brass-400)]": "color:var(--brass-400)",
    "text-[var(--felt-900)]": "color:var(--felt-900)",
    "text-[var(--rose-400)]": "color:var(--rose-400)",
    "bg-[var(--brass-400)]": "background-color:var(--brass-400)",
    "bg-[#062b21]/85": "background-color:rgba(6,43,33,0.85)",
    "bg-[#062b21]/95": "background-color:rgba(6,43,33,0.95)",
    "bg-black/15": "background-color:rgba(0,0,0,0.15)",
    "bg-black/30": "background-color:rgba(0,0,0,0.30)",
    "bg-black/55": "background-color:rgba(0,0,0,0.55)",
    # borders
    "border": "border-width:1px;border-style:solid",
    "border-4": "border-width:4px;border-style:solid",
    "border-b": "border-bottom-width:1px;border-bottom-style:solid",
    "border-t": "border-top-width:1px;border-top-style:solid",
    "border-white/10": "border-color:rgba(255,255,255,0.10)",
    "border-white/20": "border-color:rgba(255,255,255,0.20)",
    "border-t-[var(--brass-400)]": "border-top-color:var(--brass-400)",
    "rounded-full": "border-radius:9999px",
    "rounded-xl": "border-radius:0.75rem",
    "rounded-2xl": "border-radius:1rem",
    # effects / interaction
    "cursor-pointer": "cursor:pointer",
    "select-none": "user-select:none",
    "backdrop-blur": "backdrop-filter:blur(12px)",
    "backdrop-blur-sm": "backdrop-filter:blur(4px)",
    "animate-spin": "animation:spin 1s linear infinite",
}

for shade, alpha in WHITE.items():
    RULES[f"text-white/{shade}"] = f"color:rgba(255,255,255,{alpha})"

COMPOUND = {
    "space-y-2": "> * + * { margin-top:0.5rem; }",
    "space-y-3": "> * + * { margin-top:0.75rem; }",
    "space-y-4": "> * + * { margin-top:1rem; }",
    "divide-y": "> * + * { border-top-width:1px; border-top-style:solid; }",
    "divide-white/5": "> * + * { border-top-color:rgba(255,255,255,0.05); }",
}

# The skip link only becomes visible while focused.
FOCUS = {
    "focus:not-sr-only": (
        "position:static;width:auto;height:auto;padding:0;margin:0;"
        "overflow:visible;clip:auto;white-space:normal"
    ),
    "focus:absolute": "position:absolute",
    "focus:z-50": "z-index:50",
    "focus:m-3": "margin:0.75rem",
    "focus:rounded-lg": "border-radius:0.5rem",
    "focus:bg-white": "background-color:#ffffff",
    "focus:px-4": "padding-left:1rem;padding-right:1rem",
    "focus:py-2": "padding-top:0.5rem;padding-bottom:0.5rem",
    "focus:text-emerald-950": "color:#022c22",
}

SMALL = {
    "sm:hidden": "display:none",
    "sm:p-5": "padding:1.25rem",
    "sm:pb-8": "padding-bottom:2rem",
    "sm:gap-3": "gap:0.75rem",
    "sm:items-end": "align-items:end",
    "sm:grid-cols-[auto_1fr]": "grid-template-columns:auto minmax(0,1fr)",
}


def escape(name: str) -> str:
    """CSS-escape a class name so `bg-[#062b21]/85` becomes a valid selector."""
    return re.sub(r"([^a-zA-Z0-9_-])", r"\\\1", name)


def build() -> str:
    lines = [
        "    /* ---- Utility layer ------------------------------------------------",
        "       Hand-written replacements for the Tailwind utilities this page uses.",
        "       Generated once by scripts/gen_utilities.py; the page ships no CDN",
        "       script, so it renders identically offline or behind a proxy. */",
        "    *, *::before, *::after { box-sizing: border-box; border-width: 0; border-style: solid; }",
        "    body, h1, h2, h3, p, figure, blockquote, dl, dd { margin: 0; }",
        "    h1, h2, h3 { font-size: inherit; font-weight: inherit; }",
        "    button, input, select, textarea { font: inherit; color: inherit; margin: 0; }",
        "    button { background: none; cursor: pointer; }",
        "    img, canvas, svg, video { display: block; max-width: 100%; }",
        "    table { border-collapse: collapse; }",
        "    summary { cursor: pointer; }",
        "    @keyframes spin { to { transform: rotate(360deg); } }",
        "",
    ]
    for name, body in RULES.items():
        lines.append(f"    .{escape(name)} {{ {body}; }}")
    for name, body in COMPOUND.items():
        lines.append(f"    .{escape(name)} {body}")
    for name, body in FOCUS.items():
        lines.append(f"    .{escape(name)}:focus {{ {body}; }}")
    lines.append("")
    lines.append("    @media (min-width: 640px) {")
    for name, body in SMALL.items():
        lines.append(f"      .{escape(name)} {{ {body}; }}")
    lines.append("    }")
    return "\n".join(lines)


def main() -> None:
    path = REPO / "web" / "index.html"
    html = path.read_text(encoding="utf-8")

    marker = "    .sr-only {"
    assert marker in html, "anchor for the utility layer not found"
    html = html.replace(marker, build() + "\n\n" + marker, 1)

    html = html.replace('  <script src="https://cdn.tailwindcss.com"></script>\n', "")
    html = html.replace(
        "  <script>\n    tailwind.config = { corePlugins: { preflight: true } };\n  </script>\n",
        "",
    )

    path.write_text(html, encoding="utf-8")
    print(f"utility layer written: {len(RULES) + len(COMPOUND) + len(FOCUS) + len(SMALL)} classes")
    print("tailwind cdn removed:", "cdn.tailwindcss.com" not in html)


if __name__ == "__main__":
    main()
