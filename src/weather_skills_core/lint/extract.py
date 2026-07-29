"""AST extraction of a skill script's declaration surface.

The extractor reads the ``@weather_skill`` call, the ``_SKILL_VERSION``
constant, and the PEP 723 inline-metadata block from the script's source text.
The script is parsed, never imported: extraction works without the script's
dependencies installed and runs none of its code.

Literal declaration values are evaluated; a non-literal value (a name, an
f-string, a call) is recorded as dynamic and excluded from shape comparison,
with a note on the declaration.
"""

import ast
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from weather_skills_core import types

#: Sentinel for a declaration value that is not a literal in the source.
DYNAMIC = "<dynamic>"

#: The declaration constants, resolved by name off a ``types.<NAME>`` reference.
_TYPE_CONSTANTS = {name: value for name, value in vars(types).items() if name.isupper()}

_TOGGLE_KEYWORDS = (
    "start_time",
    "end_time",
    "date",
    "bbox",
    "variable",
    "workers",
    "title",
    "dims",
    "time_dim",
)

_BARE_TYPE_NAMES = {"int", "float", "str", "bool"}

# The PEP 723 inline-metadata block grammar (the regular expression given by
# the specification, anchored to the "script" block type by the extractor).
_PEP723_RE = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")

_REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize_requirement_name(requirement: str) -> str | None:
    """The normalized project name of one PEP 508 requirement string, or None."""
    m = _REQUIREMENT_NAME_RE.match(requirement)
    if m is None:
        return None
    return re.sub(r"[-_.]+", "-", m.group(1)).lower()


@dataclass(frozen=True)
class ArgShape:
    """The comparable CLI shape of one declared ``extra_args`` entry."""

    dest: str
    flags: tuple[str, ...] = ()
    positional: bool = False
    arity: str = "single"  # "single" | "append" | "store_true"
    nargs: object = None
    type_name: str | None = None  # None: the raw CLI string
    choices: tuple | None = None
    required: bool = False
    dynamic_keys: tuple[str, ...] = ()  # dict-spec keys whose values are not literals
    dynamic: bool = False  # the whole spec is not a recognized literal form

    @property
    def primary_flag(self) -> str | None:
        return self.flags[0] if self.flags else None

    @property
    def identity(self) -> str:
        """The name rules match across skills: the primary flag, or the dest."""
        return self.primary_flag or self.dest


@dataclass
class SkillDeclaration:
    """Everything extraction learned about one skill script."""

    skill_dir: Path
    script: Path
    name: str | None = None
    error: str | None = None  # set: the script could not be analyzed at all
    toggles: dict = field(default_factory=dict)  # toggle keyword -> literal value or DYNAMIC
    extra_args: dict[str, ArgShape] = field(default_factory=dict)
    extra_args_dynamic: bool = False  # the declared-flag set is not statically knowable
    input_names: list[str] | None = None
    has_input: bool = False
    input_arity: str = "single"
    has_output: bool = False
    writes_zarr: bool = False  # output_type is a zarr envelope type, not PNG
    sets_source: bool = False  # the script calls set_source() somewhere
    bare_validate_type: bool = False  # a validate_type() call passes no dims
    version_constant: bool = False
    version_passed: bool = False
    pep723_deps: list[str] | None = None  # None: no parseable script block
    notes: list[str] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        return self.name or self.skill_dir.name

    @property
    def key(self) -> str:
        """Collision-proof identity: the script path relative to the skill dir's parent.

        Two scripts in one skill directory, and two skill directories that
        pick the same display name, get distinct keys; this is the identity
        findings and per-skill scores are grouped by, where the display name
        can collide.
        """
        try:
            return str(self.script.relative_to(self.skill_dir.parent))
        except ValueError:
            return str(self.script)

    def toggle_enabled(self, keyword: str) -> bool:
        """True when a standard toggle keyword is declared with a non-off value.

        A dynamic value counts as enabled: the keyword is present and none of
        the off spellings (absent, ``False``, ``None``) are written literally.
        """
        if keyword not in self.toggles:
            return False
        value = self.toggles[keyword]
        if value is DYNAMIC:
            return True
        return value is not False and value is not None


def _literal(node):
    """The node's literal value, or DYNAMIC when it is not a plain literal.

    A ``types.<NAME>`` declaration constant is as static as the string it
    holds, so it resolves to its value rather than reading as dynamic --
    inside a literal tuple/list/dict too (``input_type=[types.ALL, types.ALL]``).
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return _type_constant(node)


def _type_constant(node):
    """Resolve a ``types.<NAME>`` reference, or a literal container of them, else DYNAMIC."""
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "types":
            return _TYPE_CONSTANTS.get(node.attr, DYNAMIC)
        return DYNAMIC
    if isinstance(node, ast.Tuple | ast.List):
        values = [_literal(element) for element in node.elts]
        if DYNAMIC in values:
            return DYNAMIC
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict) and None not in node.keys:
        items = [(_literal(k), _literal(v)) for k, v in zip(node.keys, node.values, strict=True)]
        if any(k is DYNAMIC or not isinstance(k, str) or v is DYNAMIC for k, v in items):
            return DYNAMIC
        return dict(items)
    return DYNAMIC


def _find_decorator_calls(tree: ast.Module) -> list[tuple[str, ast.Call]]:
    """Every ``@weather_skill(...)`` application, as ``(function name, call)`` in source order."""
    calls: list[tuple[str, ast.Call]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        for dec in node.decorator_list:
            if not isinstance(dec, ast.Call):
                continue
            func = dec.func
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name == "weather_skill":
                calls.append((node.name, dec))
    calls.sort(key=lambda pair: pair[1].lineno)
    return calls


def _spec_dest(names: list[str], spec: dict) -> str:
    """The dest argparse derives for one extra_args entry, mirroring its rule.

    Dashes become underscores for an optional only; a positional's dest is its
    name verbatim, which is what the decorator computes too.
    """
    if isinstance(spec.get("dest"), str):
        return spec["dest"]
    if not names[0].startswith("-"):
        return names[0]
    longs = [n for n in names if n.startswith("--")]
    return (longs[0] if longs else names[0]).lstrip("-").replace("-", "_")


def _shape_from_spec(node, notes: list[str]) -> ArgShape | None:
    """The comparable shape of one ``extra_args`` tuple, or None when unreadable.

    The tuple is an ``add_argument`` call: leading string elements are the
    flags (or a positional's name) and an optional trailing dict holds
    argparse's keywords.
    """
    if not isinstance(node, ast.Tuple | ast.List):
        notes.append(
            "extra_args entry is not a literal tuple of add_argument arguments; "
            "recorded as dynamic and skipped from shape comparison"
        )
        return None
    elements = list(node.elts)
    kwargs_node = elements.pop() if elements and isinstance(elements[-1], ast.Dict) else None
    names = []
    for element in elements:
        value = _literal(element)
        if not isinstance(value, str) or value is DYNAMIC:
            notes.append(
                "extra_args entry names a non-literal flag; recorded as dynamic and "
                "skipped from shape comparison"
            )
            return None
        names.append(value)
    if not names:
        notes.append("extra_args entry names no flag or positional; skipped")
        return None

    spec: dict = {}
    dynamic_keys: list[str] = []
    if kwargs_node is not None:
        for key_node, value_node in zip(kwargs_node.keys, kwargs_node.values, strict=True):
            key = _literal(key_node) if key_node is not None else DYNAMIC
            if key is DYNAMIC or not isinstance(key, str):
                notes.append(f"extra_args {names[0]!r}: a non-literal keyword was skipped")
                continue
            if key == "type":
                # A type value is a callable (``int``, ``float``), never a
                # literal; a plain name is the recognized form.
                if isinstance(value_node, ast.Name):
                    spec["type"] = value_node.id
                else:
                    dynamic_keys.append(key)
                continue
            value = _literal(value_node)
            if value is DYNAMIC:
                dynamic_keys.append(key)
                continue
            spec[key] = value

    dest = _spec_dest(names, spec)
    positional = not names[0].startswith("-")
    action = spec.get("action")
    if action == "store_true":
        arity = "store_true"
    elif action == "append":
        arity = "append"
    else:
        arity = "single"
    choices = spec.get("choices")
    if choices is not None:
        if isinstance(choices, list | tuple | set):
            choices = tuple(choices)
        else:
            notes.append(f"extra_args {dest!r}: 'choices' is not a list; ignored")
            choices = None
    if dynamic_keys:
        notes.append(
            f"extra_args {dest!r}: non-literal value(s) for "
            f"{', '.join(sorted(dynamic_keys))} recorded as dynamic and skipped "
            "from shape comparison"
        )
    return ArgShape(
        dest=dest,
        flags=() if positional else tuple(names),
        positional=positional,
        arity=arity,
        nargs=spec.get("nargs"),
        type_name=spec.get("type"),
        choices=choices,
        required=bool(spec.get("required", False)),
        dynamic_keys=tuple(sorted(dynamic_keys)),
    )


def _extract_extra_args(node, notes: list[str]) -> tuple[dict[str, ArgShape], bool]:
    """Extract ``extra_args`` shapes and whether the declared-flag set is dynamic.

    The second return is True when the full set of declared flags cannot be
    determined statically -- ``extra_args`` is not a literal sequence (a name
    reference, a call), or an entry within it is unreadable. The caller
    suppresses the SKILL.md reverse check for such a declaration, which would
    otherwise flag every documented argument as undeclared.
    """
    if node is None:
        return {}, False
    if not isinstance(node, ast.Tuple | ast.List):
        notes.append(
            "extra_args is not a literal sequence; the declared-flag set is unknown, so "
            "the SKILL.md reverse check is suppressed for this script"
        )
        return {}, True
    dynamic = False
    shapes = {}
    for entry in node.elts:
        if isinstance(entry, ast.Starred):
            notes.append(
                "extra_args splices a sequence; the declared-flag set is incomplete, so "
                "the SKILL.md reverse check is suppressed for this script"
            )
            dynamic = True
            continue
        shape = _shape_from_spec(entry, notes)
        if shape is None:
            dynamic = True
            continue
        shapes[shape.dest] = shape
    return shapes, dynamic


def _extract_pep723_deps(source: str, notes: list[str]) -> list[str] | None:
    blocks = [m for m in _PEP723_RE.finditer(source) if m.group("type") == "script"]
    if not blocks:
        return None
    if len(blocks) > 1:
        notes.append(
            f"{len(blocks)} PEP 723 script blocks found; a script must have at most one. "
            "Analyzing the first."
        )
    content_lines = []
    for line in blocks[0].group("content").splitlines():
        content_lines.append(line[2:] if line.startswith("# ") else line[1:])
    try:
        metadata = tomllib.loads("\n".join(content_lines))
    except tomllib.TOMLDecodeError as exc:
        notes.append(f"PEP 723 script block is not valid TOML ({exc})")
        return None
    deps = metadata.get("dependencies", [])
    if not isinstance(deps, list):
        notes.append("PEP 723 script block has a non-list dependencies value")
        return None
    return [d for d in deps if isinstance(d, str)]


def extract_script(script: Path, skill_dir: Path) -> SkillDeclaration:
    """Extract one script's declaration surface. Never imports the script."""
    decl = SkillDeclaration(skill_dir=skill_dir, script=script)
    try:
        source = script.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        decl.error = f"script is not valid UTF-8 ({exc})"
        return decl
    except OSError as exc:
        decl.error = f"could not read the script ({exc})"
        return decl
    try:
        tree = ast.parse(source, filename=str(script))
    except SyntaxError as exc:
        decl.error = f"script does not parse (line {exc.lineno}: {exc.msg})"
        return decl

    decl.pep723_deps = _extract_pep723_deps(source, decl.notes)

    for node in tree.body:
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets = [node.target]
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "_SKILL_VERSION":
                decl.version_constant = True

    calls = _find_decorator_calls(tree)
    if not calls:
        decl.error = "no @weather_skill decorator call found"
        return decl
    if len(calls) > 1:
        skipped = ", ".join(func_name for func_name, _ in calls[1:])
        decl.notes.append(
            f"{len(calls)} @weather_skill functions in the script; only the first "
            f"({calls[0][0]}) is analyzed; skipped: {skipped}"
        )
    call = calls[0][1]

    keywords = {kw.arg: kw.value for kw in call.keywords if kw.arg is not None}
    if any(kw.arg is None for kw in call.keywords):
        decl.notes.append("declaration spreads **kwargs; those keywords are not analyzed")

    name_node = call.args[0] if call.args else keywords.get("name")
    if name_node is not None:
        name = _literal(name_node)
        if isinstance(name, str):
            decl.name = name
        else:
            decl.notes.append("skill name is not a string literal; using the directory name")

    version_node = call.args[1] if len(call.args) > 1 else keywords.get("version")
    decl.version_passed = isinstance(version_node, ast.Name) and version_node.id == "_SKILL_VERSION"

    for keyword in _TOGGLE_KEYWORDS:
        if keyword in keywords:
            decl.toggles[keyword] = _literal(keywords[keyword])

    input_type = _literal(keywords["input_type"]) if "input_type" in keywords else None
    if input_type is DYNAMIC:
        decl.has_input = True
        decl.notes.append("input_type is not a literal; input arity unknown")
    elif input_type is not None:
        decl.has_input = True
        if isinstance(input_type, str | tuple):
            # A single type or a tuple of them declares one input; only a list
            # declares one entry per input.
            n_inputs = 1
        elif isinstance(input_type, list):
            n_inputs = len(input_type)
        else:
            n_inputs = None
            decl.notes.append(
                f"input_type is a {type(input_type).__name__} literal, not a string or "
                "sequence; input arity unknown"
            )
        if n_inputs is not None:
            variadic = (
                _literal(keywords["variadic_input"]) if "variadic_input" in keywords else False
            )
            decl.input_arity = "append" if (variadic is True or n_inputs > 1) else "single"

    if "input_names" in keywords:
        input_names = _literal(keywords["input_names"])
        if isinstance(input_names, list | tuple):
            decl.input_names = [str(n) for n in input_names]
        else:
            decl.notes.append("input_names is not a literal list; dedicated input flags unknown")

    output_type = _literal(keywords["output_type"]) if "output_type" in keywords else None
    if output_type is DYNAMIC:
        decl.has_output = True
        decl.notes.append("output_type is not a literal; treated as artifact-writing")
    else:
        decl.has_output = output_type is not None
        # A zarr envelope output, as opposed to PNG or no artifact. Unknown
        # (dynamic) output types stay False: a rule must not fire on a shape
        # extraction could not read.
        zarr_types = set(types.ALL)
        if isinstance(output_type, str):
            decl.writes_zarr = output_type in zarr_types
        elif isinstance(output_type, tuple | list | set):
            decl.writes_zarr = bool(output_type) and all(t in zarr_types for t in output_type)

    decl.sets_source = any(
        (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "set_source"
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )
    # A validate_type() call classifying without the --dims the run was given:
    # the shapes it compares are not the ones the decorator validated.
    decl.bare_validate_type = any(
        (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "validate_type"
        and len(node.args) < 3
        and not any(kw.arg == "dims" for kw in node.keywords)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )

    decl.extra_args, decl.extra_args_dynamic = _extract_extra_args(
        keywords.get("extra_args"), decl.notes
    )
    return decl


def extract_skill(skill_dir: Path) -> list[SkillDeclaration]:
    """Extract every decorated declaration in a skill directory.

    Reads each ``scripts/*.py``; scripts without a decorator call are helper
    scripts and are ignored when at least one decorated script exists. When no
    script yields a declaration, the per-script error results are returned so
    each carries its analysis failure.
    """
    scripts_dir = skill_dir / "scripts"
    # os.scandir raises on an unlistable directory; Path.glob would suppress
    # the PermissionError and misreport it as "no scripts/*.py found".
    try:
        if scripts_dir.is_dir():
            with os.scandir(scripts_dir) as entries:
                scripts = sorted(
                    Path(entry.path)
                    for entry in entries
                    if entry.name.endswith(".py") and not entry.name.startswith(".")
                )
        else:
            scripts = []
    except OSError as exc:
        return [
            SkillDeclaration(
                skill_dir=skill_dir,
                script=scripts_dir,
                error=f"could not list the scripts directory ({exc})",
            )
        ]
    if not scripts:
        return [
            SkillDeclaration(
                skill_dir=skill_dir,
                script=scripts_dir,
                error="no scripts/*.py found",
            )
        ]
    results = [extract_script(script, skill_dir) for script in scripts]
    ok = [r for r in results if r.error is None]
    return ok if ok else results
