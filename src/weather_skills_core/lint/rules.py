"""The conformance rules: stable IDs, severities, and their evaluation.

Rule IDs are stable forever -- new rules take new numbers inside their band
and existing rules are never renumbered. Bands group the rules by what they
read: WSK0xx analysis failures, WSK1xx the declaration against the standard
surface, WSK2xx cross-skill comparisons over the corpus, WSK3xx the SKILL.md
manifest, WSK4xx script packaging (version constant, PEP 723).

Severities (worst first): ``error`` -- the declaration breaks an ecosystem
contract (the script cannot be analyzed, a shared flag behaves differently
across skills, versioning or packaging is broken); ``warning`` -- a
conformance divergence worth fixing (a shadowed standard flag, a duplicated
one-off, documentation drift); ``info`` -- advisory notes. All of it is
advisory: findings never gate a run by themselves.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from weather_skills_core.decorator import StandardParameter, standard_parameters
from weather_skills_core.lint.corpus import CorpusSkill
from weather_skills_core.lint.extract import ArgShape, SkillDeclaration, normalize_requirement_name

CORE_PACKAGE = "weather-skills-core"


@dataclass(frozen=True)
class Rule:
    id: str
    severity: str
    title: str
    #: Whether the rule is in the default set (the set a bare run evaluates,
    #: and the base that --select replaces / --extend-select adds to). This is
    #: orthogonal to ``severity``: severity drives display and --strict only,
    #: never default membership. WSK201 is advisory (a maintainer survey of
    #: shared flags) and default-off; every other rule is default-on.
    default_enabled: bool = True


RULES = {
    "WSK001": Rule("WSK001", "error", "skill script could not be analyzed"),
    "WSK101": Rule("WSK101", "warning", "extra argument shadows a standard parameter"),
    "WSK102": Rule("WSK102", "warning", "fetcher does not stamp a source"),
    "WSK103": Rule("WSK103", "warning", "shape assertion ignores the --dims override"),
    "WSK201": Rule(
        "WSK201", "warning", "one-off flag declared by multiple skills", default_enabled=False
    ),
    "WSK202": Rule("WSK202", "error", "shared flag name with divergent shape"),
    "WSK301": Rule("WSK301", "warning", "SKILL.md drift"),
    "WSK401": Rule("WSK401", "error", "_SKILL_VERSION missing or not passed to the decorator"),
    "WSK402": Rule("WSK402", "error", "PEP 723 block does not declare weather-skills-core"),
}

#: Rules that need a corpus beyond the skill itself; skipped (and excluded
#: from the score denominator) when none is available.
CROSS_SKILL_RULES = ("WSK201", "WSK202")
PER_SKILL_RULES = ("WSK101", "WSK102", "WSK103", "WSK301", "WSK401", "WSK402")


def default_rule_set() -> set[str]:
    """The rule codes evaluated when no ``--select`` is given."""
    return {rule.id for rule in RULES.values() if rule.default_enabled}


def expand_selector(selector: str) -> set[str]:
    """The rule codes a selector matches: a full code, or a category prefix.

    An exact rule code (``WSK201``) matches only itself; anything else is
    treated as a category prefix (``WSK2`` matches every ``WSK2xx``, ``WSK``
    matches all). Returns an empty set when the selector matches no known code,
    which the caller turns into a usage error.
    """
    if selector in RULES:
        return {selector}
    return {code for code in RULES if code.startswith(selector)}


@dataclass(frozen=True)
class Finding:
    rule: str
    severity: str
    skill: str
    flag: str | None
    file: str
    message: str


def _finding(rule_id: str, decl: SkillDeclaration, flag: str | None, message: str) -> Finding:
    return Finding(
        rule=rule_id,
        severity=RULES[rule_id].severity,
        skill=decl.display_name,
        flag=flag,
        file=str(decl.script),
        message=message,
    )


def _standard_lookup() -> dict[str, StandardParameter]:
    lookup: dict[str, StandardParameter] = {}
    for param in standard_parameters():
        lookup[param.dest] = param
        for flag in param.flags:
            lookup[flag] = param
    return lookup


def _shadow_remedy(param: StandardParameter) -> str:
    if param.kind == "io":
        return f"declare input_type/output_type so the decorator owns {'/'.join(param.flags)}"
    if param.accepts_help:
        return (
            f"declare the standard toggle {param.name}= instead (its dict form takes "
            "help/choices overrides)"
        )
    return f"declare the standard toggle {param.name}=True instead"


def _rule_shadow(decl: SkillDeclaration) -> list[Finding]:
    findings = []
    lookup = _standard_lookup()
    for shape in decl.extra_args.values():
        param = None
        for flag in shape.flags:
            if flag in lookup:
                param = lookup[flag]
                break
        if param is None and shape.dest in lookup:
            param = lookup[shape.dest]
        if param is None:
            continue
        if not decl.has_output and param.kind == "io":
            # A no-artifact skill (output_type is None) is forbidden from
            # declaring input_type and owns no decorator --output, so --input
            # and --output can only be reached through extra_args. Declaring
            # them there is the only option, not a shadow of the standard
            # surface. Other standard parameters are still flagged.
            continue
        findings.append(
            _finding(
                "WSK101",
                decl,
                shape.identity,
                f"extra_args {shape.dest!r} shadows the standard {param.name} parameter "
                f"({'/'.join(param.flags)}); {_shadow_remedy(param)}.",
            )
        )
    return findings


def _rule_source(decl: SkillDeclaration) -> list[Finding]:
    """WSK102: a fetcher-shaped skill that never names its data product.

    Fetcher-shaped means it writes a zarr envelope and declares no zarr input:
    it originates data rather than deriving it, so nothing upstream can supply
    ``weather_skills_source`` and only the skill knows what the product is. A
    transform is excluded because it inherits the attr from its input, and a
    PNG or no-artifact skill because it writes no envelope to carry it.

    As a declaration parameter this was visible in the surface; as a body call
    it can be forgotten silently, which is why the linter checks it.
    """
    if not (decl.writes_zarr and not decl.has_input) or decl.sets_source:
        return []
    return [
        _finding(
            "WSK102",
            decl,
            None,
            "fetcher writes a zarr envelope from no zarr input but never calls "
            "set_source(); name the data product (the source, not the skill) so "
            "downstream artifacts carry it.",
        )
    ]


def _rule_validate_type_dims(decl: SkillDeclaration) -> list[Finding]:
    """WSK103: a ``dims=True`` skill asserting a shape without the override.

    The decorator validates the input and the output union against the axis
    names ``--dims`` gave; a ``validate_type()`` call that omits them classifies
    by CF attrs and heuristics instead, so on the very inputs ``--dims`` exists
    for the assertion compares shapes nobody else read -- passing when the
    output drifted, or failing a correct transform.
    """
    if not (decl.toggle_enabled("dims") and decl.bare_validate_type):
        return []
    return [
        _finding(
            "WSK103",
            decl,
            "--dims",
            "skill declares dims=True but calls validate_type() without dims; pass "
            'args["dims"] so the assertion classifies by the axis names the run '
            "validated against.",
        )
    ]


def _rule_version(decl: SkillDeclaration) -> list[Finding]:
    if not decl.version_constant:
        return [
            _finding(
                "WSK401",
                decl,
                None,
                "no module-level _SKILL_VERSION constant; define it at the top of the "
                "script and pass it as the decorator's version argument.",
            )
        ]
    if not decl.version_passed:
        return [
            _finding(
                "WSK401",
                decl,
                None,
                "_SKILL_VERSION is defined but not passed as the decorator's version "
                "argument; pass the constant so the epilog and provenance carry it.",
            )
        ]
    return []


def _rule_core_dep(decl: SkillDeclaration) -> list[Finding]:
    if decl.pep723_deps is None:
        return [
            _finding(
                "WSK402",
                decl,
                None,
                "no PEP 723 inline script block found; declare the script's dependencies "
                f"there, including {CORE_PACKAGE}.",
            )
        ]
    names = {normalize_requirement_name(dep) for dep in decl.pep723_deps}
    if CORE_PACKAGE not in names:
        return [
            _finding(
                "WSK402",
                decl,
                None,
                f"the PEP 723 dependencies do not declare {CORE_PACKAGE}; add it so "
                "uv run --script can resolve the decorator.",
            )
        ]
    return []


def _declared_flags(decl: SkillDeclaration) -> tuple[dict[str, str], set[str]]:
    """The declared CLI flags: {primary flag: origin} plus every accepted spelling."""
    primary: dict[str, str] = {}
    all_spellings: set[str] = set()
    for shape in decl.extra_args.values():
        if shape.positional:
            continue
        primary.setdefault(shape.primary_flag, f"extra_args {shape.dest!r}")
        all_spellings.update(shape.flags)
    for param in standard_parameters():
        if param.kind == "toggle" and decl.toggle_enabled(param.name):
            primary.setdefault(param.flags[0], f"standard toggle {param.name}")
            all_spellings.update(param.flags)
    if decl.has_input:
        if decl.input_names:
            for name in decl.input_names:
                primary.setdefault(f"--{name}", "input_names")
                all_spellings.add(f"--{name}")
        else:
            primary.setdefault("--input", "input_type")
            all_spellings.update(("--input", "-i"))
    if decl.has_output:
        primary.setdefault("--output", "output_type")
        all_spellings.update(("--output", "-o"))
    return primary, all_spellings


def _flag_mentioned(flag: str, text: str) -> bool:
    return re.search(re.escape(flag) + r"(?![\w-])", text) is not None


def _documented_argument_flags(text: str) -> set[str]:
    """The long flags SKILL.md's Arguments section documents.

    Pragmatic parse: inside any heading containing "argument", the first
    backticked long flag of each list item is the flag being documented
    (prose mentions of other flags later in the item are not).
    """
    flags: set[str] = set()
    in_arguments = False
    section_level = 0
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s*(.+)$", line)
        if heading:
            if in_arguments and len(heading.group(1)) <= section_level:
                in_arguments = False
            if re.search(r"argument", heading.group(2), re.IGNORECASE):
                in_arguments = True
                section_level = len(heading.group(1))
            continue
        if not in_arguments:
            continue
        bullet = re.match(r"^\s*[-*+]\s+(.*)$", line)
        if bullet:
            m = re.search(r"`(--[A-Za-z0-9][A-Za-z0-9-]*)`", bullet.group(1))
            if m:
                flags.add(m.group(1))
    return flags


def _rule_skill_md(decl: SkillDeclaration, siblings: list[SkillDeclaration]) -> list[Finding]:
    """WSK301 for one script against its skill directory's SKILL.md.

    ``siblings`` is every analyzable declaration in the same skill directory
    (including ``decl``). The forward check (a declared flag absent from
    SKILL.md) is per script. The reverse check (SKILL.md documents a flag no
    script declares) unions the declarations of all ``siblings`` so a
    multi-script skill's arguments do not read as undeclared just because they
    live in another of its scripts; it is emitted once, on the lexically first
    sibling, so the finding is not duplicated across scripts. It is suppressed
    entirely when any sibling's ``extra_args`` is dynamic (its declared-flag
    set is unknown), since the reverse check would then flag documented
    arguments as undeclared; the extraction note surfaces that suppression.
    """
    skill_md = decl.skill_dir / "SKILL.md"
    if not skill_md.is_file():
        return [
            _finding(
                "WSK301",
                decl,
                None,
                "SKILL.md is missing alongside scripts/; add the skill manifest.",
            )
        ]
    try:
        text = skill_md.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [
            _finding(
                "WSK301",
                decl,
                None,
                f"SKILL.md could not be read ({exc}); fix its encoding or permissions.",
            )
        ]
    findings = []
    primary, _ = _declared_flags(decl)
    for flag, origin in sorted(primary.items()):
        if not _flag_mentioned(flag, text):
            findings.append(
                _finding(
                    "WSK301",
                    decl,
                    flag,
                    f"declared flag {flag} (from {origin}) is not mentioned in SKILL.md; "
                    "document it in the Arguments section.",
                )
            )
    reverse_holder = decl.key == min(sibling.key for sibling in siblings)
    reverse_reliable = not any(sibling.extra_args_dynamic for sibling in siblings)
    if reverse_holder and reverse_reliable:
        union_spellings: set[str] = set()
        for sibling in siblings:
            _, spellings = _declared_flags(sibling)
            union_spellings |= spellings
        for flag in sorted(_documented_argument_flags(text) - union_spellings):
            findings.append(
                _finding(
                    "WSK301",
                    decl,
                    flag,
                    f"SKILL.md documents {flag} but the skill does not declare it in any "
                    "script; remove the entry or declare the argument.",
                )
            )
    return findings


def _shape_divergences(a: ArgShape, b: ArgShape) -> tuple[list[str], bool]:
    """(differences, partial): shape components differing between two declarations.

    Components recorded as dynamic on either side are skipped from the
    comparison; ``partial`` reports that such a skip happened.
    """
    if a.dynamic or b.dynamic:
        return [], True
    dynamic = set(a.dynamic_keys) | set(b.dynamic_keys)
    partial = bool(dynamic)
    diffs = []
    if not ({"action", "repeat"} & dynamic) and a.arity != b.arity:
        diffs.append(f"arity {a.arity} vs {b.arity}")
    if "nargs" not in dynamic and a.nargs != b.nargs:
        diffs.append(f"nargs {a.nargs!r} vs {b.nargs!r}")
    if "type" not in dynamic and a.type_name != b.type_name:
        diffs.append(f"type {a.type_name or 'str'} vs {b.type_name or 'str'}")
    if "choices" not in dynamic and a.choices != b.choices:
        diffs.append(
            f"choices {list(a.choices) if a.choices else None} vs "
            f"{list(b.choices) if b.choices else None}"
        )
    return diffs, partial


def _rule_cross_skill(corpus: list[CorpusSkill]) -> list[Finding]:
    lookup = _standard_lookup()
    by_identity: dict[str, list[tuple[CorpusSkill, ArgShape]]] = {}
    for cs in corpus:
        for shape in cs.decl.extra_args.values():
            if shape.dest in lookup or any(flag in lookup for flag in shape.flags):
                continue  # shadows of the standard surface are WSK101 territory
            by_identity.setdefault(shape.identity, []).append((cs, shape))

    findings = []
    for identity, entries in sorted(by_identity.items()):
        # Holders are skill directories, not individual scripts: two scripts in
        # one skill directory sharing a flag are not a cross-skill collision, so
        # a skill is never fired against its own scripts.
        holder_dirs = {cs.decl.skill_dir.resolve() for cs, _ in entries}
        if len(holder_dirs) < 2:
            continue
        for cs, shape in entries:
            if not cs.is_target:
                continue
            this_dir = cs.decl.skill_dir.resolve()
            others = [(o, s) for o, s in entries if o.decl.skill_dir.resolve() != this_dir]
            if not others:
                continue
            other_names = ", ".join(f"{o.decl.display_name} ({o.source})" for o, _ in others)
            findings.append(
                _finding(
                    "WSK201",
                    cs.decl,
                    identity,
                    f"one-off flag {identity} is also declared by {other_names}; rename "
                    "it, or propose promoting it to a weather-skills-core standard "
                    "parameter.",
                )
            )
            divergent = []
            any_partial = False
            for other, other_shape in others:
                diffs, partial = _shape_divergences(shape, other_shape)
                any_partial = any_partial or partial
                if diffs:
                    divergent.append(
                        f"{other.decl.display_name} ({other.source}): {'; '.join(diffs)}"
                    )
            if divergent:
                suffix = " (dynamic values excluded from the comparison)" if any_partial else ""
                findings.append(
                    _finding(
                        "WSK202",
                        cs.decl,
                        identity,
                        f"{identity} diverges in shape from {' | '.join(divergent)}; "
                        f"align the declarations or rename the flags{suffix}.",
                    )
                )
    return findings


def lint_corpus(
    corpus: list[CorpusSkill], corpus_available: bool, active_rules: set[str]
) -> list[Finding]:
    """Evaluate the active rules over the corpus; findings only for target skills.

    ``active_rules`` is the resolved rule set (see
    :func:`weather_skills_core.lint.run.resolve_rule_set`). Only findings whose
    rule is active are returned, and the cross-skill pass is skipped entirely
    when no cross-skill rule is active.
    """
    findings: list[Finding] = []
    # Analyzable target declarations grouped by skill directory: the SKILL.md
    # reverse check unions all of a skill's scripts (WSK301 must not flag one
    # script's argument as undeclared when a sibling script declares it).
    siblings: dict[Path, list[SkillDeclaration]] = {}
    for cs in corpus:
        if cs.is_target and cs.decl.error is None:
            siblings.setdefault(cs.decl.skill_dir.resolve(), []).append(cs.decl)
    for cs in corpus:
        if not cs.is_target:
            continue
        decl = cs.decl
        if decl.error is not None:
            findings.append(_finding("WSK001", decl, None, f"{decl.error}."))
            continue
        findings += _rule_shadow(decl)
        findings += _rule_source(decl)
        findings += _rule_validate_type_dims(decl)
        findings += _rule_skill_md(decl, siblings[decl.skill_dir.resolve()])
        findings += _rule_version(decl)
        findings += _rule_core_dep(decl)
    if corpus_available and set(CROSS_SKILL_RULES) & active_rules:
        findings += _rule_cross_skill([cs for cs in corpus if cs.decl.error is None])
    return [f for f in findings if f.rule in active_rules]
