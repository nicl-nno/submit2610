'NEREL ontology for the RAGU-style extraction tasks (GRPO-extraction line).\n\n\nThe entity/relation type inventory, descriptions and ``DOMAIN -> RANGE``\nconstraints are vendored from RAGU\'s ``ragu/triplet/ontology/builtin/nerel.yaml``\n(commit ``abd3f295``, MIT © AnonymousUpstream) — the single upstream source of truth\nfor the NEREL vocabulary RAGU itself uses ("29 entity types, 49 predicates").\nEmbedded here as Python literals (no runtime ``yaml`` dependency); aliases and\n``retype_when`` coercion rules of the upstream file are intentionally NOT\nvendored: the corpus generator never coerces (it only renders prompts), and the\nreward-side schema check validates membership + domain/range as-is. Upstream\nprovenance is preserved verbatim in the docstrings of the type tables.\ndomain/range\n\nThe module also carries the SECOND vocabulary — NEREL-BIO (the biomedical\nextension), vendored in ``_nerel_bio_tables.py`` from the upstream brat\n``annotation.conf`` — as a SEPARATE vocabulary (``nerel_bio``), never merged\nwith NEREL: the two schemes share most general type names but differ in\ntaxonomy (BIO ``DISO`` vs NEREL ``DISEASE``, ``STATE_OR_PROVINCE`` vs\n``STATE_OR_PROV``), and merging would break the exact prompt↔reward\ncontract. Rows carry their vocabulary id in\n``extra["extraction"]["ontology"]`` (default ``"nerel"``); the reward\nresolves the vocabulary through :func:`get_ontology`. NEREL-BIO relations\ncarry ALTERNATIVE signature lines (the multi-line ``Arg1/Arg2`` form of the\nbrat config): a triple satisfies the constraint when ANY line is satisfied.\n``extra["extraction"]["ontology"]``'

from __future__ import annotations

import random
from dataclasses import dataclass, field

from llm_training.extraction._nerel_bio_tables import (
    BIO_ENTITY_TYPES,
    BIO_RELATION_TYPES,
)

# --- Entity types (29)
# ``parent`` closes upward: a subtype satisfies any constraint that accepts
# its ancestors (CITY is-a LOCATION, PERCENT is-a NUMBER).


_ENTITY_TYPES: dict[str, dict] = {
    "PERSON": {
        "description": "a human being, including full names and nicknames",
    },
    "ORGANIZATION": {
        "description": "a company, agency, political party, or other organized group",
    },
    "LOCATION": {
        "description": "a geographic location, natural or physical, not otherwise classified",
    },
    "CITY": {
        "description": "a city or town",
        "parent": "LOCATION",
    },
    "COUNTRY": {
        "description": "a sovereign state or country",
        "parent": "LOCATION",
    },
    "STATE_OR_PROV": {
        "description": "a state, province, or other first-level administrative division",
        "parent": "LOCATION",
    },
    "DISTRICT": {
        "description": "a district, borough, or other sub-city administrative unit",
        "parent": "LOCATION",
    },
    "FACILITY": {
        "description": "a man-made structure or installation, e.g. airport, stadium",
        "parent": "LOCATION",
    },
    "FAMILY": {
        "description": "a family name or a family as a social unit",
    },
    "PROFESSION": {
        "description": "a job title or occupation",
    },
    "NATIONALITY": {
        "description": "a person's nationality or ethnic affiliation",
    },
    "RELIGION": {
        "description": "a religious belief system or denomination",
    },
    "IDEOLOGY": {
        "description": "a political or philosophical belief system",
    },
    "LANGUAGE": {
        "description": "a natural language name",
    },
    "AWARD": {
        "description": "a prize, honor, or recognition",
    },
    "EVENT": {
        "description": "a named event or a concrete news event occurrence",
    },
    "WORK_OF_ART": {
        "description": "a creative work such as a book, film, painting, etc.",
    },
    "PRODUCT": {
        "description": "a commercial product, including model names",
    },
    "LAW": {
        "description": "a legal act, regulation, or code",
    },
    "CRIME": {
        "description": "an illegal act or offense",
    },
    "PENALTY": {
        "description": "a legal punishment or sanction",
    },
    "DISEASE": {
        "description": "a medical condition or illness",
    },
    "DATE": {
        "description": "a calendar date, absolute or relative",
    },
    "TIME": {
        "description": "a time expression, e.g. \"10:30 AM\", \"19:58\"",
    },
    "AGE": {
        "description": "a person's age or the age of an entity",
    },
    "MONEY": {
        "description": "a monetary amount or currency",
    },
    "NUMBER": {
        "description": "a numeric value not covered by other numeric types",
    },
    "PERCENT": {
        "description": "a percentage value",
        "parent": "NUMBER",
    },
    "ORDINAL": {
        "description": "an ordinal number, e.g. \"first\", \"2nd\"",
        "parent": "NUMBER",
    },
}

# --- Relation types (49)
# ``domain``/``range`` empty tuple = "any entity type" (upstream omits them).
# ``domain``/``range``
_RELATION_TYPES: dict[str, dict] = {
    "ABBREVIATION": {
        "description": "connects an abbreviation to its full form",
        "domain": (),
        "range": (),
    },
    "ALTERNATIVE_NAME": {
        "description": "connects different names or aliases for the same entity",
        "domain": (),
        "range": (),
    },
    "AGE_DIED_AT": {
        "description": "the age at which a person died",
        "domain": ("PERSON",),
        "range": ("AGE",),
    },
    "AGE_IS": {
        "description": "assigns an age value to a person",
        "domain": ("PERSON",),
        "range": ("AGE",),
    },
    "AGENT": {
        "description": "links an event to its active agent or performer",
        "domain": ("EVENT",),
        "range": ("PERSON", "ORGANIZATION", "COUNTRY"),
    },
    "AWARDED_WITH": {
        "description": "links a person, organization, or work to an award it received",
        "domain": ("PERSON", "ORGANIZATION", "WORK_OF_ART"),
        "range": ("AWARD",),
    },
    "CAUSE_OF_DEATH": {
        "description": "the cause of a person's death",
        "domain": ("PERSON",),
        "range": ("DISEASE", "EVENT", "CRIME"),
    },
    "CONVICTED_OF": {
        "description": "connects a person to a crime they were convicted of",
        "domain": ("PERSON", "ORGANIZATION"),
        "range": ("CRIME",),
    },
    "DATE_DEFUNCT_IN": {
        "description": "the date when an organization or entity ceased to exist",
        "domain": ("ORGANIZATION", "FACILITY", "PRODUCT"),
        "range": ("DATE",),
    },
    "DATE_FOUNDED_IN": {
        "description": "the date when an organization or entity was founded",
        "domain": ("ORGANIZATION", "FACILITY", "LOCATION"),
        "range": ("DATE",),
    },
    "DATE_OF_BIRTH": {
        "description": "the birth date of a person",
        "domain": ("PERSON",),
        "range": ("DATE",),
    },
    "DATE_OF_CREATION": {
        "description": "the creation date of a product or work of art",
        "domain": ("PRODUCT", "WORK_OF_ART", "LAW"),
        "range": ("DATE",),
    },
    "DATE_OF_DEATH": {
        "description": "the death date of a person",
        "domain": ("PERSON",),
        "range": ("DATE",),
    },
    "START_TIME": {
        "description": "the start time of an event or state",
        "domain": ("EVENT",),
        "range": ("DATE", "TIME"),
    },
    "END_TIME": {
        "description": "the end time of an event or state",
        "domain": ("EVENT",),
        "range": ("DATE", "TIME"),
    },
    "POINT_IN_TIME": {
        "description": "links an event or state to a specific timestamp",
        "domain": ("EVENT",),
        "range": ("DATE", "TIME"),
    },
    "EXPENDITURE": {
        "description": "the amount of money spent by an entity",
        "domain": ("PERSON", "ORGANIZATION", "COUNTRY"),
        "range": ("MONEY",),
    },
    "INCOME": {
        "description": "the amount of money earned by an entity",
        "domain": ("PERSON", "ORGANIZATION", "COUNTRY"),
        "range": ("MONEY",),
    },
    "PRICE_OF": {
        "description": "the price of a product or service",
        "domain": ("PRODUCT", "WORK_OF_ART", "FACILITY"),
        "range": ("MONEY",),
    },
    "FOUNDED_BY": {
        "description": "links an organization to its founder",
        "domain": ("ORGANIZATION", "FACILITY", "LOCATION"),
        "range": ("PERSON", "ORGANIZATION"),
    },
    "HAS_CAUSE": {
        "description": "links an event or state to its cause",
        "domain": ("EVENT", "DISEASE", "CRIME"),
        "range": ("EVENT", "DISEASE", "CRIME", "PERSON", "ORGANIZATION"),
    },
    "HEADQUARTERED_IN": {
        "description": "links an organization to its headquarters location",
        "domain": ("ORGANIZATION",),
        "range": ("LOCATION",),
    },
    "IDEOLOGY_OF": {
        "description": "links an organization or person to their ideology",
        "domain": ("PERSON", "ORGANIZATION"),
        "range": ("IDEOLOGY",),
    },
    "INANIMATE_INVOLVED": {
        "description": "links an event to an inanimate object involved",
        "domain": ("EVENT",),
        "range": ("PRODUCT", "FACILITY", "WORK_OF_ART", "MONEY", "LAW"),
    },
    "KNOWS": {
        "description": "links a person to another person they know",
        "domain": ("PERSON",),
        "range": ("PERSON",),
    },
    "LOCATED_IN": {
        "description": "links a physical entity to its containing location",
        "domain": ("LOCATION", "ORGANIZATION", "EVENT"),
        "range": ("LOCATION",),
    },
    "MEDICAL_CONDITION": {
        "description": "links a person to a disease or medical condition",
        "domain": ("PERSON",),
        "range": ("DISEASE",),
    },
    "MEMBER_OF": {
        "description": "links a person, organization, or country to a group it belongs to",
        "domain": ("PERSON", "ORGANIZATION", "COUNTRY"),
        "range": ("ORGANIZATION", "FAMILY", "FACILITY"),
    },
    "ORGANIZES": {
        "description": "links an organizer to the event they organize",
        "domain": ("PERSON", "ORGANIZATION", "COUNTRY"),
        "range": ("EVENT",),
    },
    "ORIGINS_FROM": {
        "description": "links an entity to its country or place of origin",
        "domain": ("PERSON", "PRODUCT", "ORGANIZATION", "WORK_OF_ART"),
        "range": ("LOCATION", "NATIONALITY"),
    },
    "OWNER_OF": {
        "description": "links an owner to an asset or product they own",
        "domain": ("PERSON", "ORGANIZATION", "COUNTRY"),
        "range": ("PRODUCT", "ORGANIZATION", "FACILITY", "WORK_OF_ART"),
    },
    "PARENT_OF": {
        "description": "links a parent to their child",
        "domain": ("PERSON",),
        "range": ("PERSON",),
    },
    "SIBLING": {
        "description": "links two persons as siblings",
        "domain": ("PERSON",),
        "range": ("PERSON",),
    },
    "SPOUSE": {
        "description": "links two persons as married partners",
        "domain": ("PERSON",),
        "range": ("PERSON",),
    },
    "RELATIVE": {
        "description": "a generic family relation between two persons",
        "domain": ("PERSON",),
        "range": ("PERSON",),
    },
    "PART_OF": {
        "description": "links a component entity to the larger whole it belongs to",
        "domain": (),
        "range": (),
    },
    "PARTICIPANT_IN": {
        "description": "links a person or organization to an event or production they took part in",
        "domain": ("PERSON", "ORGANIZATION", "COUNTRY"),
        "range": ("EVENT", "WORK_OF_ART"),
    },
    "PENALIZED_AS": {
        "description": "links a convicted person to the penalty imposed",
        "domain": ("PERSON", "ORGANIZATION"),
        "range": ("PENALTY",),
    },
    "PLACE_OF_BIRTH": {
        "description": "the location where a person was born",
        "domain": ("PERSON",),
        "range": ("LOCATION",),
    },
    "PLACE_OF_DEATH": {
        "description": "the location where a person died",
        "domain": ("PERSON",),
        "range": ("LOCATION",),
    },
    "PLACE_RESIDES_IN": {
        "description": "the location where a person lives",
        "domain": ("PERSON",),
        "range": ("LOCATION",),
    },
    "PRODUCES": {
        "description": "links a creator to a product or work it produces",
        "domain": ("ORGANIZATION", "COUNTRY", "PERSON"),
        "range": ("PRODUCT", "WORK_OF_ART"),
    },
    "RELIGION_OF": {
        "description": "links a person to their religion",
        "domain": ("PERSON", "ORGANIZATION"),
        "range": ("RELIGION",),
    },
    "SCHOOLS_ATTENDED": {
        "description": "links a person to an educational institution they attended",
        "domain": ("PERSON",),
        "range": ("ORGANIZATION", "FACILITY"),
    },
    "SUBEVENT_OF": {
        "description": "links a sub-event to a larger encompassing event",
        "domain": ("EVENT",),
        "range": ("EVENT",),
    },
    "SUBORDINATE_OF": {
        "description": "links a subordinate person to their superior",
        "domain": ("PERSON", "ORGANIZATION"),
        "range": ("PERSON", "ORGANIZATION"),
    },
    "TAKES_PLACE_IN": {
        "description": "links an event to the location where it occurs",
        "domain": ("EVENT",),
        "range": ("LOCATION",),
    },
    "WORKPLACE": {
        "description": "links a person to the organization they work for",
        "domain": ("PERSON",),
        "range": ("ORGANIZATION", "FACILITY"),
    },
    "WORKS_AS": {
        "description": "links a person to their profession or job title",
        "domain": ("PERSON",),
        "range": ("PROFESSION",),
    },
}

assert len(_ENTITY_TYPES) == 29, f"Expected 29 NEREL entity types, got {len(_ENTITY_TYPES)}"
assert len(_RELATION_TYPES) == 49, f"Expected 49 NEREL relation types, got {len(_RELATION_TYPES)}"

#: Entity types whose values are literals (dates, numbers, money...) and for
#: which nominal-case/singular normalization is structurally inapplicable
# : (there is no dictionary head form of "17


LITERAL_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "DATE", "TIME", "AGE", "MONEY", "NUMBER", "PERCENT", "ORDINAL",
        "LANGUAGE",
    }
)

assert len(BIO_ENTITY_TYPES) == 45, (
    f"Expected 45 NEREL-BIO entity types, got {len(BIO_ENTITY_TYPES)}"
)
assert len(BIO_RELATION_TYPES) == 66, (
    f"Expected 66 NEREL-BIO relation types, got {len(BIO_RELATION_TYPES)}"
)

#: Literal entity types of the NEREL-BIO vocabulary: the same
#: numeric/temporal family as NEREL minus LANGUAGE (absent from the BIO
#: scheme). DISO stays nominal — disease names are ordinary noun phrases
#: that MUST pass nominative-singular normalization.

NEREL_BIO_LITERAL_ENTITY_TYPES: frozenset[str] = frozenset(
    {"DATE", "TIME", "AGE", "MONEY", "NUMBER", "PERCENT", "ORDINAL"}
)


@dataclass(frozen=True)
class ExtractionSchema:
    'A concrete prompt/reward schema: ontology types + signature map.\n\n\n    One instance fully determines both what the prompt promises (the rendered\n    entity/relation type lists) and what the reward\'s ``schema_conformance``\n    accepts — the two cannot drift apart (the RAGU ``Ontology`` invariant).\nAttributes\n        entity_types: allowed entity type names (a subset of the vocabulary).\n        relation_types: allowed relation type names (a subset of the vocabulary).\n        signatures: ``{relation -> lines}`` — the endpoint constraints, where a\n            single-line relation maps to ``{"domain": [...], "range": [...]}``\n            and a multi-line (alternative-constraint) relation to a LIST of\n            such dicts; empty lists = any type.\n        ontology: the vocabulary id (``"nerel"`` | ``"nerel_bio"``) — which\n            :class:`Ontology` the reward resolves for membership/allowedness\n            and the literal-type set.'

    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    signatures: dict[str, dict[str, list[str]] | list[dict[str, list[str]]]] = (
        field(default_factory=dict)
    )
    ontology: str = "nerel"

    def to_extra(self) -> dict:
        """Serialize into the reward-side ``extra["extraction"]`` payload."""
        return {
            "ontology": self.ontology,
            "entity_types": list(self.entity_types),
            "relation_types": list(self.relation_types),
            "signatures": {
                rel: (
                    [dict(line) for line in sig]
                    if isinstance(sig, list)
                    else {
                        "domain": list(sig.get("domain", ())),
                        "range": list(sig.get("range", ())),
                    }
                )
                for rel, sig in self.signatures.items()
            },
        }

    @classmethod
    def from_extra(cls, payload: dict | None) -> "ExtractionSchema | None":
        """Rebuild from ``extra["extraction"]``; ``None`` when malformed.

        Tolerates BOTH signature forms (the legacy single-dict form and the
        alternative-lines list) and the absence of the ``ontology`` key
        (pre-vocabulary rows default to ``"nerel"``).
        """
        if not isinstance(payload, dict):
            return None
        entity_types = payload.get("entity_types")
        relation_types = payload.get("relation_types")
        if not isinstance(entity_types, list) or not isinstance(relation_types, list):
            return None
        signatures: dict[str, dict[str, list[str]] | list[dict[str, list[str]]]] = {}
        raw_sigs = payload.get("signatures")
        if isinstance(raw_sigs, dict):
            for rel, sig in raw_sigs.items():
                if isinstance(sig, list):
                    # Alternative constraint lines
                    lines = [
                        {
                            "domain": [str(t) for t in line.get("domain", ()) or ()],
                            "range": [str(t) for t in line.get("range", ()) or ()],
                        }
                        for line in sig
                        if isinstance(line, dict)
                    ]
                    if lines:
                        signatures[str(rel)] = lines
                elif isinstance(sig, dict):
                    signatures[str(rel)] = {
                        "domain": [str(t) for t in sig.get("domain", ()) or ()],
                        "range": [str(t) for t in sig.get("range", ()) or ()],
                    }
        return cls(
            entity_types=tuple(str(t) for t in entity_types),
            relation_types=tuple(str(t) for t in relation_types),
            signatures=signatures,
            ontology=str(payload.get("ontology") or "nerel"),
        )


def signature_lines(
    sig: dict[str, list[str]] | list[dict[str, list[str]]],
) -> list[dict[str, list[str]]]:
    'Normalize a schema signature value into constraint lines.'
    if isinstance(sig, list):
        return sig
    return [sig]


class Ontology:
    'A vendored extraction vocabulary + subset/rendering helpers.\n\n\n    Parameterized so the same machinery serves both vocabularies (NEREL and\n    NEREL-BIO); the no-arg constructor is the NEREL vocabulary (backward\n    compatible). Relation entries may carry EITHER the single-constraint\n    ``domain``/``range`` pair OR ``signatures`` — a tuple of alternative\n    constraint lines (ANY of which a triple may satisfy, the multi-line\n    ``Arg1/Arg2`` form of the brat config).'

    def __init__(
        self,
        name: str = "nerel",
        entities: dict[str, dict] | None = None,
        relations: dict[str, dict] | None = None,
        literal_types: frozenset[str] | set[str] = LITERAL_ENTITY_TYPES,
        hub_types: tuple[str, ...] = ("PERSON", "ORGANIZATION", "LOCATION", "DATE"),
    ) -> None:
        self.name = name
        self._entities = dict(entities if entities is not None else _ENTITY_TYPES)
        self._relations = dict(
            relations if relations is not None else _RELATION_TYPES
        )
        self.literal_types = frozenset(literal_types)
        self.hub_types = tuple(hub_types)
        self._ancestors: dict[str, frozenset[str]] = {
            name_: frozenset(self._walk_ancestors(name_)) for name_ in self._entities
        }

    def _walk_ancestors(self, name: str) -> set[str]:
        seen: set[str] = {name}
        current = self._entities.get(name, {}).get("parent")
        while current and current not in seen:
            seen.add(current)
            current = self._entities.get(current, {}).get("parent")
        return seen

    def _relation_lines(self, entry: dict) -> tuple[tuple[str, ...], ...]:
        'The constraint lines of a relation entry: ``((domain, range), ...)``.\n\n        alternative lines; otherwise the single ``domain``/``range`` pair.'
        sigs = entry.get("signatures")
        if isinstance(sigs, (tuple, list)) and sigs:
            return tuple(
                (tuple(line.get("domain") or ()), tuple(line.get("range") or ()))
                for line in sigs
            )
        single = (
            tuple(entry.get("domain") or ()),
            tuple(entry.get("range") or ()),
        )
        return (single,)

    # ------------------------------------------------------------------ checks
    def knows_entity_type(self, name: str) -> bool:
        return name in self._entities

    def knows_relation_type(self, name: str) -> bool:
        return name in self._relations

    def ancestors(self, entity_type: str) -> frozenset[str]:
        """``{self, parent, grandparent, ...}`` (empty for unknown types)."""
        return self._ancestors.get(entity_type, frozenset())

    def allows(self, relation: str, source_type: str, target_type: str) -> bool:
        'True iff ``(source_type, relation, target_type)`` satisfies domain/range.\n\n        satisfy constraints that accept their ancestors (CITY is-a LOCATION);\n        an empty domain/range accepts any type; unknown types reject. A\n        multi-line (alternative-signature) relation is satisfied when ANY of\n        its lines is (the standard brat ``Arg1/Arg2`` semantics).'
        rel = self._relations.get(relation)
        if rel is None:
            return False
        if not self.knows_entity_type(source_type) or not self.knows_entity_type(target_type):
            return False
        for domain, rng in self._relation_lines(rel):
            if domain and not (self.ancestors(source_type) & set(domain)):
                continue
            if rng and not (self.ancestors(target_type) & set(rng)):
                continue
            return True
        return False

    # -------------------------------------------------------------- rendering
    def render_entity_types(self, names: list[str] | None = None) -> str:
        'Render the type list for the prompt: ``NAME (description), ...``.'
        selected = names if names is not None else list(self._entities)
        parts = []
        for name in selected:
            entry = self._entities.get(name)
            if entry is None:
                continue
            desc = entry.get("description") or ""
            parts.append(f"{name} ({desc})" if desc else name)
        return ", ".join(parts)

    def render_relation_types(
        self, names: list[str] | None = None, with_signatures: bool = False
    ) -> str:
        "Render the predicate list; with signatures as ``WORKPLACE [PERSON -> ORGANIZATION|FACILITY]``.\n\n        ``WORKPLACE [PERSON -> ORGANIZATION|FACILITY]`` (RAGU's\n        ``show_type_signatures`` rendering: without it the model cannot know\n        which endpoint types a predicate accepts)."
        selected = names if names is not None else list(self._relations)
        parts = []
        for name in selected:
            entry = self._relations.get(name)
            if entry is None:
                continue
            desc = entry.get("description") or ""
            if with_signatures:
                rendered_lines = [
                    f"[{self._render_side(domain)} -> {self._render_side(rng)}]"
                    for domain, rng in self._relation_lines(entry)
                ]
                sig = " ".join(rendered_lines)
                parts.append(f"{name} {sig} ({desc})" if desc else f"{name} {sig}")
            else:
                parts.append(f"{name} ({desc})" if desc else name)
        return ", ".join(parts)

    @staticmethod
    def _render_side(types: tuple[str, ...]) -> str:
        if not types:
            return "ANY"
        return "|".join(types)

    # ---------------------------------------------------------------- subsets
    def subset(
        self,
        rng: random.Random,
        n_entity_types: int = 10,
        n_relation_types: int | None = None,
        hub_probability: float = 0.7,
        with_signatures: bool = True,
    ) -> ExtractionSchema:
        'Sample a valid schema subset (deterministic under ``rng``).\n\n        ``rng``). Guarantees: ≥2 entity types, every kept relation is USABLE\n        within the sampled entity types (its domain and range each accept at\n        least one sampled type through the parent hierarchy), ≥1 relation when\n        any is usable. ``hub_probability``: chance for each of the hub types\n        (PERSON/ORGANIZATION/LOCATION/DATE — the endpoints of most predicates)\n        to be force-included, keeping small subsets relation-rich.'
        n_entity_types = max(2, min(int(n_entity_types), len(self._entities)))
        hubs = [t for t in self.hub_types if rng.random() < hub_probability]
        pool = [t for t in self._entities if t not in hubs]
        rng.shuffle(pool)
        chosen = list(dict.fromkeys(hubs + pool))[:n_entity_types]
        chosen_set = set(chosen)
        # A relation is usable when SOME constraint line's domain AND range
        # each accept ≥1 chosen type (subtype -> ancestor closure).
        
        
        usable: list[str] = []
        for rel, entry in self._relations.items():
            for domain, rng_types in self._relation_lines(entry):
                if domain and not any(
                    self.ancestors(t) & set(domain) for t in chosen
                ):
                    continue
                if rng_types and not any(
                    self.ancestors(t) & set(rng_types) for t in chosen
                ):
                    continue
                usable.append(rel)
                break
        rng.shuffle(usable)
        if n_relation_types is not None:
            n_relation_types = max(1, min(int(n_relation_types), len(usable)))
            kept = usable[:n_relation_types]
        else:
            kept = usable
        signatures: dict[str, dict[str, list[str]] | list[dict[str, list[str]]]] = {}
        if with_signatures:
            for rel in kept:
                signatures[rel] = self._signature_payload(self._relations[rel])
        return ExtractionSchema(
            entity_types=tuple(sorted(chosen_set)),
            relation_types=tuple(sorted(kept)),
            signatures=signatures,
            ontology=self.name,
        )

    def _signature_payload(
        self, entry: dict
    ) -> dict[str, list[str]] | list[dict[str, list[str]]]:
        "Serialize a relation entry's constraints for ``ExtractionSchema``.\n\ndict (legacy\n        compact form"
        lines = self._relation_lines(entry)
        if len(lines) == 1:
            domain, rng = lines[0]
            return {"domain": list(domain), "range": list(rng)}
        return [
            {"domain": list(domain), "range": list(rng)}
            for domain, rng in lines
        ]

    def full_schema(self, with_signatures: bool = True) -> ExtractionSchema:
        """The complete ontology as a schema (the "full vocabulary" variant)."""
        signatures: dict[str, dict[str, list[str]] | list[dict[str, list[str]]]] = {}
        if with_signatures:
            for rel, entry in self._relations.items():
                signatures[rel] = self._signature_payload(entry)
        return ExtractionSchema(
            entity_types=tuple(self._entities),
            relation_types=tuple(self._relations),
            signatures=signatures,
            ontology=self.name,
        )


# : Module-level singletons (immutable, cheap).
NEREL_ONTOLOGY = Ontology()

#: The NEREL-BIO vocabulary (biomedical extension, see
#: ``_nerel_bio_tables.py``). Hubs are the endpoint workhorses of biomedical
#: predicates (DISO/CHEM/ANATOMY) plus PERSON/DATE for the general lines.
NEREL_BIO_ONTOLOGY = Ontology(
    name="nerel_bio",
    entities=BIO_ENTITY_TYPES,
    relations=BIO_RELATION_TYPES,
    literal_types=NEREL_BIO_LITERAL_ENTITY_TYPES,
    hub_types=("DISO", "CHEM", "ANATOMY", "PERSON", "DATE"),
)

# : The vocabulary registry
ONTOLOGIES: dict[str, Ontology] = {
    NEREL_ONTOLOGY.name: NEREL_ONTOLOGY,
    NEREL_BIO_ONTOLOGY.name: NEREL_BIO_ONTOLOGY,
}


def get_ontology(name: str | None) -> Ontology:
    "Resolve a vocabulary by id; ``None``/absent → NEREL (the legacy default).\n\n    Raises ``ValueError`` for an unknown id — a typo in a row's ontology field\n    must fail loudly, never silently fall back (the prompt↔reward contract)."
    key = (name or "nerel").strip()
    try:
        return ONTOLOGIES[key]
    except KeyError:
        raise ValueError(
            f"Unknown extraction ontology: {name!r} (known: {sorted(ONTOLOGIES)})"
        ) from None
