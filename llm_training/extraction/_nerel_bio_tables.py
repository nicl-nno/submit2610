'Vendored NEREL-BIO ontology tables (GRPO-extraction line).\n\n\nThe entity/relation type inventory and the ``Arg1/Arg2`` constraints are\nvendored from NEREL-BIO\'s ``nerel-bio-v1.0/annotation.conf``\n(github.com/nerel-ds/NEREL-BIO, master as of 2026-10: 45 entity types,\n66 relation types, 163 constraint lines) — the brat annotation config is\nthe single upstream source of truth for the scheme. Embedded here as\nPython literals (no runtime ``yaml``/config dependency). The ``<OVERLAP>``\nservice record of the config (brat overlap settings) is intentionally NOT\nvendored. Entity descriptions are taken from Table 4 of the Bioinformatics\n2023 paper (Loukachevitch et al., "NEREL-BIO: a dataset of biomedical\nabstracts annotated with nested named entities",\ndoi:10.1093/bioinformatics/btad161) and follow the style of the NEREL table\nin ``ontology.py``; relation descriptions are written for this repo in the\nsame style (the config carries none). A relation carrying several\n``Arg1/Arg2`` lines in the config carries them as ALTERNATIVE signature\nlines — a triple satisfies the constraint when ANY line is satisfied (the\nstandard brat semantics). The vocabulary is used for research purposes;\nplease cite the Bioinformatics 2023 paper.\n``nerel-bio-v1.0/annotation.conf`` NEREL-BIO (45'

from __future__ import annotations


#: Entity types (45: 29 general-domain + 16 biomedical) /

#: ``parent`` mirrors the NEREL hierarchy for the shared general types
#: (CITY is-a LOCATION, PERCENT is-a NUMBER); biomedical types are flat.
BIO_ENTITY_TYPES: dict[str, dict] = {

    "AGE": {
        "description": "a person's age or the age of an entity",
    },
    "AWARD": {
        "description": "a prize, honor, or recognition",
    },
    "CITY": {
        "description": "a city or town",
        "parent": "LOCATION",
    },
    "COUNTRY": {
        "description": "a sovereign state or country",
        "parent": "LOCATION",
    },
    "CRIME": {
        "description": "an illegal act or offense",
    },
    "DATE": {
        "description": "a calendar date, absolute or relative",
    },
    "DISO": {
        "description": "any deviation from the normal state of an organism: a disease, symptom, dysfunction or organ abnormality, excluding injuries and poisoning",
    },
    "DISTRICT": {
        "description": "a district, borough, or other sub-city administrative unit",
        "parent": "LOCATION",
    },
    "EVENT": {
        "description": "a named event or a concrete news event occurrence, e.g. an epidemic or a natural disaster",
    },
    "FACILITY": {
        "description": "a man-made structure or installation, e.g. hospital building, stadium",
        "parent": "LOCATION",
    },
    "FAMILY": {
        "description": "a family name or a family as a social unit",
    },
    "IDEOLOGY": {
        "description": "a political or philosophical belief system",
    },
    "LANGUAGE": {
        "description": "a natural language name",
    },
    "LAW": {
        "description": "a legal act, regulation, or code",
    },
    "LOCATION": {
        "description": "a geographic location, natural or physical, not otherwise classified",
    },
    "MONEY": {
        "description": "a monetary amount or currency",
    },
    "NATIONALITY": {
        "description": "a person's nationality or ethnic affiliation",
    },
    "NUMBER": {
        "description": "a numeric value not covered by other numeric types",
    },
    "ORDINAL": {
        "description": "an ordinal number, e.g. \"first\", \"2nd\"",
        "parent": "NUMBER",
    },
    "ORGANIZATION": {
        "description": "a company, agency, political party, or other organized group",
    },
    "PENALTY": {
        "description": "a legal punishment or sanction",
    },
    "PERCENT": {
        "description": "a percentage value",
        "parent": "NUMBER",
    },
    "PERSON": {
        "description": "a human being, including full names and nicknames",
    },
    "PRODUCT": {
        "description": "a commercial product, including model names",
    },
    "PROFESSION": {
        "description": "a job title or occupation",
    },
    "RELIGION": {
        "description": "a religious belief system or denomination",
    },
    "STATE_OR_PROVINCE": {
        "description": "a state, province, or other first-level administrative division",
        "parent": "LOCATION",
    },
    "TIME": {
        "description": "a time expression, e.g. \"10:30 AM\", \"19:58\"",
    },
    "WORK_OF_ART": {
        "description": "a creative work such as a book, film, painting, etc.",
    },
    "ANATOMY": {
        "description": "an anatomical structure: organs, body parts, cells and cell components, body substances",
    },
    "LIVB": {
        "description": "any individual living (or previously living) being except humans",
    },
    "ACTIVITY": {
        "description": "human or animal behavior",
    },
    "CHEM": {
        "description": "a chemical, including legal and illegal drugs and biological molecules",
    },
    "DEVICE": {
        "description": "a manufactured object used for medical purposes",
    },
    "PHYS": {
        "description": "a biological function or process in an organism, including organism attributes, excluding mental processes",
    },
    "MEDPROC": {
        "description": "a procedure concerned with remedial treatment of diseases, including surgical procedures",
    },
    "SCIPROC": {
        "description": "a scientific study procedure, including mathematical methods, clinical studies, scales, classifiers",
    },
    "LABPROC": {
        "description": "a laboratory procedure or another diagnostic procedure, e.g. biochemical analysis, ultrasonography",
    },
    "FINDING": {
        "description": "a result of a scientific study or experiment described in the abstract",
    },
    "MENTALPROC": {
        "description": "a conceptual function or thinking process",
    },
    "GENE": {
        "description": "a nucleic acid sequence functioning as a unit of heredity, including DNA and chromosomes",
    },
    "FOOD": {
        "description": "a substance taken in by the body to provide nourishment, including fresh water and alcoholic beverages",
    },
    "INJURY_POISONING": {
        "description": "damage inflicted on the body as a direct or indirect result of external force, including poisoning",
    },
    "ADMINISTRATION_ROUTE": {
        "description": "a way of administering a drug or another chemical to an organism, including drug forms",
    },
    "HEALTH_CARE_ACTIVITY": {
        "description": "a health care administration and organization activity, e.g. hospitalization, medical evacuation",
    },
}

#: Relation types (66) with alternative signature lines /

#: ``domain``/``range`` = the single-constraint form (empty = any endpoint
#: type, the ``Arg1:<ENTITY>`` case); ``signatures`` = alternative
#: constraint lines, ANY of which a triple may satisfy (the multi-line
#: form of the brat config).
BIO_RELATION_TYPES: dict[str, dict] = {
    "ABBREVIATION": {
        "description": "connects an abbreviation to its full form",
        "signatures": (
            {
                "domain": ("AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
                "range": ("AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
            },
            {
                "domain": ("ANATOMY",),
                "range": ("ANATOMY",),
            },
            {
                "domain": ("LIVB",),
                "range": ("LIVB",),
            },
            {
                "domain": ("ACTIVITY",),
                "range": ("ACTIVITY",),
            },
            {
                "domain": ("CHEM",),
                "range": ("CHEM",),
            },
            {
                "domain": ("DEVICE",),
                "range": ("DEVICE",),
            },
            {
                "domain": ("PHYS",),
                "range": ("PHYS",),
            },
            {
                "domain": ("MEDPROC",),
                "range": ("MEDPROC",),
            },
            {
                "domain": ("SCIPROC",),
                "range": ("SCIPROC",),
            },
            {
                "domain": ("LABPROC",),
                "range": ("LABPROC",),
            },
            {
                "domain": ("FINDING",),
                "range": ("FINDING",),
            },
            {
                "domain": ("MENTALPROC",),
                "range": ("MENTALPROC",),
            },
            {
                "domain": ("GENE",),
                "range": ("GENE",),
            },
            {
                "domain": ("FOOD",),
                "range": ("FOOD",),
            },
            {
                "domain": ("INJURY_POISONING",),
                "range": ("INJURY_POISONING",),
            },
            {
                "domain": ("ADMINISTRATION_ROUTE",),
                "range": ("ADMINISTRATION_ROUTE",),
            },
            {
                "domain": ("HEALTH_CARE_ACTIVITY",),
                "range": ("HEALTH_CARE_ACTIVITY",),
            },
            {
                "domain": ("PERSON",),
                "range": ("PERSON",),
            },
            {
                "domain": ("NUMBER",),
                "range": ("NUMBER",),
            },
            {
                "domain": ("AGE",),
                "range": ("AGE",),
            },
            {
                "domain": ("DISO",),
                "range": ("DISO",),
            },
        ),
    },
    "ALTERNATIVE_NAME": {
        "description": "connects different names or aliases for the same entity",
        "signatures": (
            {
                "domain": ("AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
                "range": ("AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
            },
            {
                "domain": ("ANATOMY",),
                "range": ("ANATOMY",),
            },
            {
                "domain": ("LIVB",),
                "range": ("LIVB",),
            },
            {
                "domain": ("ACTIVITY",),
                "range": ("ACTIVITY",),
            },
            {
                "domain": ("CHEM",),
                "range": ("CHEM",),
            },
            {
                "domain": ("DEVICE",),
                "range": ("DEVICE",),
            },
            {
                "domain": ("PHYS",),
                "range": ("PHYS",),
            },
            {
                "domain": ("MEDPROC",),
                "range": ("MEDPROC",),
            },
            {
                "domain": ("SCIPROC",),
                "range": ("SCIPROC",),
            },
            {
                "domain": ("LABPROC",),
                "range": ("LABPROC",),
            },
            {
                "domain": ("FINDING",),
                "range": ("FINDING",),
            },
            {
                "domain": ("MENTALPROC",),
                "range": ("MENTALPROC",),
            },
            {
                "domain": ("GENE",),
                "range": ("GENE",),
            },
            {
                "domain": ("FOOD",),
                "range": ("FOOD",),
            },
            {
                "domain": ("INJURY_POISONING",),
                "range": ("INJURY_POISONING",),
            },
            {
                "domain": ("ADMINISTRATION_ROUTE",),
                "range": ("ADMINISTRATION_ROUTE",),
            },
            {
                "domain": ("HEALTH_CARE_ACTIVITY",),
                "range": ("HEALTH_CARE_ACTIVITY",),
            },
            {
                "domain": ("PERSON",),
                "range": ("PERSON",),
            },
            {
                "domain": ("NUMBER",),
                "range": ("NUMBER",),
            },
            {
                "domain": ("AGE",),
                "range": ("AGE",),
            },
            {
                "domain": ("DISO",),
                "range": ("DISO",),
            },
        ),
    },
    "KNOWS": {
        "description": "links a person to another person they know",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
            },
        ),
    },
    "AGE_IS": {
        "description": "assigns an age value to a person",
        "signatures": (
            {
                "domain": (),
                "range": ("AGE",),
            },
        ),
    },
    "AGE_DIED_AT": {
        "description": "the age at which a person died",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("AGE",),
            },
        ),
    },
    "AWARDED_WITH": {
        "description": "links a person, organization, or work to an award it received",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "ORGANIZATION", "WORK_OF_ART"),
                "range": ("AWARD",),
            },
        ),
    },
    "PLACE_OF_BIRTH": {
        "description": "the location where a person was born",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "LOCATION", "STATE_OR_PROVINCE"),
            },
        ),
    },
    "DATE_DEFUNCT_IN": {
        "description": "the date when an organization or entity ceased to exist",
        "signatures": (
            {
                "domain": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "EVENT", "LOCATION", "ORGANIZATION", "STATE_OR_PROVINCE", "PROFESSION"),
                "range": ("DATE",),
            },
        ),
    },
    "DATE_FOUNDED_IN": {
        "description": "the date when an organization or entity was founded",
        "signatures": (
            {
                "domain": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "EVENT", "LOCATION", "ORGANIZATION", "STATE_OR_PROVINCE", "WORK_OF_ART"),
                "range": ("DATE",),
            },
        ),
    },
    "DATE_OF_BIRTH": {
        "description": "the birth date of a person",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("DATE",),
            },
        ),
    },
    "DATE_OF_CREATION": {
        "description": "the creation date of a product, work of art, or award",
        "signatures": (
            {
                "domain": ("WORK_OF_ART", "LAW", "AWARD"),
                "range": ("DATE",),
            },
        ),
    },
    "DATE_OF_DEATH": {
        "description": "the death date of a person",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("DATE",),
            },
        ),
    },
    "POINT_IN_TIME": {
        "description": "links an event, state, or medical procedure to a specific timestamp",
        "signatures": (
            {
                "domain": ("EVENT", "PENALTY", "CRIME", "WORK_OF_ART", "AWARD", "PRODUCT", "HEALTH_CARE_ACTIVITY"),
                "range": ("DATE", "TIME"),
            },
            {
                "domain": ("MEDPROC", "SCIPROC", "LABPROC", "HEALTH_CARE_ACTIVITY"),
                "range": ("DATE", "TIME"),
            },
        ),
    },
    "PLACE_OF_DEATH": {
        "description": "the location where a person died",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "NATIONALITY"),
                "range": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "LOCATION", "STATE_OR_PROVINCE"),
            },
        ),
    },
    "FOUNDED_BY": {
        "description": "links an organization or place to its founder",
        "signatures": (
            {
                "domain": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "EVENT", "LOCATION", "ORGANIZATION", "STATE_OR_PROVINCE", "PROFESSION"),
                "range": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "EVENT", "LOCATION", "ORGANIZATION", "PERSON", "PROFESSION", "STATE_OR_PROVINCE"),
            },
        ),
    },
    "HEADQUARTERED_IN": {
        "description": "links an organization to its headquarters location",
        "signatures": (
            {
                "domain": ("ORGANIZATION",),
                "range": ("LOCATION", "CITY", "COUNTRY", "DISTRICT", "STATE_OR_PROVINCE", "FACILITY"),
            },
        ),
    },
    "IDEOLOGY_OF": {
        "description": "links an organization or person to their ideology",
        "signatures": (
            {
                "domain": ("PERSON", "ORGANIZATION", "PROFESSION", "COUNTRY", "FACILITY"),
                "range": ("IDEOLOGY",),
            },
        ),
    },
    "SPOUSE": {
        "description": "links two persons as married partners",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("PERSON", "PROFESSION"),
            },
        ),
    },
    "MEMBER_OF": {
        "description": "links a person, profession, or living being to a group it belongs to",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "LIVB"),
                "range": ("ORGANIZATION", "IDEOLOGY", "COUNTRY", "FAMILY", "PERSON", "LIVB"),
            },
        ),
    },
    "ORGANIZES": {
        "description": "links an organizer to the event they organize",
        "signatures": (
            {
                "domain": ("CITY", "COUNTRY", "DISTRICT", "ORGANIZATION", "PERSON", "PROFESSION", "STATE_OR_PROVINCE"),
                "range": ("EVENT",),
            },
        ),
    },
    "OWNER_OF": {
        "description": "links an owner to an asset or product they own",
        "signatures": (
            {
                "domain": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
                "range": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
            },
        ),
    },
    "PARENT_OF": {
        "description": "links a parent to their child",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("PERSON", "PROFESSION"),
            },
        ),
    },
    "PARTICIPANT_IN": {
        "description": "links a person, organization, or profession to an event, study, or procedure they took part in",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "ORGANIZATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE", "FACILITY", "AWARD", "WORK_OF_ART", "FAMILY", "NATIONALITY", "IDEOLOGY", "RELIGION"),
                "range": ("EVENT", "WORK_OF_ART", "CRIME", "PENALTY", "HEALTH_CARE_ACTIVITY", "ACTIVITY"),
            },
            {
                "domain": ("PROFESSION", "ORGANIZATION"),
                "range": ("LABPROC", "SCIPROC", "MEDPROC", "HEALTH_CARE_ACTIVITY"),
            },
        ),
    },
    "PLACE_RESIDES_IN": {
        "description": "the location where a person lives",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "LOCATION", "STATE_OR_PROVINCE"),
            },
        ),
    },
    "PRICE_OF": {
        "description": "the price of a product, service, or treatment",
        "signatures": (
            {
                "domain": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISO", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
                "range": ("MONEY",),
            },
        ),
    },
    "PRODUCES": {
        "description": "links a creator to a product or work it produces",
        "signatures": (
            {
                "domain": ("CITY", "COUNTRY", "DISTRICT", "ORGANIZATION", "PERSON", "PROFESSION", "STATE_OR_PROVINCE"),
                "range": ("AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
            },
        ),
    },
    "RELATIVE": {
        "description": "a generic family relation between two persons",
        "signatures": (
            {
                "domain": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
                "range": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
            },
        ),
    },
    "RELIGION_OF": {
        "description": "links a person to their religion",
        "signatures": (
            {
                "domain": ("PERSON", "ORGANIZATION", "PROFESSION", "COUNTRY", "FACILITY"),
                "range": ("RELIGION",),
            },
        ),
    },
    "SCHOOLS_ATTENDED": {
        "description": "links a person to an educational institution they attended",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("ORGANIZATION",),
            },
        ),
    },
    "SIBLING": {
        "description": "links two persons as siblings",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("PERSON", "PROFESSION"),
            },
        ),
    },
    "SUBEVENT_OF": {
        "description": "links a sub-event to a larger encompassing event",
        "signatures": (
            {
                "domain": ("EVENT",),
                "range": ("EVENT",),
            },
        ),
    },
    "SUBORDINATE_OF": {
        "description": "links a subordinate person to their superior",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("PERSON", "PROFESSION"),
            },
        ),
    },
    "TAKES_PLACE_IN": {
        "description": "links an event, procedure, or activity to the location where it occurs",
        "signatures": (
            {
                "domain": ("EVENT", "MEDPROC", "SCIPROC", "LABPROC", "HEALTH_CARE_ACTIVITY", "ACTIVITY"),
                "range": ("CITY", "COUNTRY", "DISTRICT", "ORGANIZATION", "PERSON", "STATE_OR_PROVINCE", "FACILITY", "LOCATION"),
            },
        ),
    },
    "WORKPLACE": {
        "description": "links a person to the organization they work for",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION"),
                "range": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "EVENT", "LOCATION", "IDEOLOGY", "ORGANIZATION", "STATE_OR_PROVINCE"),
            },
        ),
    },
    "WORKS_AS": {
        "description": "links a person to their profession or job title",
        "signatures": (
            {
                "domain": ("PERSON",),
                "range": ("PROFESSION",),
            },
        ),
    },
    "CONVICTED_OF": {
        "description": "connects a person to a crime they were convicted of",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "ORGANIZATION", "FAMILY", "NATIONALITY"),
                "range": ("CRIME",),
            },
        ),
    },
    "PENALIZED_AS": {
        "description": "links a convicted person to the penalty imposed",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "ORGANIZATION", "FAMILY", "NATIONALITY"),
                "range": ("PENALTY",),
            },
        ),
    },
    "START_TIME": {
        "description": "the start time of an event or state",
        "signatures": (
            {
                "domain": ("EVENT", "PENALTY", "CRIME", "WORK_OF_ART"),
                "range": ("DATE", "TIME"),
            },
        ),
    },
    "END_TIME": {
        "description": "the end time of an event or state",
        "signatures": (
            {
                "domain": ("EVENT", "PENALTY", "CRIME", "WORK_OF_ART"),
                "range": ("DATE", "TIME"),
            },
        ),
    },
    "EXPENDITURE": {
        "description": "the amount of money spent by an entity",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "CITY", "COUNTRY", "DISTRICT", "ORGANIZATION", "FAMILY", "STATE_OR_PROVINCE", "NATIONALITY"),
                "range": ("MONEY",),
            },
        ),
    },
    "AGENT": {
        "description": "links an event to its active agent or performer",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "ORGANIZATION", "CITY", "COUNTRY", "STATE_OR_PROVINCE", "FAMILY", "NATIONALITY", "IDEOLOGY", "RELIGION"),
                "range": ("EVENT",),
            },
        ),
    },
    "INANIMATE_INVOLVED": {
        "description": "links an event to an inanimate object involved",
        "signatures": (
            {
                "domain": ("PERSON", "PRODUCT", "FACILITY", "AWARD", "WORK_OF_ART", "LAW", "MONEY"),
                "range": ("EVENT", "WORK_OF_ART", "CRIME", "PENALTY"),
            },
        ),
    },
    "INCOME": {
        "description": "the amount of money earned by an entity",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "CITY", "COUNTRY", "DISTRICT", "ORGANIZATION", "FAMILY", "STATE_OR_PROVINCE", "NATIONALITY"),
                "range": ("MONEY",),
            },
        ),
    },
    "SUBCLASS_OF": {
        "description": "links a concept to its more general parent class within the same entity type",
        "signatures": (
            {
                "domain": ("ANATOMY",),
                "range": ("ANATOMY",),
            },
            {
                "domain": ("LIVB",),
                "range": ("LIVB",),
            },
            {
                "domain": ("ACTIVITY",),
                "range": ("ACTIVITY",),
            },
            {
                "domain": ("CHEM",),
                "range": ("CHEM",),
            },
            {
                "domain": ("DEVICE",),
                "range": ("DEVICE",),
            },
            {
                "domain": ("PHYS",),
                "range": ("PHYS",),
            },
            {
                "domain": ("MEDPROC",),
                "range": ("MEDPROC",),
            },
            {
                "domain": ("SCIPROC",),
                "range": ("SCIPROC",),
            },
            {
                "domain": ("LABPROC",),
                "range": ("LABPROC",),
            },
            {
                "domain": ("FINDING",),
                "range": ("FINDING",),
            },
            {
                "domain": ("MENTALPROC",),
                "range": ("MENTALPROC",),
            },
            {
                "domain": ("GENE",),
                "range": ("GENE",),
            },
            {
                "domain": ("FOOD",),
                "range": ("FOOD",),
            },
            {
                "domain": ("INJURY_POISONING",),
                "range": ("INJURY_POISONING",),
            },
            {
                "domain": ("ADMINISTRATION_ROUTE",),
                "range": ("ADMINISTRATION_ROUTE",),
            },
            {
                "domain": ("HEALTH_CARE_ACTIVITY",),
                "range": ("HEALTH_CARE_ACTIVITY",),
            },
            {
                "domain": ("PERSON",),
                "range": ("PERSON",),
            },
            {
                "domain": ("NUMBER",),
                "range": ("NUMBER",),
            },
            {
                "domain": ("AGE",),
                "range": ("AGE",),
            },
            {
                "domain": ("DISO",),
                "range": ("DISO",),
            },
            {
                "domain": ("ORGANIZATION",),
                "range": ("ORGANIZATION",),
            },
            {
                "domain": ("FACILITY",),
                "range": ("FACILITY",),
            },
            {
                "domain": ("LOCATION",),
                "range": ("LOCATION",),
            },
            {
                "domain": ("COUNTRY",),
                "range": ("COUNTRY",),
            },
            {
                "domain": ("DISTRICT",),
                "range": ("DISTRICT",),
            },
            {
                "domain": ("STATE_OR_PROVINCE",),
                "range": ("STATE_OR_PROVINCE",),
            },
            {
                "domain": ("CITY",),
                "range": ("CITY",),
            },
            {
                "domain": ("PROFESSION",),
                "range": ("PROFESSION",),
            },
            {
                "domain": ("EVENT",),
                "range": ("EVENT",),
            },
        ),
    },
    "PART_OF": {
        "description": "links a component entity to the larger whole it belongs to",
        "signatures": (
            {
                "domain": ("ORGANIZATION", "WORK_OF_ART", "LAW", "FACILITY", "PRODUCT", "AWARD"),
                "range": ("ORGANIZATION", "WORK_OF_ART", "LAW", "FACILITY", "PRODUCT", "AWARD"),
            },
            {
                "domain": ("ANATOMY",),
                "range": ("ANATOMY", "PERSON", "LIVB"),
            },
        ),
    },
    "LOCATED_IN": {
        "description": "links a physical entity, disease, or finding to its containing location or anatomical structure",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "CITY", "COUNTRY", "DISTRICT", "FACILITY", "LOCATION", "ORGANIZATION", "PRODUCT", "STATE_OR_PROVINCE", "WORK_OF_ART"),
                "range": ("CITY", "COUNTRY", "DISTRICT", "FACILITY", "LOCATION", "ORGANIZATION", "STATE_OR_PROVINCE"),
            },
            {
                "domain": ("DISO", "INJURY_POISONING"),
                "range": ("ANATOMY",),
            },
            {
                "domain": ("DISO",),
                "range": ("CITY", "COUNTRY", "STATE_OR_PROVINCE", "DISTRICT", "LOCATION"),
            },
        ),
    },
    "TREATED_USING": {
        "description": "links a disease, condition, or physiological process to the treatment used",
        "signatures": (
            {
                "domain": ("PERSON", "LIVB", "PROFESSION", "NATIONALITY", "DISO", "INJURY_POISONING"),
                "range": ("MEDPROC", "CHEM"),
            },
            {
                "domain": ("PHYS", "MENTALPROC"),
                "range": ("CHEM",),
            },
        ),
    },
    "ORIGINS_FROM": {
        "description": "links an entity to its origin: a source organism, anatomical structure, or study",
        "signatures": (
            {
                "domain": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
                "range": ("AGE", "AWARD", "CITY", "COUNTRY", "CRIME", "DATE", "DISTRICT", "EVENT", "FACILITY", "FAMILY", "IDEOLOGY", "LANGUAGE", "LAW", "LOCATION", "MONEY", "NATIONALITY", "NUMBER", "ORDINAL", "ORGANIZATION", "PENALTY", "PERCENT", "PERSON", "PRODUCT", "PROFESSION", "RELIGION", "STATE_OR_PROVINCE", "TIME", "WORK_OF_ART"),
            },
            {
                "domain": ("CHEM", "GENE"),
                "range": ("PERSON", "LIVB", "ANATOMY"),
            },
            {
                "domain": ("FINDING",),
                "range": ("SCIPROC", "LABPROC"),
            },
            {
                "domain": ("ANATOMY",),
                "range": ("DISO",),
            },
        ),
    },
    "TO_DETECT_OR_STUDY": {
        "description": "links a laboratory, scientific, or medical procedure to the object it detects or studies",
        "signatures": (
            {
                "domain": ("LABPROC", "SCIPROC", "MEDPROC", "HEALTH_CARE_ACTIVITY"),
                "range": ("DISO", "PHYS", "INJURY_POISONING", "FINDING"),
            },
            {
                "domain": ("LABPROC", "SCIPROC"),
                "range": ("GENE", "CHEM", "ANATOMY"),
            },
            {
                "domain": ("LABPROC", "SCIPROC"),
                "range": ("LIVB",),
            },
            {
                "domain": ("LABPROC", "SCIPROC"),
                "range": ("MEDPROC",),
            },
        ),
    },
    "AFFECTS": {
        "description": "links a disease, finding, physiological process, or chemical to what it affects",
        "signatures": (
            {
                "domain": ("DISO", "INJURY_POISONING"),
                "range": ("ANATOMY", "PERSON", "LIVB", "MENTALPROC"),
            },
            {
                "domain": ("FINDING",),
                "range": ("ANATOMY", "PHYS"),
            },
            {
                "domain": ("PHYS",),
                "range": ("PHYS", "ANATOMY", "FINDING"),
            },
            {
                "domain": ("DISO",),
                "range": ("FINDING",),
            },
            {
                "domain": ("CHEM",),
                "range": ("PHYS",),
            },
        ),
    },
    "HAS_CAUSE": {
        "description": "links a disease, finding, or injury to its cause",
        "signatures": (
            {
                "domain": ("EVENT", "CRIME", "PENALTY"),
                "range": ("EVENT", "CRIME", "PENALTY", "LAW"),
            },
            {
                "domain": ("DISO",),
                "range": ("CHEM", "PHYS", "ACTIVITY", "LIVB", "MEDPROC", "FINDING", "DISO", "INJURY_POISONING", "GENE"),
            },
            {
                "domain": ("FINDING",),
                "range": ("CHEM", "PHYS", "ACTIVITY", "MEDPROC", "FINDING", "DISO", "INJURY_POISONING"),
            },
            {
                "domain": ("INJURY_POISONING",),
                "range": ("CHEM", "MEDPROC", "FINDING", "DISO", "INJURY_POISONING", "GENE"),
            },
            {
                "domain": ("PHYS",),
                "range": ("CHEM", "PHYS", "ACTIVITY", "MEDPROC", "FINDING", "DISO", "INJURY_POISONING"),
            },
        ),
    },
    "APPLIED_TO": {
        "description": "links a procedure, chemical, or administration route to what it is applied to",
        "signatures": (
            {
                "domain": ("MEDPROC", "LABPROC", "CHEM", "SCIPROC"),
                "range": ("ANATOMY", "PHYS"),
            },
            {
                "domain": ("SCIPROC",),
                "range": ("MENTALPROC", "FINDING"),
            },
            {
                "domain": ("ADMINISTRATION_ROUTE",),
                "range": ("ANATOMY",),
            },
        ),
    },
    "USED_IN": {
        "description": "links a device, chemical, food, or procedure to the procedure it is used in",
        "signatures": (
            {
                "domain": ("LABPROC", "SCIPROC", "MEDPROC"),
                "range": ("LABPROC", "MEDPROC", "SCIPROC", "CHEM"),
            },
            {
                "domain": ("CHEM",),
                "range": ("LABPROC", "MEDPROC", "CHEM"),
            },
            {
                "domain": ("DEVICE",),
                "range": ("LIVB", "PERSON", "ANATOMY", "MEDPROC"),
            },
            {
                "domain": ("FOOD",),
                "range": ("FOOD",),
            },
            {
                "domain": ("CHEM",),
                "range": ("FOOD",),
            },
            {
                "domain": ("DEVICE", "PRODUCT", "FACILITY"),
                "range": ("LABPROC", "MEDPROC", "SCIPROC"),
            },
        ),
    },
    "ASSOCIATED_WITH": {
        "description": "links a physiological process, study, or disease to an associated entity",
        "signatures": (
            {
                "domain": ("PHYS",),
                "range": ("DISO", "INJURY_POISONING", "CHEM", "GENE", "PHYS"),
            },
            {
                "domain": ("SCIPROC",),
                "range": ("DISO",),
            },
            {
                "domain": ("DISO",),
                "range": ("FINDING",),
            },
        ),
    },
    "HAS_ADMINISTRATION_ROUTE": {
        "description": "links a chemical (drug) to its administration route",
        "signatures": (
            {
                "domain": ("CHEM",),
                "range": ("ADMINISTRATION_ROUTE",),
            },
        ),
    },
    "HAS_STRENGTH": {
        "description": "links a chemical (drug) to its strength or concentration",
        "signatures": (
            {
                "domain": ("CHEM",),
                "range": ("NUMBER",),
            },
        ),
    },
    "DURATION_OF": {
        "description": "the duration of a chemical course or a procedure",
        "signatures": (
            {
                "domain": ("CHEM", "MEDPROC", "SCIPROC", "LABPROC"),
                "range": ("NUMBER", "DATE", "TIME"),
            },
        ),
    },
    "VALUE_IS": {
        "description": "the measured value of a laboratory or medical procedure result",
        "signatures": (
            {
                "domain": ("LABPROC", "MEDPROC", "SCIPROC", "PHYS", "FINDING", "CHEM"),
                "range": ("PERCENT", "AGE", "DATE", "NUMBER"),
            },
        ),
    },
    "PHYSIOLOGY_OF": {
        "description": "links a physiological process to the organism, person, or anatomical structure it belongs to",
        "signatures": (
            {
                "domain": ("PHYS",),
                "range": ("ANATOMY", "PERSON", "FAMILY", "NATIONALITY", "LIVB", "PROFESSION"),
            },
        ),
    },
    "PROCEDURE_PERFORMED": {
        "description": "links a person or living being to a laboratory, scientific, or medical procedure performed on them",
        "signatures": (
            {
                "domain": ("PERSON", "FAMILY", "NATIONALITY", "LIVB", "PROFESSION"),
                "range": ("LABPROC", "SCIPROC", "MEDPROC"),
            },
        ),
    },
    "MENTAL_PROCESS_OF": {
        "description": "links a mental process to the person or living being it belongs to",
        "signatures": (
            {
                "domain": ("MENTALPROC",),
                "range": ("ANATOMY", "PERSON", "FAMILY", "NATIONALITY", "LIVB", "PROFESSION"),
            },
        ),
    },
    "MEDICAL_CONDITION": {
        "description": "links a person or living being to their disease or injury",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "LIVB"),
                "range": ("DISO", "INJURY_POISONING"),
            },
        ),
    },
    "DOSE_IS": {
        "description": "the dose of a chemical (drug)",
        "signatures": (
            {
                "domain": ("CHEM",),
                "range": ("NUMBER",),
            },
        ),
    },
    "FINDING_OF": {
        "description": "links a study finding to the entity it is a finding of",
        "signatures": (
            {
                "domain": ("FINDING",),
                "range": ("PERSON", "INJURY_POISONING", "PHYS", "LIVB"),
            },
        ),
    },
    "CAUSE_OF_DEATH": {
        "description": "the cause of a person's death",
        "signatures": (
            {
                "domain": ("PERSON", "PROFESSION", "NATIONALITY"),
                "range": ("DISO", "EVENT", "INJURY_POISONING"),
            },
        ),
    },
    "CONSUME": {
        "description": "links a person to a food or substance they consume",
        "signatures": (
            {
                "domain": ("PERSON",),
                "range": ("FOOD",),
            },
        ),
    },
}

