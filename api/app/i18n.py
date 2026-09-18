"""Server-side visitor-facing wording, in both languages.

Only text a visitor can read lives here. Three things deliberately do not:

* The collection manifest. Its domain labels are Chinese, but the manifest
  carries a content hash and a version, so the English labels are keyed by
  domain id here instead of being edited into frozen data.
* ``collections.py``'s Chinese term map. That is a query-expansion aid for
  Chinese search input, not copy; an English query never reaches it.
* The admin dashboard, which is blocked at nginx and is not a visitor surface.
"""

from __future__ import annotations

from typing import Literal

Language = Literal["zh", "en"]


def pick(language: str, zh: str, en: str) -> str:
    return en if language == "en" else zh


# Zhang Yanyuan (c. 815-877) wrote the first comprehensive history of Chinese
# painting. The name stays as-is in English with a romanisation, because it is
# a person, not a label to be translated.
CURATOR_NAME_EN = "Yanyuan"

# Domain ids are corpus routing labels. The Chinese wording lives in the
# collection manifest; these are the English equivalents, keyed by the same id.
DOMAIN_LABELS_EN: dict[str, tuple[str, str]] = {
    "global:nature-place": (
        "Nature and place",
        "Landscape, plants and animals, and how people make sense of where they are",
    ),
    "global:belief-ritual": (
        "Belief and ritual",
        "How different cultures move conviction into objects and spaces",
    ),
    "global:death-afterlife": (
        "Death and the afterlife",
        "How the dead are remembered and the hereafter imagined",
    ),
    "global:power-status": (
        "Power and status",
        "How rule, rank and standing are made visible",
    ),
    "global:body-identity": (
        "Body and identity",
        "Who portraits, dress and the depicted body speak for",
    ),
    "global:making-material": (
        "Material and making",
        "Materials, technique, and what the maker knew",
    ),
    "global:text-memory": (
        "Writing and memory",
        "How script, inscription and archive hold on to the past",
    ),
    "global:exchange-mobility": (
        "Exchange and movement",
        "How trade, migration and travel change objects",
    ),
    "global:daily-life": (
        "Daily life",
        "What eating, dwelling, work and play leave behind",
    ),
    "global:image-story": (
        "Image, looking and story",
        "How pictures organise looking and carry a narrative",
    ),
    "landscape-brush": (
        "Landscape and brushwork",
        "Ways of seeing in landscape, bird-and-flower and literati painting",
    ),
    "calligraphy-inscription": (
        "Writing on the picture",
        "How characters enter a painting, and what they do to it",
    ),
    "ritual-bronze": (
        "Ritual bronzes",
        "The shape, ornament and order of sacrificial vessels",
    ),
    "ceramics-glaze": (
        "Ceramics and glaze",
        "A thousand-year experiment with clay, fire and glaze",
    ),
    "buddhist-devotion": (
        "Buddhist images",
        "Posture, hand gesture and the manner of devotion",
    ),
    "funerary-afterlife": (
        "Burial and the afterlife",
        "What grave goods reveal about the imagined hereafter",
    ),
    "court-daily-life": (
        "Court and everyday",
        "Status as carried by dress, furniture and the scholar's desk",
    ),
    "trade-exchange": (
        "Trade and exchange",
        "Objects moving along the land and sea routes",
    ),
}

DEFAULT_DOMAIN_HINT_EN = "One comparable thread through the collection"


# -- interview wording -------------------------------------------------------

MOTIVATION_LABELS_EN: dict[str, str] = {
    "explorer": "There's something I want to figure out",
    "recharger": "Just browsing, to unwind",
    "facilitator": "I'm bringing someone with me",
    "professional": "I know the field and want depth",
}

MOTIVATION_HINTS_EN: dict[str, str] = {
    "explorer": "Fuller argument and comparative material",
    "recharger": "Shorter labels; image and pacing first",
    "facilitator": "Plainer language, more to talk about",
    "professional": "Keeps terminology, dating and material detail",
}

PRIOR_KNOWLEDGE_EN: dict[str, tuple[str, str]] = {
    "none": ("First time", "Starts from the most basic way of looking"),
    "some": ("I know a little", "Skips the basics, goes straight to the argument"),
    "familiar": ("Fairly familiar", "More detail, more exceptions, more dispute"),
}

DURATION_HINTS_EN: dict[str, str] = {
    "5": "5 objects · 2 narrative segments",
    "10": "8 objects · 3 narrative segments",
    "15": "12 objects · 4 narrative segments",
}

EXCLUSION_LABELS_EN: dict[str, str] = {
    "religion": "Religious content",
    "funerary": "Burial and death",
    "war": "War and violence",
    "none": "Nothing in particular",
}

# Deliberately about how one looks at objects rather than about any particular
# subject, so they stay true whatever the collection turns out to hold.
GENERIC_OPEN_QUESTIONS_EN = (
    "Who were these things originally made to be seen by?",
    "On the same theme, where do different places diverge?",
    "Which one looks least like its own period?",
)

# Culture packs are corpus facets; the Chinese labels live in the manifest.
CULTURE_PACK_LABELS_EN: dict[str, str] = {
    "europe": "Europe",
    "east_asia": "East Asia",
    "americas": "the Americas",
    "west_asia_north_africa": "West Asia and North Africa",
    "southeast_asia": "Southeast Asia",
    "south_asia": "South Asia",
    "africa": "Africa",
    "oceania": "Oceania",
}
