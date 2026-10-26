#!/usr/bin/env python3
"""
Build the standalone museum-grade HTML reading edition
('docs/thompson_1913_aristotle_as_a_biologist.html') from
'docs/THOMPSON_1913_ARISTOTLE_AS_A_BIOLOGIST.md', complete with:
  - Full SEO metadata (canonical URL, description, keywords, author, publisher)
  - Open Graph (og:*) and Twitter Card (twitter:*) social preview tags
  - Highwire Press / Google Scholar citation_* meta tags (for academic indexing)
  - Rich Schema.org JSON-LD (@graph of ScholarlyArticle + BreadcrumbList + ImageObject + FAQPage)
  - Interactive Table of Contents with anchor-linked headings and semantic HTML5 landmarks
  - Toggleable Commentary Mode:
      (1) Research & Morphogenesis Commentary (Editorial Commentaries 1–13 + Morpholib Synthesis)
      (2) Friendly Student Guide (13-Year-Old Level: Reader Notes 1–25 + Figures 1–6 from the .docx)
      (3) Show Both Side-by-Side
"""

from __future__ import annotations

from pathlib import Path
import html
import json
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
MD_PATH = REPO_ROOT / "docs" / "THOMPSON_1913_ARISTOTLE_AS_A_BIOLOGIST.md"
HTML_PATH = REPO_ROOT / "docs" / "thompson_1913_aristotle_as_a_biologist.html"

CANONICAL_URL = "https://fractastical.github.io/morpholib/thompson_1913_aristotle_as_a_biologist.html"
OG_IMAGE_URL = "https://fractastical.github.io/morpholib/media/thompson_1913_fig2_lesbos_pyrrha_lagoon.png"


def slugify(text: str) -> str:
    clean = re.sub(r"<[^>]+>", "", text)
    clean = re.sub(r"[^a-zA-Z0-9]+", "-", clean.lower()).strip("-")
    return clean or "section"


def format_inline(text: str) -> str:
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank" rel="noopener">\1</a>',
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r'<span class="editorial-font">\1</span>', text)
    return text


def build_json_ld() -> str:
    schema = {
        "@context": "https://schema.org",
        "@graph": [
            {
                "@type": "BreadcrumbList",
                "@id": f"{CANONICAL_URL}#breadcrumb",
                "itemListElement": [
                    {
                        "@type": "ListItem",
                        "position": 1,
                        "name": "Project Delphi (morpholib)",
                        "item": "https://fractastical.github.io/morpholib/",
                    },
                    {
                        "@type": "ListItem",
                        "position": 2,
                        "name": "D’Arcy Wentworth Thompson (1913) — On Aristotle as a Biologist (Annotated Museum Edition)",
                        "item": CANONICAL_URL,
                    },
                ],
            },
            {
                "@type": "ScholarlyArticle",
                "@id": f"{CANONICAL_URL}#article",
                "headline": "On Aristotle as a Biologist, With a Prooemion on Herbert Spencer (1913) — Annotated & Museum-Illustrated Edition",
                "alternativeHeadline": "D’Arcy Wentworth Thompson’s 1913 Oxford Herbert Spencer Lecture with Critical Morphogenesis Commentary, 17th–19th Century Copperplates, and Student Reader Notes",
                "description": (
                    "Full annotated and museum-illustrated edition of D’Arcy Wentworth Thompson’s 1913 Oxford lecture "
                    "'On Aristotle as a Biologist, With a Prooemion on Herbert Spencer', featuring 17th–19th century "
                    "copperplates (Malpighi 1673, Johannes Müller 1842 Mustelus laevis yolk-sac placenta, Vérany 1851 "
                    "Argonauta, 1848/1860 Admiralty Chart of Lesbos & Pyrrha Lagoon, Turpin 1837 Urpflanze), 13 critical "
                    "morphogenesis commentaries, 25 student reader notes, and connections to computational morphometrics "
                    "and bioelectric patterning."
                ),
                "author": [
                    {
                        "@type": "Person",
                        "name": "D’Arcy Wentworth Thompson",
                        "birthDate": "1860-05-02",
                        "deathDate": "1948-06-21",
                        "sameAs": "https://en.wikipedia.org/wiki/D%27Arcy_Wentworth_Thompson",
                    },
                    {
                        "@type": "Person",
                        "name": "Joel Dietz",
                        "affiliation": {
                            "@type": "Organization",
                            "name": "Project Delphi",
                            "url": "https://github.com/fractastical/morpholib",
                        },
                    },
                ],
                "about": [
                    {"@type": "Thing", "name": "Aristotle's Biology", "sameAs": "https://en.wikipedia.org/wiki/Aristotle%27s_biology"},
                    {"@type": "Thing", "name": "On Growth and Form", "sameAs": "https://en.wikipedia.org/wiki/On_Growth_and_Form"},
                    {"@type": "Thing", "name": "Herbert Spencer", "sameAs": "https://en.wikipedia.org/wiki/Herbert_Spencer"},
                    {"@type": "Thing", "name": "Morphogenesis", "sameAs": "https://en.wikipedia.org/wiki/Morphogenesis"},
                    {"@type": "Place", "name": "Gulf of Kalloni (Lagoon of Pyrrha), Lesbos", "sameAs": "https://en.wikipedia.org/wiki/Gulf_of_Kalloni"},
                ],
                "datePublished": "1913-02-14",
                "dateModified": "2026-10-25",
                "inLanguage": "en",
                "isAccessibleForFree": True,
                "license": "https://creativecommons.org/publicdomain/mark/1.0/",
                "mainEntityOfPage": CANONICAL_URL,
                "publisher": {
                    "@type": "Organization",
                    "name": "Project Delphi",
                    "url": "https://fractastical.github.io/morpholib/",
                },
                "image": [
                    "https://fractastical.github.io/morpholib/media/thompson_1913_fig1_spencer_vs_aristotle.png",
                    "https://fractastical.github.io/morpholib/media/thompson_1913_fig2_lesbos_pyrrha_lagoon.png",
                    "https://fractastical.github.io/morpholib/media/thompson_1913_fig3_cephalopod_embryology_soul.png",
                    "https://fractastical.github.io/morpholib/media/thompson_1913_fig4_morpholib_four_connections.png",
                    "https://fractastical.github.io/morpholib/media/thompson_1913_fig5_nested_soul_museum.png",
                    "https://fractastical.github.io/morpholib/media/thompson_1913_fig6_first_actuality_museum.png",
                ],
            },
            {
                "@type": "FAQPage",
                "@id": f"{CANONICAL_URL}#faq",
                "mainEntity": [
                    {
                        "@type": "Question",
                        "name": "Where did Aristotle conduct his main marine biology observations according to D’Arcy Wentworth Thompson?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": (
                                "In his 1913 Oxford lecture 'On Aristotle as a Biologist', D’Arcy Wentworth Thompson shows from "
                                "internal geographical evidence in the Historia Animalium that Aristotle conducted a pivotal cluster "
                                "of his marine zoological studies at the sheltered Lagoon of Pyrrha (modern Gulf of Kalloni) on the "
                                "island of Lesbos (Mitylene) between 345 and 342 BC, prior to teaching in the Athens Lyceum."
                            ),
                        },
                    },
                    {
                        "@type": "Question",
                        "name": "What zoological discoveries of Aristotle were later confirmed by modern anatomists like Cuvier, Steno, and Johannes Müller?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": (
                                "Thompson highlights three famous examples: (1) the reproductive hectocotylus arm of male cephalopods "
                                "and their cephalic yolk-sac; (2) the yolk-sac placenta (placenta vitellina) of the smooth dogfish "
                                "Mustelus laevis (galeos leios), rediscovered by Steno in 1673 and Johannes Müller in 1842; and "
                                "(3) the day-by-day embryology of the chick in the egg, beginning with the beating heart (punctum saliens) "
                                "on the third day."
                            ),
                        },
                    },
                    {
                        "@type": "Question",
                        "name": "How does D’Arcy Thompson contrast Herbert Spencer and Aristotle as thinkers about living form?",
                        "acceptedAnswer": {
                            "@type": "Answer",
                            "text": (
                                "Thompson argues that Herbert Spencer approached biology with the training and habits of a railway and "
                                "bridge engineer—focusing on mechanical causation, stress, strain, and scaling limits—whereas Aristotle "
                                "was first and foremost a field naturalist and anatomist whose study of living organization, development, "
                                "and biological capacities shaped his psychology (De Anima) and philosophy."
                            ),
                        },
                    },
                ],
            },
        ],
    }
    return json.dumps(schema, ensure_ascii=False, indent=2)


# Reader Notes 1–25 from On_Aristotle_as_a_Biologist_Readable (1).docx
# mapped to exact Editorial Commentary numbers or section hooks so readers can view
# either the Research Commentary, the 13-Year-Old Student Reader Notes, or Both!
READER_NOTES_BY_COMMENTARY = {
    "INTRO": [
        (
            "Reader Note 0 — Thompson and the Purpose of This Lecture (Student Guide)",
            "D’Arcy Wentworth Thompson (1860–1948) was a Scottish zoologist and classical scholar. He studied at Edinburgh and Cambridge, became professor of biology at Dundee in 1884, and moved to St Andrews in 1917. He built a teaching museum, worked on fisheries, and translated Aristotle’s <em>History of Animals</em> into English in 1910. His knowledge of both Greek and animals helps explain the confidence and affection of this lecture. [E1]<br/><br/>"
            "<strong>Reader Note 1 — Why the lecture begins with Spencer:</strong> A <em>prooemion</em> is an opening address or prelude. This was the Herbert Spencer commemorative lecture, so Thompson first honours Spencer before turning to Aristotle. Some opening allusions are tributes: <em>Ipse Epicurus obit</em> means “Epicurus himself died”; <em>ex cathedra</em> means “speaking with authority”; <em>ein tapfrer Krieger</em> means “a brave warrior”."
        )
    ],
    "1": [
        (
            "Reader Note 2–4 — Evolution, Differentiation & Who Goethe Was (Student Guide)",
            "<strong>Homogeneous vs. Heterogeneous:</strong> <em>Homogeneous</em> means alike throughout; <em>heterogeneous</em> means made up of unlike parts. Imagine a developing organism in which regions become distinguishable as different tissues and organs—that is <em>differentiation</em>.<br/><br/>"
            "<strong>Who Goethe Was:</strong> Johann Wolfgang von Goethe (1749–1832), the German poet and statesman, also pursued anatomy, botany, geology, and colour. In <em>The Metamorphosis of Plants</em> (1790), Goethe compared leaves, sepals, petals, and stamens as related variations on an underlying organ type. Repeated, similar parts give way to more specialized parts with different jobs (<em>subordination of parts</em>). [E3, E4]"
        )
    ],
    "2": [
        (
            "Reader Note 5–7 — The Unknowable, Why Size Affects Shape & Ethics (Student Guide)",
            "<strong>Why Size and Forces Affect Shape:</strong> <em>Stress</em> is force distributed over an area; <em>strain</em> is the resulting relative deformation. If an animal were enlarged uniformly to twice its length, its volume and weight would increase eightfold ($2^3 = 8$), while the cross-sectional bone area supporting that weight would increase only fourfold ($2^2 = 4$). Keeping the exact same proportions therefore changes the mechanical demands on its supports—which is why a bridge-builder’s knowledge matters to anatomy. <em>Entwickelungsmechanik</em> means developmental mechanics.<br/><br/>"
            "<strong>From Biology to Ethics:</strong> <em>Altruism</em> concerns others’ welfare; <em>egoism</em> centres on one’s own; <em>bellum omnium contra omnes</em> means “the war of all against all”; and <em>confessio ignorantis</em> means an honest admission of not knowing."
        )
    ],
    "3": [
        (
            "Reader Note 8 — Why Aristotle Is Important Here (Student Guide)",
            "Aristotle (384–322 BCE) studied in Plato’s Academy, later taught Alexander, and founded his own school, the Lyceum, in Athens. He investigated animals alongside logic, ethics, politics, and the nature of knowledge. <em>Maestro di color che sanno</em> (Dante’s phrase) means “master of those who know”. <em>Natural History</em> here refers to Aristotle’s <em>History of Animals</em>. [E6]"
        )
    ],
    "4": [
        (
            "Reader Note 9 — The Relationship Between Aristotle and Alexander (Student Guide)",
            "The printed 1913 lecture accidentally calls Aristotle Alexander’s ‘pupil’ in the story about collecting specimens. The historical relationship is the reverse: Aristotle was Alexander’s tutor. [E2, E7] Thompson himself rejects the specimen-collecting legend as unsupported."
        )
    ],
    "5": [
        (
            "Reader Note 10 — How the Place Names Support the Argument (Student Guide)",
            "Thompson treats repeated references to the same coastal region as clues to where Aristotle observed animals in person. <em>Insula nobilis et amoena</em> means “a celebrated and pleasant island” (Lesbos). If Aristotle’s biological investigations belong to his years at the Lagoon of Pyrrha on Lesbos, they preceded his later philosophical teaching in Athens."
        )
    ],
    "6": [
        (
            "Reader Note 11 — Why an Unattractive Animal Can Reward Study (Student Guide)",
            "The beauty at issue is partly the intelligibility of a living whole: how its parts and activities belong together. Aristotle’s argument in <em>Parts of Animals</em> I.5 is that even humble animals we find unpleasant to look at can reward the search for causes. [E3, E8]"
        )
    ],
    "7": [
        (
            "Reader Note 12–13 — Unfamiliar Marine Terms & Why the Shark Placenta Matters (Student Guide)",
            "<strong>Cephalopod Terms:</strong> In these paragraphs, <em>cuttle-fishes</em> is used broadly for cephalopods (cuttlefish, squid, and octopuses). <em>Sepia</em> is a cuttlefish; <em>calamaries</em> are squid. The <em>mantle</em> is the muscular covering around the main body, and the <em>funnel</em> directs expelled water. A <em>cuttle-bone</em> is the internal support of a cuttlefish; the <em>pen</em> is the thinner internal support in squid. Aristotle’s <em>bloodless animals</em> lacked red vertebrate blood (though they still have circulating fluids).<br/><br/>"
            "<strong>The Shark Placenta:</strong> <em>Viviparous</em> means bearing live young; <em>oviparous</em> means laying eggs. In the smooth dogfish (<em>Mustelus laevis</em>), the embryonic <em>yolk-sac</em> interweaves with maternal tissue to form a nutrient-exchange <em>placenta</em>."
        )
    ],
    "8": [
        (
            "Reader Note 14 — Why Aristotle Put the Centre of Sensation in the Heart (Student Guide)",
            "<em>Punctum saliens</em> means the “leaping point”: the conspicuous beating heart speck in a three-day chick embryo. Its early visibility, its connection to blood vessels, and its central position made it a plausible centre of living activity for Aristotle. The <em>allantois</em> is an embryonic membrane involved in gas exchange and waste storage. <em>Cor</em> and <em>cerebrum</em> mean “heart” and “brain”, terms early botanists also applied to the central growing node of a seedling."
        )
    ],
    "9": [
        (
            "Reader Note 15 — What a Natural History of Constitutions Means (Student Guide)",
            "A <em>politeia</em> is a constitution or arrangement of civic government. Thompson compares gathering descriptions of more than 200 city constitutions with gathering descriptions of different animals. <em>Inductive</em> means working from particular observed cases toward a general account."
        )
    ],
    "10": [
        (
            "Reader Note 16–20 — Hylomorphism, the Eye Analogy & the Nested Capacities of Soul (Student Guide)",
            "<strong>What <em>Psychē</em> Means:</strong> <em>Psychē</em>, conventionally translated soul, has a wider scope than the modern idea of a conscious mind. Plants have it because they nourish themselves, grow, and reproduce. Animals also perceive; human beings also reason.<br/><br/>"
            "<strong>Form and Matter (Hylomorphism):</strong> <em>Hylē</em> means matter (what a thing is made of) and <em>morphē</em> means form (what makes it the kind of thing it is). If an eye were an entire animal, its power of sight would stand to it as soul stands to an organism; a painted eye shares the outline while lacking the capacity.<br/><br/>"
            "<strong>Why the Lower Capacities Remain in the Higher:</strong> A person who reasons still needs nourishment and can still feel and perceive—just as a triangle is contained inside a four-sided figure (Figure 5 below). [E4, E9]"
        )
    ],
    "11": [
        (
            "Reader Note 21–22 — A Thought Process vs. a Time Process & the Ladder vs. the Tree (Student Guide)",
            "<strong>Thought Process vs. Time Process:</strong> A <em>time process</em> describes what happened first and what happened later in history. A <em>thought process</em> orders concepts by what they presuppose or include (for example, perception presupposes nutrition).<br/><br/>"
            "<strong>The Ladder vs. the Branching Tree:</strong> <em>Scala naturae</em> means “ladder of nature” (one graded scale). Bergson replaces the single ladder with a branching tree in which plant torpor, insect instinct, and vertebrate intelligence diverge along separate directions."
        )
    ],
    "12": [
        (
            "Reader Note 23 — Potentiality, First Actuality & Entelechy (Student Guide)",
            "<em>Entelecheia</em>, usually translated actuality, concerns the realization of a capacity. Aristotle distinguishes having an ability from actively using it (Figure 6 below): someone who has learned to read possesses the ability (<em>first actuality</em>) even while asleep or not looking at a page; actively reading right now is <em>second actuality</em>. Soul is <em>first actuality</em>—the organized living capacity of a body—which is distinct from Hans Driesch’s later vitalist claim that life is an exception to physical laws. [E9]"
        )
    ],
    "13": [
        (
            "Reader Note 24–25 — Four Ways of Answering ‘Why’ & Science with the Classics (Student Guide)",
            "<strong>Aristotle’s Four Causes (Illustrated with a Bronze Statue):</strong><br/>"
            "• <strong>Material Cause:</strong> What is it made from? (<em>Bronze</em> / living tissue & molecules)<br/>"
            "• <strong>Formal Cause:</strong> What makes it this kind of thing? (<em>Its defining structure as a statue</em> / body plan)<br/>"
            "• <strong>Efficient Cause:</strong> What brings it into being? (<em>The sculptor’s making</em> / developmental mechanics)<br/>"
            "• <strong>Final Cause:</strong> For what end is it made? (<em>To commemorate a person</em> / biological function)<br/><br/>"
            "<strong>Closing Connection:</strong> Greek scholarship lets Thompson read Aristotle closely; zoology lets him recognize the real animals, organs, and developmental problems being described. [E5, E11]"
        )
    ],
}


def main():
    raw_lines = MD_PATH.read_text(encoding="utf-8").splitlines()
    body_parts = []
    toc_items = []
    i = 0
    n = len(raw_lines)

    while i < n:
        line = raw_lines[i].strip()
        if not line:
            i += 1
            continue
        if line == "---":
            body_parts.append("<hr />")
            i += 1
            continue
        if line.startswith("# "):
            body_parts.append(f'<h1 itemprop="headline">{format_inline(line[2:])}</h1>')
            i += 1
            continue
        if line.startswith("## "):
            sec_title = line[3:]
            sec_id = slugify(sec_title)
            is_ed = "EDITORIAL" in sec_title or "SOURCES" in sec_title
            cls = ' class="editorial-heading"' if is_ed else ""
            body_parts.append(f'<h2 id="{sec_id}"{cls}>{format_inline(sec_title)}</h2>')
            toc_items.append((sec_id, re.sub(r"[`*]", "", sec_title)))
            if "PROOEMION" in sec_title:
                for rn_title, rn_body in READER_NOTES_BY_COMMENTARY.get("INTRO", []):
                    body_parts.append(
                        f'<aside class="student-note-card" data-layer="student">'
                        f'<div class="student-note-hdr">🎓 {rn_title}</div>'
                        f'<div class="student-note-body">{rn_body}</div>'
                        f"</aside>"
                    )
            i += 1
            continue
        if line.startswith("!["):
            m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
            if m:
                alt, src = m.group(1), m.group(2)
                body_parts.append(
                    f'<figure class="plate-figure" itemprop="image" itemscope itemtype="https://schema.org/ImageObject">'
                    f'<img src="{src}" alt="{html.escape(alt)}" loading="lazy" itemprop="contentUrl" />'
                    f'<meta itemprop="caption" content="{html.escape(alt)}" />'
                    f"</figure>"
                )
            i += 1
            continue
        if line.startswith("> "):
            quote_lines = []
            while i < n and raw_lines[i].strip().startswith(">"):
                q_line = raw_lines[i].strip()[1:].strip()
                quote_lines.append(q_line)
                i += 1
            joined_q = "<br />".join(format_inline(q) for q in quote_lines)
            if "EDITORIAL COMMENTARY" in joined_q:
                m_num = re.search(r"EDITORIAL COMMENTARY\s+(\d+)", joined_q)
                c_num = m_num.group(1) if m_num else ""
                c_id = f"editorial-commentary-{c_num}" if c_num else "editorial-commentary"
                body_parts.append(
                    f'<aside id="{c_id}" class="editorial-card" data-layer="research">{joined_q}</aside>'
                )
                # Inject the matching 13-year-old level Student Reader Note from the .docx!
                for rn_title, rn_body in READER_NOTES_BY_COMMENTARY.get(c_num, []):
                    body_parts.append(
                        f'<aside class="student-note-card" data-layer="student">'
                        f'<div class="student-note-hdr">🎓 {rn_title}</div>'
                        f'<div class="student-note-body">{rn_body}</div>'
                        f"</aside>"
                    )
                if c_num == "10":
                    body_parts.append(
                        '<figure class="plate-figure" data-layer="student">'
                        '<img src="./media/thompson_1913_fig5_nested_soul_museum.png" '
                        'alt="Figure 5 — Nested Capacities of the Soul in Aristotle De Anima II.2-3 framed by Vatican Manuscript and 1579 Scala Naturae" loading="lazy" />'
                        '<p class="plate-caption"><em>Figure 5 (Student Reader Edition) — Nested capacities in the account Thompson presents: a human retains nutritive and sensitive capacities while also possessing rational capacities (“as a triangle within a square,” De Anima II.2–3), framed by the medieval Vatican manuscript of De Anima and Valades’s 1579 Scala Naturae copperplate.</em></p>'
                        "</figure>"
                    )
                elif c_num == "12":
                    body_parts.append(
                        '<figure class="plate-figure" data-layer="student">'
                        '<img src="./media/thompson_1913_fig6_first_actuality_museum.png" '
                        'alt="Figure 6 — Potentiality, First Actuality and Second Actuality in Aristotle De Anima" loading="lazy" />'
                        '<p class="plate-caption"><em>Figure 6 (Student Reader Edition) — Having an ability differs from actively exercising it: why soul as “first actuality” (ἐντελέχεια ἡ πρώτη) remains present even when an organism is asleep, and why that definition differs from modern vitalism.</em></p>'
                        "</figure>"
                    )
            else:
                body_parts.append(f'<blockquote class="verse-quote">{joined_q}</blockquote>')
            continue
        if line.startswith("- ") or line.startswith("* "):
            ul_items = []
            while i < n and (raw_lines[i].strip().startswith("- ") or raw_lines[i].strip().startswith("* ")):
                item_txt = raw_lines[i].strip()[2:].strip()
                is_conn = "Morpholib connection" in item_txt
                li_cls = ' class="morpholib-conn" data-layer="research"' if is_conn else ""
                i += 1
                # Attach indented continuation lines (e.g. source URLs) to the same <li>
                while i < n and raw_lines[i].startswith("  ") and raw_lines[i].strip():
                    item_txt += " <br/>" + raw_lines[i].strip()
                    i += 1
                ul_items.append(f"<li{li_cls}>{format_inline(item_txt)}</li>")
            body_parts.append("<ul>\n" + "\n".join(ul_items) + "\n</ul>")
            continue

        para_lines = [line]
        i += 1
        while i < n and raw_lines[i].strip() and not raw_lines[i].strip().startswith(("#", "![", ">", "- ", "* ", "---")):
            para_lines.append(raw_lines[i].strip())
            i += 1
        p_txt = " ".join(para_lines)
        if p_txt.startswith("*Plate ") or p_txt.startswith("*Figure "):
            body_parts.append(f'<p class="plate-caption">{format_inline(p_txt)}</p>')
        elif p_txt.startswith("`") and p_txt.endswith("`"):
            body_parts.append(f'<p class="editorial-synthesis-para" data-layer="research">{format_inline(p_txt)}</p>')
        else:
            body_parts.append(f"<p>{format_inline(p_txt)}</p>")

    toc_html = " · ".join(
        f'<a href="#{sid}">{html.escape(stitle)}</a>' for sid, stitle in toc_items
    ) + ' · <a href="#faq-section">FAQ &amp; Key Questions</a>'
    json_ld_str = build_json_ld()

    faq_visible_html = """
    <hr />
    <section id="faq-section" class="faq-section" aria-labelledby="faq-heading">
      <h2 id="faq-heading" class="editorial-heading">FREQUENTLY ASKED QUESTIONS &amp; KEY TAKEAWAYS</h2>
      <details class="faq-item" open>
        <summary>1. Where did Aristotle conduct his main marine biology observations according to D’Arcy Wentworth Thompson?</summary>
        <p>In his 1913 Oxford lecture <em>On Aristotle as a Biologist</em>, D’Arcy Wentworth Thompson demonstrates from internal geographical evidence in the <em>Historia Animalium</em> that Aristotle conducted a pivotal cluster of his marine zoological studies at the sheltered <strong>Lagoon of Pyrrha</strong> (modern <strong>Gulf of Kalloni</strong>) on the island of <strong>Lesbos (Mitylene)</strong> between 345 and 342 BCE—before he went to Macedon to tutor Alexander and ten years before he founded the Lyceum in Athens.</p>
      </details>
      <details class="faq-item" open>
        <summary>2. What zoological discoveries of Aristotle were later confirmed by modern anatomists like Cuvier, Steno, and Johannes Müller?</summary>
        <p>Thompson highlights three famous anatomical and embryological cases: <strong>(a)</strong> the specialized reproductive <em>hectocotylus</em> arm of male cephalopods and the cephalic yolk-sac of the cuttlefish (<em>Sepia</em>) embryo; <strong>(b)</strong> the yolk-sac placenta (<em>placenta vitellina</em>) of the smooth dogfish <em>Mustelus laevis</em> (γαλεὸς λεῖος), rediscovered by Nicolaus Steno in 1673 and celebrated by Johannes Müller in 1842; and <strong>(c)</strong> the day-by-day embryology of the chick in the egg, beginning with the leaping point (<em>punctum saliens</em>) of the beating heart on the third day.</p>
      </details>
      <details class="faq-item" open>
        <summary>3. How does D’Arcy Thompson contrast Herbert Spencer and Aristotle as thinkers about living form?</summary>
        <p>Thompson argues that <strong>Herbert Spencer</strong> approached biology with the training and habits of a railway and bridge engineer—focusing on mechanical causation, stress, strain, and scaling limits ($\propto L^2$ bone cross-section versus $\propto L^3$ gravitational bulk)—whereas <strong>Aristotle</strong> was first and foremost a field naturalist and comparative anatomist whose study of living organization, developmental stages, and biological capacities shaped his psychology (<em>De Anima</em>) and philosophy.</p>
      </details>
      <details class="faq-item" open>
        <summary>4. How does Thompson’s 1913 lecture connect to On Growth and Form (1917) and modern morphogenesis research?</summary>
        <p>The 1913 lecture bridges Spencer’s physical mechanics (Roux’s <em>Entwicklungsmechanik</em> and Culmann’s stress trajectories) with Aristotle’s focus on organized wholes. In <em>On Growth and Form</em> (1917), Thompson operationalizes this through coordinate transformations of biological outlines. In modern computational morphometrics and bioelectric patterning (e.g., Pai et al. 2012 on <em>Xenopus</em> transmembrane voltage $V_{\\text{mem}}$), this requires separating representation choice (600-point outlines vs. 17 skeletal landmarks), contour provenance, sampled numerical Jacobian checks, and experimental perturbation.</p>
      </details>
    </section>
    <footer class="crosslinks-footer" role="contentinfo">
      <strong>Explore Related Modules in Project Delphi (<code>morpholib</code>):</strong>
      <div class="crosslinks-grid">
        <a href="index.html">🏛️ Aristotelian 10×10 Morphospace Atlas (116 Verified Models)</a>
        <a href="EXPLAINED_FOR_EVERYONE.md">🐟 Visual Guide: Resolving Thompson’s 1917 Fish Paradox Across 10,303 Species</a>
        <a href="bioelectricity/index.html">⚡ Interactive Bioelectricity &amp; Thompson Coordinate Sandbox</a>
        <a href="bioelectricity/sorting-morphogenesis.html">🧬 Self-Sorting Cells &amp; Causal Emergence 2.0 Lab</a>
        <a href="bioelectricity/galvani-interactive.html">🐸 Galvani (1791) Bioelectricity &amp; Live Animated Twitch Lab</a>
        <a href="substack_levin_latent_space_article.html">📄 Testing Michael Levin’s Latent Platonic Space Claims</a>
      </div>
    </footer>
    """

    html_doc = f"""<!DOCTYPE html>
<html lang="en" prefix="og: https://ogp.me/ns# article: https://ogp.me/ns/article#">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>On Aristotle as a Biologist (1913) — D’Arcy Wentworth Thompson | Annotated Museum Edition &amp; Morphogenesis Guide</title>

  <!-- Primary SEO Meta Tags -->
  <meta name="description" content="Complete annotated and museum-illustrated edition of D’Arcy Wentworth Thompson’s 1913 Oxford lecture 'On Aristotle as a Biologist, With a Prooemion on Herbert Spencer'. Features 17th–19th century copperplates (Malpighi, Johannes Müller’s Mustelus shark placenta, Vérany’s Argonauta, 1848 Admiralty Chart of Lesbos &amp; Pyrrha Lagoon, Turpin’s 1837 Urpflanze), 13 morphogenesis research commentaries, and 25 student reader notes." />
  <meta name="keywords" content="D'Arcy Wentworth Thompson, On Aristotle as a Biologist, Herbert Spencer Lecture 1913, Lagoon of Pyrrha Lesbos, Gulf of Kalloni, Historia Animalium, De Anima, entelechy, Mustelus laevis yolk-sac placenta, Johannes Müller 1842, Marcello Malpighi chick embryo, Goethe Urpflanze morphology, On Growth and Form, morphogenesis, bioelectric patterning, Project Delphi, morpholib" />
  <meta name="author" content="D’Arcy Wentworth Thompson (1913); Annotated &amp; Illustrated by Joel Dietz (Project Delphi)" />
  <meta name="robots" content="index, follow, max-image-preview:large, max-snippet:-1" />
  <link rel="canonical" href="{CANONICAL_URL}" />

  <!-- Open Graph / Facebook / LinkedIn / Substack -->
  <meta property="og:type" content="article" />
  <meta property="og:site_name" content="Project Delphi (morpholib)" />
  <meta property="og:title" content="On Aristotle as a Biologist (1913) — D’Arcy Wentworth Thompson | Annotated Museum Edition" />
  <meta property="og:description" content="Follow knowledge from the Lagoon of Pyrrha on Lesbos and the dissecting table into De Anima and On Growth and Form—illustrated with 17th–19th century museum copperplates, research commentary, and friendly student notes." />
  <meta property="og:url" content="{CANONICAL_URL}" />
  <meta property="og:image" content="{OG_IMAGE_URL}" />
  <meta property="og:image:alt" content="Plate II — 1848/1860 British Admiralty Chart of Lesbos (Lagoon of Pyrrha) and 17th–19th Century Museum Specimen Engravings" />
  <meta property="og:locale" content="en_GB" />
  <meta property="article:published_time" content="1913-02-14" />
  <meta property="article:modified_time" content="2026-10-25" />
  <meta property="article:section" content="History and Philosophy of Biology &amp; Morphogenesis" />

  <!-- Twitter / X Card -->
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="On Aristotle as a Biologist (1913) — D’Arcy Wentworth Thompson (Museum-Illustrated Edition)" />
  <meta name="twitter:description" content="D’Arcy Thompson’s 1913 Oxford lecture on Herbert Spencer and Aristotle at the Lagoon of Pyrrha, illustrated with archival museum copperplates, MorphoEval connections, and student reader notes." />
  <meta name="twitter:image" content="{OG_IMAGE_URL}" />

  <!-- Highwire Press / Google Scholar Citation Metadata -->
  <meta name="citation_title" content="On Aristotle as a Biologist, With a Prooemion on Herbert Spencer (Annotated &amp; Illustrated Edition)" />
  <meta name="citation_author" content="Thompson, D'Arcy Wentworth" />
  <meta name="citation_author" content="Dietz, Joel" />
  <meta name="citation_publication_date" content="1913/02/14" />
  <meta name="citation_online_date" content="2026/10/25" />
  <meta name="citation_publisher" content="Clarendon Press, Oxford (1913); Project Delphi (2026)" />
  <meta name="citation_public_url" content="{CANONICAL_URL}" />
  <meta name="citation_language" content="en" />

  <!-- Structured Data (Schema.org JSON-LD: ScholarlyArticle + BreadcrumbList + FAQPage) -->
  <script type="application/ld+json">
{json_ld_str}
  </script>

  <style>
    :root {{
      --ink: #1c1917;
      --paper: #f7f3e8;
      --card: #fcfaf2;
      --ed-bg: #f0f9ff;
      --ed-border: #0284c7;
      --ed-ink: #0f172a;
      --student-bg: #fefce8;
      --student-border: #b45309;
      --student-ink: #451a03;
      --conn-bg: #f0fdf4;
      --conn-border: #15803d;
      --gold-border: #8c6d3f;
    }}
    body {{
      margin: 0;
      padding: 0;
      background: #0f172a;
      color: var(--ink);
      font-family: "Georgia", "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Palatino, serif;
      line-height: 1.78;
      font-size: 18px;
    }}
    .topbar {{
      position: sticky;
      top: 0;
      z-index: 100;
      background: #0b1325;
      color: #f8fafc;
      padding: 10px 22px;
      border-bottom: 2px solid #d97706;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 10px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 13.5px;
    }}
    .topbar-links {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .topbar a {{
      color: #fde68a;
      text-decoration: none;
      font-weight: 600;
    }}
    .topbar a:hover {{
      text-decoration: underline;
    }}
    .mode-btn {{
      background: #1e293b;
      color: #e2e8f0;
      border: 1px solid #475569;
      border-radius: 999px;
      padding: 5px 11px;
      font-size: 12.5px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }}
    .mode-btn.active {{
      background: #f59e0b;
      color: #0f172a;
      border-color: #fef08a;
      font-weight: 800;
    }}
    nav.toc-bar {{
      background: #efe9d8;
      border: 1px solid var(--gold-border);
      border-radius: 8px;
      padding: 12px 16px;
      margin: 18px 0 28px 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 0.86rem;
      line-height: 1.6;
    }}
    nav.toc-bar a {{
      color: #78350f;
      font-weight: 600;
      text-decoration: none;
    }}
    nav.toc-bar a:hover {{
      text-decoration: underline;
    }}
    main.reader {{
      max-width: 920px;
      margin: 30px auto 80px auto;
      background: var(--paper);
      padding: 52px 62px;
      border-radius: 12px;
      border: 1px solid var(--gold-border);
      box-shadow: 0 14px 40px rgba(0, 0, 0, 0.5);
    }}
    @media (max-width: 768px) {{
      main.reader {{
        padding: 26px 20px;
        margin: 12px;
      }}
    }}
    h1 {{
      font-size: 2.15rem;
      line-height: 1.24;
      color: #1c1917;
      margin-top: 0;
      margin-bottom: 10px;
    }}
    h2 {{
      font-size: 1.45rem;
      color: #1c1917;
      border-bottom: 2px solid var(--gold-border);
      padding-bottom: 6px;
      margin-top: 42px;
    }}
    h2.editorial-heading {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      color: #0369a1;
      border-bottom-color: #38bdf8;
      letter-spacing: 0.02em;
    }}
    p {{
      margin: 16px 0;
      text-align: justify;
    }}
    /* Research & Morphogenesis Commentary Card */
    .editorial-card {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      background: var(--ed-bg);
      border-left: 5px solid var(--ed-border);
      border-radius: 6px;
      padding: 15px 20px;
      margin: 22px 0;
      color: var(--ed-ink);
      font-size: 0.94rem;
      line-height: 1.62;
      box-shadow: 0 2px 8px rgba(2, 132, 199, 0.08);
    }}
    /* Friendly 13-Year-Old Student Reader Note Card (from .docx) */
    .student-note-card {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      background: var(--student-bg);
      border-left: 5px solid var(--student-border);
      border-radius: 6px;
      padding: 14px 20px;
      margin: 18px 0;
      color: var(--student-ink);
      font-size: 0.93rem;
      line-height: 1.62;
      box-shadow: 0 2px 8px rgba(180, 83, 9, 0.08);
    }}
    .student-note-hdr {{
      font-weight: 800;
      color: #92400e;
      margin-bottom: 6px;
      font-size: 0.96rem;
    }}
    .editorial-card .editorial-font,
    .editorial-synthesis-para .editorial-font,
    li.morpholib-conn .editorial-font {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      background: transparent;
      padding: 0;
    }}
    .editorial-synthesis-para {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      background: #f8fafc;
      border-left: 4px solid #64748b;
      padding: 12px 18px;
      border-radius: 4px;
      font-size: 0.95rem;
      line-height: 1.65;
      color: #1e293b;
      text-align: left;
    }}
    li.morpholib-conn {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
      background: var(--conn-bg);
      border-left: 4px solid var(--conn-border);
      padding: 12px 16px;
      margin: 12px 0;
      border-radius: 6px;
      font-size: 0.94rem;
      line-height: 1.6;
      list-style: none;
    }}
    blockquote.verse-quote {{
      margin: 20px 36px;
      padding-left: 18px;
      border-left: 3px solid var(--gold-border);
      font-style: italic;
      color: #44403c;
    }}
    figure.plate-figure {{
      margin: 28px 0 10px 0;
      text-align: center;
    }}
    figure.plate-figure img {{
      max-width: 100%;
      height: auto;
      border-radius: 6px;
      border: 1.5px solid var(--gold-border);
      box-shadow: 0 6px 20px rgba(28, 25, 23, 0.18);
    }}
    p.plate-caption {{
      font-family: "Georgia", serif;
      font-size: 0.88rem;
      line-height: 1.55;
      color: #44403c;
      text-align: left;
      background: #efe9d8;
      border-left: 3px solid var(--gold-border);
      padding: 10px 14px;
      border-radius: 4px;
      margin-top: 6px;
      margin-bottom: 28px;
    }}
    hr {{
      border: none;
      border-top: 1px solid #d6cbb1;
      margin: 36px 0;
    }}
    sup a {{
      color: #92400e;
      text-decoration: none;
      font-weight: 700;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      font-size: 0.78em;
    }}
    ul {{
      padding-left: 20px;
    }}
    li {{
      margin-bottom: 8px;
    }}
    .faq-section {{
      margin-top: 36px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    details.faq-item {{
      background: #efe9d8;
      border: 1px solid var(--gold-border);
      border-radius: 6px;
      padding: 12px 16px;
      margin: 12px 0;
    }}
    details.faq-item summary {{
      font-weight: 700;
      color: #1c1917;
      cursor: pointer;
      font-size: 0.96rem;
    }}
    details.faq-item p {{
      margin: 10px 0 4px 0;
      font-size: 0.92rem;
      line-height: 1.6;
      color: #292524;
      text-align: left;
    }}
    footer.crosslinks-footer {{
      margin-top: 38px;
      padding: 18px 20px;
      background: #0b1325;
      color: #f8fafc;
      border-radius: 8px;
      border: 1px solid #38bdf8;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 0.9rem;
    }}
    .crosslinks-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .crosslinks-grid a {{
      color: #7dd3fc;
      text-decoration: none;
      background: #16223b;
      padding: 8px 12px;
      border-radius: 6px;
      border: 1px solid #263554;
      font-weight: 600;
      font-size: 0.84rem;
    }}
    .crosslinks-grid a:hover {{
      border-color: #facc15;
      color: #fef08a;
    }}
    body[data-view="research"] [data-layer="student"] {{
      display: none !important;
    }}
    body[data-view="student"] [data-layer="research"] {{
      display: none !important;
    }}
    body[data-view="lecture-only"] [data-layer="research"],
    body[data-view="lecture-only"] [data-layer="student"] {{
      display: none !important;
    }}
  </style>
</head>
<body data-view="both">
  <header class="topbar" role="banner">
    <div>
      <strong>Project Delphi</strong> · D’Arcy Wentworth Thompson (1913) — <em>On Aristotle as a Biologist</em>
    </div>
    <div class="topbar-links" role="navigation" aria-label="Commentary layer and navigation controls">
      <span style="color:#94a3b8;font-size:12px;">Annotations:</span>
      <button type="button" class="mode-btn active" onclick="setCommentaryMode('both', this)" title="Show both the Research/Morphogenesis Commentary and the Friendly Student Guide Notes">🏛️ + 🎓 Both Layers</button>
      <button type="button" class="mode-btn" onclick="setCommentaryMode('research', this)" title="Show only the Critical Morphogenesis Research Commentary (Commentaries 1–13 &amp; Morpholib Connections)">🔬 Research Commentary</button>
      <button type="button" class="mode-btn" onclick="setCommentaryMode('student', this)" title="Show only the Friendly 13-Year-Old Level Student Guide (Reader Notes 1–25 &amp; Figures 5–6)">🎓 Student Guide (13yo)</button>
      <button type="button" class="mode-btn" onclick="setCommentaryMode('lecture-only', this)" title="Read D'Arcy Thompson's 1913 Oxford Lecture and museum plates uninterrupted">📜 1913 Text Only</button>
      <a href="index.html">← Morphospace Atlas</a>
      <a href="THOMPSON_1913_ARISTOTLE_AS_A_BIOLOGIST.md">Markdown</a>
    </div>
  </header>

  <main class="reader" role="main" itemscope itemtype="https://schema.org/ScholarlyArticle">
    <nav class="toc-bar" aria-label="Table of Contents">
      <strong>Contents &amp; Quick Jump:</strong> {toc_html}
    </nav>
    {chr(10).join(body_parts)}
    {faq_visible_html}
  </main>

  <script>
    function setCommentaryMode(mode, btn) {{
      document.body.setAttribute('data-view', mode);
      document.querySelectorAll('.mode-btn').forEach(function(b) {{
        b.classList.remove('active');
      }});
      if (btn) btn.classList.add('active');
    }}
  </script>
</body>
</html>
"""
    HTML_PATH.write_text(html_doc, encoding="utf-8")
    print(f"Saved {HTML_PATH} ({len(html_doc):,} bytes)")


if __name__ == "__main__":
    main()
