# Aspect taxonomy (reusable, domain-specific)

Used by the manual annotation in `docs/aspect_annotation_guidelines.md`. Two domains are covered and they are kept apart: **product reviews** (Amazon All Beauty, English) and **social-media comments** (DravidianCodeMix, YouTube movie comments in Tamil, Malayalam and Kannada, code-mixed with English). No category is shared between the two domains except `GENERAL` and `OTHER_ASPECT`; cross-domain aspect comparison is not claimed.

**Status: proposed vocabulary, nothing here has been validated by annotation yet.** Aliases are matching hints for annotators and for keyword pre-screening; they are not annotations and their accuracy is unmeasured.

Machine-readable sources: product reviews `ontology/product_aspect_taxonomy.json` (new); social media `ontology/aspect_taxonomy.json` and `ontology/wsd_lexicon_full.json` (existing, used by the WSD component and left unchanged).

## 1. Principles

1. An **aspect** is a facet of the entity being discussed (a product, a film), named or clearly implied in the message. One span gets exactly one category.
2. `GENERAL` means the opinion targets the entity as a whole ("love it", "waste of money"). It is a real target, so it is **not** the same as a missing aspect.
3. `OTHER_ASPECT` means a clear facet exists but no category covers it; the annotator writes it in `notes`. Frequent cases are promoted to categories in a later taxonomy version.
4. Missing or unclear aspects are recorded in the separate `aspect_status` field (`NO_ASPECT`, `UNCLEAR_ASPECT`), never as a sentiment label. `NEUTRAL` is a sentiment toward a clearly present aspect.
5. Aliases never decide a label by themselves: `mass`, `punch` and `climax` (social media) and `light`, `work`, `last` (product reviews) are ambiguous. The annotator reads the sentence.
6. Versioning: any change to a category id, definition or alias list gets a new version number in the file's `_meta` and a line in the changelog below; annotations always record the version they used.

## 2. Product reviews (Amazon All Beauty, English)

File: `ontology/product_aspect_taxonomy.json`. Families group categories for reporting only.

| Family | Meaning |
|---|---|
| PERFORMANCE | How well the product does its job or what it does to the user. |
| SENSORY | What the user perceives directly: smell, feel, look. |
| PHYSICAL | The product as an object: packaging, size, durability. |
| ECONOMIC | Price and value. |
| SERVICE | Everything around the product: delivery, seller, authenticity. |
| GENERAL | The product as a whole, with no specific facet. |

The last column is the share of the 23,736 reviews of the project's OET sequences in which at least one alias occurs (`scripts/sequence_readiness.py`, whole-word match on the original text). It shows how often a category *can* occur; it overstates true aspect mentions because aliases such as `soft`, `used` or `work well` also occur in other senses.

| Category | Family | Definition | Aliases (`*` = any word ending) | Alias hit rate |
|---|---|---|---|---|
| `EFFECTIVENESS` | PERFORMANCE | Whether the product achieves its purpose: results, moisturising, cleaning, volume, hold, shine. | effective*, results, result, works, worked, work well, difference, improv*, helped, helps, moisturiz*, moisturis*, hydrat*, cleans, cleaned, volume, shine, shiny, frizz*, smooth*, softer, soft | 35.0% |
| `SKIN_HAIR_REACTION` | PERFORMANCE | Effect on the body that is a reaction or safety issue: irritation, allergy, rash, breakout, burning, ingredients and chemicals. | irritat*, rash, allerg*, break out, breakout*, broke out, burn*, itch*, sting*, sensitive skin, ingredient*, chemical*, toxic, paraben*, sulfate*, dry skin, oily skin, acne, pimple* | 10.9% |
| `FRAGRANCE` | SENSORY | Smell of the product. | scent*, smell*, fragrance*, odor, odour, perfume*, aroma*, stink*, stinks, cologne | 16.3% |
| `TEXTURE` | SENSORY | How the product feels or behaves in the hand and on the body: consistency, greasiness, lather. | texture, consistency, thick, thin, greasy, sticky, creamy, runny, watery, oily, lather*, foam*, gritty, heavy, lightweight, silky | 19.5% |
| `APPEARANCE_COLOR` | SENSORY | Colour, shade, finish or visual look of the product or of the result. | color, colour, shade, shades, tone, pigment*, coverage, finish, looks, looked, appearance, glitter, sparkl* | 19.4% |
| `PACKAGING` | PHYSICAL | Container, packaging and closures: bottle, pump, cap, jar, tube, leaks, seals. | packaging, package, packaged, bottle*, container*, pump, cap, lid, jar, tube, leak*, sealed, seal, dispenser, spray nozzle, box | 16.2% |
| `SIZE_QUANTITY` | PHYSICAL | How much product there is or how big the item is. | size, sized, amount, quantity, ounce*, oz, ml, small, tiny, bigger, larger, large, little bit, half empty | 16.4% |
| `DURABILITY` | PHYSICAL | How long the product or its effect lasts, and whether it breaks or deteriorates. | last, lasts, lasted, long lasting, long-lasting, longevity, durable, durability, broke, broken, fell apart, wear, shelf life, expire*, dried out, stopped working | 13.2% |
| `EASE_OF_USE` | PERFORMANCE | How easy the product is to apply, handle or use, including applicators and tools. | easy to use, easy to apply, easily, application, applicator, applies, apply, applied, spread*, blend*, convenient, hard to use, difficult, instructions | 18.3% |
| `PRICE_VALUE` | ECONOMIC | Price, cost and whether the product is worth it. | price*, cost*, expensive, cheap*, value, worth, money, overpriced, affordable, bargain, deal, pricey | 19.6% |
| `AUTHENTICITY` | SERVICE | Whether the item is genuine, new and as described. | fake, counterfeit, authentic*, genuine, knock off, knockoff, imitation, used, tampered, not as described, as described, as pictured | 13.0% |
| `DELIVERY_SERVICE` | SERVICE | Shipping, delivery, seller behaviour, returns and refunds. | shipping, shipped, arrived, arrive, delivery, delivered, seller, customer service, refund*, return, returned, replacement, late, packaging damaged, damaged in | 5.7% |
| `GENERAL` | GENERAL | Opinion about the product as a whole with no specific facet (love it, waste of money, would buy again). Not a missing aspect: the target is the product itself. | overall, love it, loved it, hate it, waste of money, buy again, recommend, highly recommend, great product, best product, worst product | 16.3% |
| `OTHER_ASPECT` | none | A clear facet that no category above covers. Annotators write the facet in the notes column; frequent cases are promoted to new categories. | none | not applicable |

Practical notes for annotators:

* `EFFECTIVENESS` versus `SKIN_HAIR_REACTION`: a product that does not work is `EFFECTIVENESS`; a product that causes a rash, burn or breakout is `SKIN_HAIR_REACTION`.
* `DURABILITY` covers both the product lasting and the effect lasting ("fragrance fades in an hour").
* `AUTHENTICITY` and `DELIVERY_SERVICE` are about the sale, not the product's qualities; "arrived broken" is `DELIVERY_SERVICE`, "broke after a week of use" is `DURABILITY`.
* Reviews are about one product. If a sentence is about a different product the reviewer compares to, annotate only the reviewed product's aspects and note `COMPARISON`.

## 3. Social-media comments (DravidianCodeMix, movie YouTube comments)

Source: the existing WSD taxonomy (7 categories derived from corpus frequency, `ontology/aspect_taxonomy.json`) with its multilingual alias lists (`ontology/wsd_lexicon_full.json`). Both are reused unchanged. This domain is movie and celebrity comments, so product-review aspects (battery, price, packaging) do not apply.

`Corpus %` is the share of the 53,263 comments containing a seed keyword, as measured when the taxonomy was built. Alias counts are the number of surface forms per language in the lexicon (Latin and native script together).

| Category | Family | Definition | Corpus % | Aliases (ta / ml / kn / en) |
|---|---|---|---|---|
| `fan_stardom` | POPULARITY | Commentary about an actor's or star's fan following, fan base, mass appeal, or hype -- not about the film's content itself. | 13.15 | 14/9/8/10 |
| `trailer_teaser` | PROMOTION | Reactions to a film's promotional trailer, teaser, or promo video, as distinct from reactions to the finished film. | 8.9 | 8/7/7/6 |
| `music_bgm` | CONTENT | Opinions about a film's songs, background score (BGM), or music in general. | 4.67 | 13/11/8/11 |
| `box_office_collection` | POPULARITY | Commentary on a film's commercial performance -- collections, hit/flop status, box-office numbers. | 3.85 | 11/9/8/8 |
| `dialogue` | CONTENT | Opinions about specific dialogue lines, dialogue delivery, or 'punch' dialogues. | 1.39 | 9/7/7/8 |
| `story_screenplay` | CONTENT | Opinions about the film's story, plot, or screenplay/script. | 1.09 | 11/9/9/8 |
| `hero_character` | CONTENT | Opinions about a lead hero, heroine, villain, or other named character/casting choice. | 1.08 | 15/16/17/10 |
| `GENERAL` | GENERAL | Opinion about the film, actor or video as a whole with no listed facet ("padam super", "semma movie"). Not part of the existing lexicon; judged by the annotator. | not measured | none |
| `OTHER_ASPECT` | none | A clear facet not covered above (for example cinematography, comedy, direction); written in `notes`. | not measured | none |

Families for this domain: `CONTENT` (the film's craft: music, dialogue, story, cast), `PROMOTION` (trailers and teasers), `POPULARITY` (fans and box office), `GENERAL`.

### Aliases by category and language

Latin-script forms are listed in full; the number of native-script forms (Tamil, Malayalam, Kannada script) is shown in brackets. The complete lists, with native-script forms, are in `ontology/wsd_lexicon_full.json`.

**`fan_stardom`**

* tamil: abhimani, fan, fans, fansla, hype, mass, massaa, massu, rasigan, rasigar, rasigargal (+3 native-script)
* malayalam: aaradhakan, aaradhakar, aradhakan, fan, fans, fansaanu, mass (+2 native-script)
* kannada: abhimani, abhimanigalu, fan, fangalu, fans, mass (+2 native-script)
* english: fan, fans, fanbase, mass, hype, fandom, fanboy, supporter, followers, stan

**`trailer_teaser`**

* tamil: firstlook, munnottam, promo, teaser, teaseru, trailer, traileru (+1 native-script)
* malayalam: promo, teaser, teesar, trailer, treyilar (+2 native-script)
* kannada: promo, teaser, tiisar, trailer, trellar (+2 native-script)
* english: trailer, teaser, promo, preview, first look, mass

**`music_bgm`**

* tamil: bgm, isai, music, paadal, paattu, paatu, pattu, song, songs, songu (+3 native-script)
* malayalam: bgm, gaanam, music, paattu, pattu, sangeetham, song, songs (+3 native-script)
* kannada: bgm, haadu, music, sangeeta, song, songs (+2 native-script)
* english: music, song, songs, bgm, background music, score, soundtrack, melody, tune, lyrics, mass

**`box_office_collection`**

* tamil: blockbuster, boxoffice, collection, flop, hit, superhit, tholvi, vasool, vasul (+2 native-script)
* malayalam: blockbuster, collection, flop, hit, hittu, kalekshan, superhit (+2 native-script)
* kannada: blockbuster, collection, flop, hit, kalekshan, superhit (+2 native-script)
* english: box office, boxoffice, collection, hit, flop, blockbuster, disaster, crore

**`dialogue`**

* tamil: dialog, dialogu, dialogue, punch, punchline, vasanam, vasanangal (+2 native-script)
* malayalam: dayalogu, dialog, dialogue, punch, sambhaashanam (+2 native-script)
* kannada: dailaag, dialog, dialogue, punch, sambhaashane (+2 native-script)
* english: dialogue, dialog, punch, punchline, delivery, lines, mass, climax

**`story_screenplay`**

* tamil: climax, kadhai, katha, kathai, screenplay, script, story, thiraikkathai, twist (+2 native-script)
* malayalam: climax, katha, screenplay, script, story, thirakkatha, twist (+2 native-script)
* kannada: chitrakathe, climax, kathe, screenplay, script, story, twist (+2 native-script)
* english: story, plot, screenplay, script, narrative, twist, climax, ending

**`hero_character`**

* tamil: acting, actor, actress, hero, heroine, kathanayagan, nadigai, nadigar, role, villain, villan (+4 native-script)
* malayalam: acting, actor, actress, hero, heroine, naayakan, naayika, nadan, nadi, villain, villan (+5 native-script)
* kannada: acting, actor, actress, hero, heroine, khalanaayaka, naayaka, naayaki, nata, nati, villain, villan (+5 native-script)
* english: hero, heroine, villain, actor, actress, character, role, casting, lead, punch

### Ambiguous forms

Three forms map to several categories in the lexicon. The WSD component resolves them by context; for manual annotation the annotator decides from the sentence, and an unresolvable use is annotated `UNCLEAR_ASPECT`.

| Form | Candidate categories |
|---|---|
| `mass` | dialogue, fan_stardom, music_bgm, trailer_teaser |
| `punch` | dialogue, hero_character |
| `climax` | dialogue, story_screenplay |

Existing WSD measurements on the first 5,000 comments per language (144,973 tokens): 4,997 lexicon matches, 682 ambiguous occurrences, 121 resolved. Sense-resolution accuracy has not been measured; manual annotation is the first gold data for aspects in this domain.

## 4. Adding a domain

1. Derive candidate categories from the domain's own text (frequency over the real corpus, as in the two files above); do not import categories from another domain.
2. Write a one-sentence definition and a boundary rule for each pair of easily confused categories.
3. Add `GENERAL` and `OTHER_ASPECT`.
4. Store the vocabulary as a JSON file under `ontology/` with a `_meta` block (purpose, status, derivation).
5. Pilot-annotate a small set, record disagreements, revise, and only then annotate at scale.

## Changelog

* v0.1 (2026-11-14): initial proposal. Product-review categories are new and unvalidated; social-media categories reuse the existing WSD taxonomy.
