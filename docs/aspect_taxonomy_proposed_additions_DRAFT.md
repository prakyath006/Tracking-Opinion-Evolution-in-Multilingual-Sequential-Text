# Proposed taxonomy additions (DRAFT, not approved)

**The official taxonomy is unchanged.** `docs/aspect_taxonomy.md` and `ontology/product_aspect_taxonomy.json` still define the 14 product categories plus `OTHER_ASPECT`.
No row of the provisional annotation file uses a category outside them. Everything below is a proposal from the adjudication workbook (rules sheet P2 to P6, Q3, Q4) and needs team approval, a taxonomy version bump and a changelog entry (taxonomy 1.6) before use.

| Proposal | What it would do | Current handling (provisional file) | Rows affected (workbook item IDs) | Cost if adopted |
|---|---|---|---|---|
| P2 `BUILD_QUALITY` | new category for manufacturing and build quality | `OTHER_ASPECT` with the facet in notes | ADJ-069, 092, 114; agreed pair 'manufacturing' (679989) | category list 14 to 15; BIO tag set grows; re-map rows |
| P3 `PHYSICAL_PROPERTIES` | new category for rigidity, weight, shape | `OTHER_ASPECT` | ADJ-061, 062, 070, 073, 108 | same |
| P4 `FEATURES_ACCESSORIES` | new category for pins, ribbons, battery, extras | `OTHER_ASPECT` | ADJ-016, 056, 057, 066, 067, 072, 074, 126 | same |
| P5 `DELIVERY_SERVICE` scope | extend the definition to availability; a rename to `SELLER_SERVICE_AVAILABILITY` was also suggested | kept as `DELIVERY_SERVICE` | ADJ-029, 136; rejected ADJ-137 (needs human) | a rename would break the category id in existing files; widening the definition does not |
| P6 speed / time-saving rule | map "takes less time" consistently | no row cites it | none | guideline text only |
| Q3 workaround rule | an evaluation of the reviewer's own workaround is not a product evaluation | rejection stays in the audit trail as NEEDS_HUMAN | ADJ-075 | guideline text only |
| Q4 fragrance-mediated effects | "boosts my mood" goes to `FRAGRANCE` | rejection kept (same mention as the fragrance span) | ADJ-049 | guideline text only |

**Recommendation (not applied).** Keep the 14 categories for the pilot. Count `OTHER_ASPECT` examples as the dataset grows and promote P2 to P4 in a later taxonomy version once about 20 examples per facet exist.
Widening `DELIVERY_SERVICE` to availability (P5) is the only proposal that changes no existing file.

**Not yet satisfied under the current taxonomy.** Taxonomy principle 3 requires the facet of every `OTHER_ASPECT` row to be written in `notes`. Two rows still lack it: 'ribbon tie' (message 377837) and 'ribbon ties' (message 102847).
