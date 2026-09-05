# Module 2 — WSD Evaluation Results

## 1. Coverage Statistics

### Overall

| Metric | Value |
|--------|-------|
| total_words | 144973 |
| aspect_words | 4997 |
| ambiguous_words | 682 |
| resolved_words | 121 |
| coverage_pct | 3.45 |
| resolution_pct | 17.74 |

### Per Language

| Language | Total Words | Aspect Words | Coverage % | Ambiguous | Resolved | Resolution % |
|----------|------------|-------------|-----------|-----------|----------|-------------|
| tamil | 51021 | 2161 | 4.24% | 484 | 96 | 19.83% |
| malayalam | 52070 | 1768 | 3.4% | 190 | 22 | 11.58% |
| kannada | 41882 | 1068 | 2.55% | 8 | 3 | 37.5% |

## 2. Aspect Distribution Across Corpus

| Aspect | Frequency |
|--------|-----------|
| music_bgm | 1727 |
| trailer_teaser | 1127 |
| fan_stardom | 920 |
| hero_character | 382 |
| box_office_collection | 327 |
| dialogue | 230 |
| story_screenplay | 210 |

## 3. WSD vs Most-Common-Sense Baseline

| Metric | WSD (Context) | MCS (Baseline) |
|--------|--------------|----------------|
| High-confidence predictions | 301 (89.85%) | 330 (98.51%) |
| Total compared | 335 | 335 |

## 4. Sample Annotations (Flagged for Human Review)

**33 samples** with aspect annotations below. `human_verified: false` — requires manual check.

### Sample 1
**Text:** intha padam vetri pera vazhthukkal from thalapathy fans

- `fans` → **fan_stardom** (conf: 1.00)

### Sample 2
**Text:** adipoli trailer..joseph vijay anna .sikiraangaaa vaangaaa can't wait.waiting fr a huge blast..semma trailer..

- `trailer` → **trailer_teaser** (conf: 1.00)
- `trailer` → **trailer_teaser** (conf: 1.00)

### Sample 3
**Text:** ട്രൈലെർ സൈലന്റ് അപ്പൊ ഒരു കാര്യം ഉറപ്പാ പടം സൂപ്പർ ഹിറ്റ് ആവും ഏട്ടൻ മുത്താണ് മുത്ത്

- `ഹിറ്റ്` → **box_office_collection** (conf: 1.00)

### Sample 4
**Text:** 2109 blockbuster list la kaithi kandippa varum

- `blockbuster` → **box_office_collection** (conf: 1.00)

### Sample 5
**Text:** ellathayum vida lasta yuvan bgm

- `bgm` → **music_bgm** (conf: 1.00)

### Sample 6
**Text:** dislike madidavaru dagar rashimka fans

- `fans` → **fan_stardom** (conf: 1.00)

### Sample 7
**Text:** pakka super hit

- `hit` → **box_office_collection** (conf: 1.00)

### Sample 8
**Text:** superb trailer ..

- `trailer` → **trailer_teaser** (conf: 1.00)

### Sample 9
**Text:** super film. ee song hindi song tu hi haqqeqat pole und kurachokke.

- `song` → **music_bgm** (conf: 1.00)
- `song` → **music_bgm** (conf: 1.00)

### Sample 10
**Text:** wow thala semma , yuvan music

- `music` → **music_bgm** (conf: 1.00)

### Sample 11
**Text:** thalaiva nee vera level padam sema hit .

- `hit` → **box_office_collection** (conf: 1.00)

### Sample 12
**Text:** trailer kandu thudangyalo 7.9m views aarunu. kandu kazinjapo 8m aayi!

- `trailer` → **trailer_teaser** (conf: 1.00)

### Sample 13
**Text:** pakka mass film.lalettan

- `mass` → **music_bgm** (conf: 0.50)

### Sample 14
**Text:** super song

- `song` → **music_bgm** (conf: 1.00)

### Sample 15
**Text:** thalaivaaa kola mass therikka viduddinga anna .. thalaivaaa mass bgm ellam therikkudhu

- `mass` → **music_bgm** (conf: 0.25)
- `mass` → **music_bgm** (conf: 0.40)
- `bgm` → **music_bgm** (conf: 1.00)

### Sample 16
**Text:** therii trailer 1 crore views guarantee

- `trailer` → **trailer_teaser** (conf: 1.00)
- `crore` → **box_office_collection** (conf: 1.00)

### Sample 17
**Text:** en chindi bgm guru sakkath..jai asn. from appu abhimani

- `bgm` → **music_bgm** (conf: 1.00)
- `abhimani` → **fan_stardom** (conf: 1.00)

### Sample 18
**Text:** power pack acting and dialogue delivery and vera level look thala..kola waiting for nerkonda parvai.mass bgm.

- `acting` → **hero_character** (conf: 1.00)
- `dialogue` → **dialogue** (conf: 1.00)
- `delivery` → **dialogue** (conf: 1.00)
- `mass` → **music_bgm** (conf: 1.00)
- `bgm` → **music_bgm** (conf: 1.00)

### Sample 19
**Text:** ഇന്നാണ് ഈ പാട്ട് ശ്രദ്ധിച്ചത് കേട്ടിട്ടും കേട്ടിട്ടും മതിയാകുന്നില്ല

- `പാട്ട്` → **music_bgm** (conf: 1.00)

### Sample 20
**Text:** padam kandu mass alla maranamass

- `mass` → **music_bgm** (conf: 0.50)
