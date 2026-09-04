# Module 2 — WSD Evaluation Results

## 1. Coverage Statistics

### Overall

| Metric | Value |
|--------|-------|
| total_words | 144973 |
| aspect_words | 4997 |
| ambiguous_words | 0 |
| resolved_words | 0 |
| coverage_pct | 3.45 |
| resolution_pct | 100.0 |

### Per Language

| Language | Total Words | Aspect Words | Coverage % | Ambiguous | Resolved | Resolution % |
|----------|------------|-------------|-----------|-----------|----------|-------------|
| tamil | 51021 | 2161 | 4.24% | 0 | 0 | 100.0% |
| malayalam | 52070 | 1768 | 3.4% | 0 | 0 | 100.0% |
| kannada | 41882 | 1068 | 2.55% | 0 | 0 | 100.0% |

## 2. Aspect Distribution Across Corpus

| Aspect | Frequency |
|--------|-----------|
| fan_stardom | 1547 |
| music_bgm | 1182 |
| trailer_teaser | 1057 |
| box_office_collection | 402 |
| hero_character | 368 |
| dialogue | 228 |
| story_screenplay | 213 |

## 3. WSD vs Most-Common-Sense Baseline

| Metric | WSD (Context) | MCS (Baseline) |
|--------|--------------|----------------|
| High-confidence predictions | 337 (100.0%) | 332 (98.52%) |
| Total compared | 337 | 337 |

## 4. Sample Annotations (Flagged for Human Review)

**26 samples** with aspect annotations below. `human_verified: false` — requires manual check.

### Sample 1
**Text:** nkn benki song..asn

- `song` → **music_bgm** (conf: 1.00)

### Sample 2
**Text:** super song in kannada

- `song` → **music_bgm** (conf: 1.00)

### Sample 3
**Text:** sanchit bulhara bgm nte ponno no raksha

- `bgm` → **music_bgm** (conf: 1.00)

### Sample 4
**Text:** prajju prajju nan prajwala ondu hittu ninu kodappa from fan

- `hittu` → **box_office_collection** (conf: 1.00)
- `fan` → **fan_stardom** (conf: 1.00)

### Sample 5
**Text:** ഷൈൻ മലയാള സിനിമക്ക് പ്രതീക്ഷ തരുന്ന നടൻ..

- `നടൻ` → **hero_character** (conf: 1.00)

### Sample 6
**Text:** fan show set.arokke fan showkk pokunnu und

- `fan` → **fan_stardom** (conf: 1.00)
- `fan` → **fan_stardom** (conf: 1.00)

### Sample 7
**Text:** song is ok.. but it looks remake of tiger zinda hai song - swag se karoonga sab ka swagath..

- `song` → **music_bgm** (conf: 1.00)
- `song` → **music_bgm** (conf: 1.00)

### Sample 8
**Text:** kanan kollam pakshe kids ne pattu.. katayillata kata

- `pattu` → **music_bgm** (conf: 1.00)

### Sample 9
**Text:** செல்வராகவன் ரொம்ப எதிர் பாக்குறோம் யுவன் fan and சூர்யா fan லைக்

- `fan` → **fan_stardom** (conf: 1.00)
- `fan` → **fan_stardom** (conf: 1.00)

### Sample 10
**Text:** adhi anna songs ellame sema. ennaikume unga fan thaan adhi anna. super ravi anna vera level acting.

- `songs` → **music_bgm** (conf: 1.00)
- `fan` → **fan_stardom** (conf: 1.00)
- `acting` → **hero_character** (conf: 1.00)

### Sample 11
**Text:** viswasam trailer alavuku la illa pa

- `trailer` → **trailer_teaser** (conf: 1.00)

### Sample 12
**Text:** film hit ahagokai e tharadha song irbaiku

- `hit` → **box_office_collection** (conf: 1.00)
- `song` → **music_bgm** (conf: 1.00)

### Sample 13
**Text:** trailer ഇൽ ഉള്ള ഒന്നും പടത്തിൽ ഇല്ലല്ലോ പക്ഷെ പടം no രക്ഷ പൊളി

- `trailer` → **trailer_teaser** (conf: 1.00)

### Sample 14
**Text:** dileepetta njan kathirikkunnu santaye kanan mass all marana mass alla athukkum mele trailor polichu

- `mass` → **fan_stardom** (conf: 1.00)
- `mass` → **fan_stardom** (conf: 1.00)

### Sample 15
**Text:** marana mass da en deivam next oscar

- `mass` → **fan_stardom** (conf: 1.00)

### Sample 16
**Text:** thirumbi vanthitenu sollu oru varshathiku munnadi epidi ponaro atha vida gethaa thirumbi vanthiten sollu luv u suriya anna fan from kerala

- `fan` → **fan_stardom** (conf: 1.00)

### Sample 17
**Text:** 13 1- rajkumar chitragalanu naanu nodidheeni. haadu ishta pattidheeni. adhrey avara haadey best antha illaa.nn2- songs music composing aaa? ilayaraja goththaa? shankar ganesh goththaa? haris jayaraj g

- `haadu` → **music_bgm** (conf: 1.00)
- `songs` → **music_bgm** (conf: 1.00)
- `music` → **music_bgm** (conf: 1.00)
- `hit` → **box_office_collection** (conf: 1.00)
- `songs` → **music_bgm** (conf: 1.00)
- `music` → **music_bgm** (conf: 1.00)
- `songs` → **music_bgm** (conf: 1.00)
- `tune` → **music_bgm** (conf: 1.00)

### Sample 18
**Text:** madhura raja oru sadarana padam aaan twist twist

- `twist` → **story_screenplay** (conf: 1.00)
- `twist` → **story_screenplay** (conf: 1.00)

### Sample 19
**Text:** bgm copy anonnu oru samshayam januvariyil viriyumi enna songinte music anennu

- `bgm` → **music_bgm** (conf: 1.00)
- `music` → **music_bgm** (conf: 1.00)

### Sample 20
**Text:** your father is a great actor.but industry nalli ollethankke bele kodalla.olle natarige sambavane kodalla.mohini maukha holuva parabasha marina hakkondu movie dabba movie madi

- `actor` → **hero_character** (conf: 1.00)
