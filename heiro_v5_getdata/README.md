# Ramses V1: Comprehensive Ancient Egyptian Hieroglyphics Dataset

## Goal
Build the most comprehensive dataset of Ancient Egyptian hieroglyphic transliterations and translations by aggregating data from multiple authoritative sources.

## Data Sources

### 1. Thesaurus Linguae Aegyptiae (TLA)
- **URL**: https://thesaurus-linguae-aegyptiae.de/
- **Coverage**: ~1.5 million words from all periods of Ancient Egyptian
- **Format**: XML/Database export
- **Status**: Already have HuggingFace dataset, but can expand with direct scraping

### 2. Ramses Online
- **URL**: http://ramses.ulg.ac.be/
- **Coverage**: Extensive corpus of hieroglyphic texts
- **Format**: Web-based database
- **Status**: Primary new target

### 3. Additional Sources (To Investigate)
- **Trismegistos**: https://www.trismegistos.org/
- **Digital Egypt for Universities**: http://www.ucl.ac.uk/museums-static/digitalegypt/
- **Ancient Egyptian Texts Online**: Various university archives

## Architecture

### Scrapers (`scrapers/`)
Each scraper will be a standalone module:
- `tla_scraper.py`: Enhanced TLA data collection
- `ramses_scraper.py`: Ramses Online scraper
- `base_scraper.py`: Abstract base class with common functionality

### Data Pipeline
1. **Raw Data** (`data/raw/`): Original scraped data (JSON/XML)
2. **Processed Data** (`data/processed/`): Cleaned, normalized, deduplicated
3. **Final Corpus** (`data/corpus.txt`): Ready for model training

### Quality Control
- Deduplication across sources
- Validation of hieroglyphic transliterations
- Metadata preservation (source, date, text type)

## Ethical Considerations
- Respect robots.txt
- Rate limiting (1-2 requests/second)
- Attribution and licensing compliance
- Academic use only

## Expected Output
- **Target**: 50,000+ unique hieroglyphic sentences
- **Format**: TSV with columns: `transliteration`, `translation`, `source`, `period`, `text_type`
