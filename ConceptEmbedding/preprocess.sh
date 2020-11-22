#!/usr/bin/env bash
python WikiExtractor.py ./Dataset/Corpus/enwiki-xxxxxxxx-pages-articles-multistream.xml.bz2 --output ../Dataset/WikiOutput/Extractor/ --lustyle --html --bytes 200M --filter_disambig_pages --min_text_length 100
python ParseWikiExtractorOut.py ../Dataset/WikiOutput/Extractor/AA/