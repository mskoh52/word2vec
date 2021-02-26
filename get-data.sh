mkdir -p data &&
    curl https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip -O &&
    unzip -d data wikitext-2-v1.zip
