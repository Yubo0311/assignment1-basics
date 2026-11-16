from cs336_basics.tokenization import train_bpe
import argparse
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("bpe.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BPE tokenizer parameters parser")
    parser.add_argument("filename", type=str, default="data/small_piece.txt",  help = "Corpus file")
    parser.add_argument("-c", "--count", type=int, default=300, help="vocabulary size")
    parser.add_argument("-l", "--logging", action='store_true', help='Enable DEBUG output')
    args = parser.parse_args()
    logger = logging.getLogger(__name__)
    print(train_bpe(Path(args.filename), args.count, logger = logger))

