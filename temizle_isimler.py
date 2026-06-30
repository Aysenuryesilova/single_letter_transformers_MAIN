from pathlib import Path
import argparse


TURKISH_LOWER_MAP = str.maketrans({
    "I": "ı",
    "İ": "i",
})


def turkish_lower(text: str) -> str:
    return text.translate(TURKISH_LOWER_MAP).lower()


def temizle_ve_ayristir(lines: list[str]) -> list[str]:
    seen = set()
    sonuc = []

    for line in lines:
        for parca in turkish_lower(line).split():
            if parca and parca not in seen:
                seen.add(parca)
                sonuc.append(parca)

    return sonuc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Türkçe isimleri küçültür, çoklu isimleri böler ve tekrarları siler."
    )
    parser.add_argument("input", nargs="?", default="isimler.txt", help="Girdi dosyası")
    parser.add_argument("output", nargs="?", default="temiz_isimler.txt", help="Çıktı dosyası")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    sonuc = temizle_ve_ayristir(lines)
    output_path.write_text("\n".join(sonuc) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()