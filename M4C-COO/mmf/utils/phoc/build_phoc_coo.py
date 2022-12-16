import numpy as np

_alphabet = [
    "!",
    ".",
    "?",
    "~",
    "★",
    "☆",
    "♡",
    "♥",
    "♩",
    "♪",
    "♫",
    "♬",
    "、",
    "。",
    "ぁ",
    "あ",
    "ぃ",
    "い",
    "ぅ",
    "う",
    "ゔ",
    "ぇ",
    "え",
    "ぉ",
    "お",
    "か",
    "が",
    "き",
    "ぎ",
    "く",
    "ぐ",
    "け",
    "げ",
    "こ",
    "ご",
    "さ",
    "ざ",
    "し",
    "じ",
    "す",
    "ず",
    "せ",
    "ぜ",
    "そ",
    "ぞ",
    "た",
    "だ",
    "ち",
    "ぢ",
    "っ",
    "つ",
    "づ",
    "て",
    "で",
    "と",
    "ど",
    "な",
    "に",
    "ぬ",
    "ね",
    "の",
    "は",
    "ば",
    "ぱ",
    "ひ",
    "び",
    "ぴ",
    "ふ",
    "ぶ",
    "ぷ",
    "へ",
    "べ",
    "ぺ",
    "ほ",
    "ぼ",
    "ぽ",
    "ま",
    "み",
    "む",
    "め",
    "も",
    "ゃ",
    "や",
    "ゅ",
    "ゆ",
    "ょ",
    "よ",
    "ら",
    "り",
    "る",
    "れ",
    "ろ",
    "ゎ",
    "わ",
    "を",
    "ん",
    "゛",
    "゜",
    "ァ",
    "ア",
    "ィ",
    "イ",
    "ゥ",
    "ウ",
    "ヴ",
    "ェ",
    "エ",
    "ォ",
    "オ",
    "カ",
    "ガ",
    "キ",
    "ギ",
    "ク",
    "グ",
    "ケ",
    "ゲ",
    "コ",
    "ゴ",
    "サ",
    "ザ",
    "シ",
    "ジ",
    "ス",
    "ズ",
    "セ",
    "ゼ",
    "ソ",
    "ゾ",
    "タ",
    "ダ",
    "チ",
    "ヂ",
    "ッ",
    "ツ",
    "ヅ",
    "テ",
    "デ",
    "ト",
    "ド",
    "ナ",
    "ニ",
    "ヌ",
    "ネ",
    "ノ",
    "ハ",
    "バ",
    "パ",
    "ヒ",
    "ビ",
    "ピ",
    "フ",
    "ブ",
    "プ",
    "ヘ",
    "ベ",
    "ペ",
    "ホ",
    "ボ",
    "ポ",
    "マ",
    "ミ",
    "ム",
    "メ",
    "モ",
    "ャ",
    "ヤ",
    "ュ",
    "ユ",
    "ョ",
    "ヨ",
    "ラ",
    "リ",
    "ル",
    "レ",
    "ロ",
    "ヮ",
    "ワ",
    "ン",
    "ヶ",
    "・",
    "ー",
]

bigrams_50 = [
    "・・",
    "ーン",
    "オオ",
    "チャ",
    "はは",
    "シャ",
    "ワー",
    "ドド",
    "ゴゴ",
    "はっ",
    "ーッ",
    "ドキ",
    "ザワ",
    "ーっ",
    "アア",
    "!!",
    "パチ",
    "シュ",
    "ああ",
    "ルル",
    "ハハ",
    "ーん",
    "タン",
    "ガラ",
    "カチ",
    "ガタ",
    "カッ",
    "バッ",
    "バタ",
    "くっ",
    "ザッ",
    "ンッ",
    "ゴロ",
    "ー・",
    "ャー",
    "クッ",
    "おお",
    "キャ",
    "ドン",
    "キッ",
    "ガッ",
    "シッ",
    "バシ",
    "ガチ",
    "はあ",
    "きゃ",
    "タッ",
    "ウウ",
    "ュッ",
    "ハア",
]


def build_phoc_coo(token):
    token = token.lower().strip()
    token = "".join([c for c in token if c in _alphabet])
    phoc = phoc_coo(token, _alphabet, bigrams_50)
    phoc = np.array(phoc, dtype=np.float32)
    return phoc


# python porting from https://github.com/facebookresearch/mmf/blob/main/mmf/utils/phoc/src/cphoc.c
def phoc_coo(token, _alphabet, bigrams_50):

    phoc = np.zeros([2648])
    n = len(token)
    for index in range(n):
        char_occ0 = float(index) / n
        char_occ1 = float(index + 1) / n

        char_index = _alphabet.index(token[index])

        # check unigram levels
        for level in range(2, 6):
            for region in range(level):
                region_occ0 = float(region) / level
                region_occ1 = float(region + 1) / level
                overlap0 = max(char_occ0, region_occ0)
                overlap1 = min(char_occ1, region_occ1)
                kkk = (overlap1 - overlap0) / (char_occ1 - char_occ0)
                if kkk >= 0.5:
                    _sum = 0
                    for l in range(2, 6):
                        if l < level:
                            _sum += 1

                    feat_vec_index = _sum * 182 + region * 182 + char_index
                    phoc[feat_vec_index] = 1
                    # print("uni", feat_vec_index, level, region, _alphabet[char_index], _sum)

    # add bigrams
    ngram_offset = 182 * 14  # 14 = 2 + 3 + 4 + 5 levels
    for i in range(n - 1):
        bigram = token[i] + token[i + 1]
        try:
            ngram_index = bigrams_50.index(bigram)
            ngram_occ0 = float(i) / n
            ngram_occ1 = float(i + 2) / n

            level = 2
            for region in range(level):
                region_occ0 = float(region) / level
                region_occ1 = float(region + 1) / level
                overlap0 = max(ngram_occ0, region_occ0)
                overlap1 = min(ngram_occ1, region_occ1)
                if (overlap1 - overlap0) / (ngram_occ1 - ngram_occ0) >= 0.5:
                    phoc[ngram_offset + region * 50 + ngram_index] = 1
                    # print("bi", ngram_offset + region * 50 + ngram_index, region, ngram_index, bigrams_50[ngram_index])

        except:
            continue

    return phoc
