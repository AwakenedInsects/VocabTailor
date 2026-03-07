"""
Unicode-based vocabulary filtering utilities for the profiling pipeline.
Vendored from TokenDiet utils/unicode_filtering_utils.py for self-contained packaging.
"""
import json
import time
from typing import Any, Optional


def find_intersection(vocab_a: dict, vocab_b: dict) -> dict:
    """Finds the intersection of two vocabularies (token -> id)."""
    common_ids = set(vocab_a.values()) & set(vocab_b.values())
    return {token: id for token, id in vocab_a.items() if id in common_ids}


def convert_to_unicode_vocab_dict(unicode_points_dict: dict, verbose: bool = True) -> dict:
    """
    Flatten user-selected unicode vocabulary to a dict of unicode code point characters.
    Keys are characters, values are 0.

    Args:
        unicode_points_dict (dict): Category -> list of character strings.
        verbose (bool, optional): Print unique character count. Defaults to True.

    Returns:
        dict: Character -> 0 (flat vocab for pruning).
    """
    unicode_points_set = {
        point for code_points in unicode_points_dict.values() for point in code_points
    }
    unicode_vocab_dict = dict.fromkeys(unicode_points_set, 0)
    if verbose:
        print(f"Unique # of characters in the final candidate unicode vocab: {len(unicode_vocab_dict)}")
    return unicode_vocab_dict


def expand_unicode_range(code_points_list: list) -> list:
    """Expand unicode block ranges into a flat list of code point integers.

    Args:
        code_points_list (list): List of int code points or (start, end) range tuples.

    Returns:
        list: Flat list of integer code points.
    """
    expanded_list = []
    for i in code_points_list:
        if isinstance(i, tuple):
            expanded_list += list(range(i[0], i[1] + 1))
        else:
            expanded_list.append(i)
    return expanded_list


def expand_unicode_range_to_characters(code_points: list) -> list:
    """Expand unicode ranges to a list of characters.

    Args:
        code_points (list): List of int code points or (start, end) range tuples.

    Returns:
        list: List of character strings.
    """
    expanded_list = []
    for i in code_points:
        if isinstance(i, tuple):
            expanded_list += list(range(i[0], i[1] + 1))
        else:
            expanded_list.append(i)
    return [chr(i) for i in expanded_list]


def expand_and_filter_to_characters(
    category: str, unicode_blocks: dict, non_assigned_code_points: dict
) -> list:
    """Extract characters from specified Unicode range(s), excluding non-assigned.

    Args:
        category (str): Key in unicode_blocks (e.g. "chinese", "math_symbols").
        unicode_blocks (dict): Category -> list of code points or ranges.
        non_assigned_code_points (dict): Category -> list of code points to exclude.

    Returns:
        list: List of character strings.
    """
    if category not in unicode_blocks:
        raise KeyError(
            f"Category must be a key in unicode_blocks. category: {category}, "
            f"keys: {list(unicode_blocks.keys())}"
        )
    non_assigned_list = []
    if category in non_assigned_code_points:
        non_assigned_list = expand_unicode_range(non_assigned_code_points[category])
    character_list = expand_unicode_range(unicode_blocks[category])
    return [chr(i) for i in character_list if i not in non_assigned_list]


def get_unicode_code_points_dict_from_user_inputs(
    user_input: list[str], extreme_compress: bool = False
) -> dict:
    """
    Build a dict of unicode code point characters per category from user input.

    Args:
        user_input (list[str]): Categories to include, e.g. ['english', 'chinese', 'math'].
            Allowed: code, math, english, spanish, portuguese, italian, german, french, hindi, thai, chinese.
        extreme_compress (bool, optional): If True, omit some unicode categories. Defaults to False.

    Returns:
        dict: Category name -> list of character strings for that category.
    """
    user_input = [i.lower() for i in user_input]
    allowed_inputs = [
        "code", "math", "english", "spanish", "portuguese", "italian",
        "german", "french", "hindi", "thai", "chinese"
    ]
    for i in user_input:
        if i not in allowed_inputs:
            raise ValueError(f"User input not allowed: {i}. Allowed: {allowed_inputs}")

    unicode_blocks = {
        "control_codes": [(0, 31), (127, 159)],
        "punctuations": [
            (32, 35), (37, 42), (44, 47), 58, 59, 63, 64, (91, 96), 123, 125, 126,
        ],
        "punctuations_supp": [(0x2E5E, 0x2E7F)],
        "general_punctuations": [(0x2000, 0x206F)],
        "digits": [(48, 57)],
        "math_symbols": [
            43, (60, 62), 124,
            166, 172, 177, 181, (188, 190), 215, 247,
            (0x2200, 0x22FF), (0x2A00, 0x2AFF), (0x27C0, 0x27EF), (0x2980, 0x29FF),
            (0x1D400, 0x1D7FF),
        ],
        "superscripts_subscripts": [170, 176, 178, 179, 185, (0x2070, 0x209F)],
        "currency_symbols": [36, (162, 165), (0x20A0, 0x20CF)],
        "letterlike_symbols": [(0x2100, 0x214F)],
        "number_forms": [(0x2150, 0x218F)],
        "arrows": [(0x2190, 0x21FF)],
        "enclosed_alphanumerics": [(0x2460, 0x24FF)],
        "alphabets_basic_latin": [(65, 90), (97, 122)],
        "alphabets_latin_1": [(0x00C0, 0x00FF)],
        "alphabets_latin_extA": [(0x0100, 0x017F)],
        "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
        "thai": [(0x0E00, 0x0E7F)],
        "chinese": [
            (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
            (0x2B740, 0x2B81F), (0x2B820, 0x2CEAF), (0x2CEB0, 0x2EBEF), (0x30000, 0x3134F),
            (0x31350, 0x323AF), (0x2EBF0, 0x2EE5F), (0x3000, 0x303F), (0x16EF0, 0x16FFF),
            (0x2E80, 0x2EFF), (0x2F00, 0x2FDF), (0x31C0, 0x31EF), (0x3200, 0x32FF),
            (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0x2F800, 0x2FA1F), (0x3100, 0x312F),
            (0x31A0, 0x31BF), (0x20EA, 0x20EB), (0xFF00, 0xFFEF),
        ],
    }

    non_assigned_code_points = {
        "control_pictures": [(0x242A, 0x243F)],
        "punctuations": [(0x2E5E, 0x2E7F)],
        "punctuations_supp": [(0x2E5E, 0x2E7F)],
        "general_punctuations": [0x2065],
        "math_symbols": [
            int("0x1D" + i, 16) for i in [
                "455", "49D", "4A0", "4A1", "4A3", "4A4", "4A7", "4A8", "4AD", "4BA",
                "4BC", "4C4", "506", "50B", "50C", "515", "51D", "53A", "53F", "545",
                "547", "548", "549", "551", "6A6", "6A7", "7CC", "7CD",
            ]
        ],
        "superscripts_subscripts": [int("0x20" + i, 16) for i in ["72", "73", "8F", "9D", "9E", "9F"]],
        "currency_symbols": [(0x20C1, 0x20CF)],
        "number_forms": [(0x218C, 0x218F)],
        "arrows": [
            (0x1F80C, 0x1F80F), (0x1F848, 0x1F84F), (0x1F85A, 0x1F85F), (0x1F888, 0x1F88F),
            (0x1F8AE, 0x1F8AF), (0x1F8BC, 0x1F8BF), (0x1F8C2, 0x1F8FF),
        ] + [int("0x2B" + i, 16) for i in ["74", "75", "96"]],
        "alphabets_latin_1": [0x00D7, 0x00F7],
        "devanagari_extA": [(0x11B0A, 0x11B5F)],
        "thai": [0x0E00, (0x0E3B, 0x0E3E), (0x0E5C, 0x0E7F)],
    }

    unicode_pts_dict = {}
    for category in ["control_codes", "punctuations", "digits", "currency_symbols"]:
        unicode_pts_dict[category] = expand_and_filter_to_characters(
            category, unicode_blocks, non_assigned_code_points
        )

    if "code" in user_input:
        unicode_pts_dict["math_symbols"] = expand_and_filter_to_characters(
            "math_symbols", unicode_blocks, non_assigned_code_points
        )
    if "math" in user_input:
        unicode_pts_dict["math_symbols"] = expand_and_filter_to_characters(
            "math_symbols", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["superscripts_subscripts"] = expand_and_filter_to_characters(
            "superscripts_subscripts", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["letterlike_symbols"] = expand_and_filter_to_characters(
            "letterlike_symbols", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["number_forms"] = expand_and_filter_to_characters(
            "number_forms", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["arrows"] = expand_and_filter_to_characters(
            "arrows", unicode_blocks, non_assigned_code_points
        )
    if not extreme_compress:
        for category in [
            "punctuations_supp", "general_punctuations", "superscripts_subscripts",
            "letterlike_symbols", "number_forms", "arrows", "enclosed_alphanumerics",
        ]:
            unicode_pts_dict[category] = expand_and_filter_to_characters(
                category, unicode_blocks, non_assigned_code_points
            )

    if "english" in user_input:
        unicode_pts_dict["alphabets_english"] = expand_and_filter_to_characters(
            "alphabets_basic_latin", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["punctuations"] += [chr(i) for i in [160, 167, 168, 169, 173, 174, 180, 182, 183]]
    if "spanish" in user_input:
        alphabets_spanish = [193, 201, 205, 209, 211, 218, 220, 225, 233, 237, 241, 243, 250, 252]
        punctuation_spanish = [160, 161, 168, 169, 171, 173, 174, 175, 180, 183, 186, 187, 191]
        unicode_pts_dict["alphabets_spanish"] = expand_and_filter_to_characters(
            "alphabets_basic_latin", unicode_blocks, non_assigned_code_points
        ) + [chr(i) for i in alphabets_spanish]
        unicode_pts_dict["punctuations"] += [chr(i) for i in punctuation_spanish]
    if "portuguese" in user_input:
        alphabets_portuguese = [(192, 195), 199, (201, 202), 205, (211, 213), 218, (224, 227), 231, (233, 234), 237, (243, 245), 250]
        punctuation_portuguese = [160, 168, 169, 171, 173, 174, 180, 184, 186, 187]
        unicode_pts_dict["alphabets_portuguese"] = expand_and_filter_to_characters(
            "alphabets_basic_latin", unicode_blocks, non_assigned_code_points
        ) + expand_unicode_range_to_characters(alphabets_portuguese)
        unicode_pts_dict["punctuations"] += [chr(i) for i in punctuation_portuguese]
    if "italian" in user_input:
        alphabets_italian = [192, 200, 201, (204, 206), 210, 211, 217, 218, 224, 232, 233, (236, 238), 242, 243, 249, 250]
        punctuation_italian = [160, 169, 171, 173, 174, 180, 186, 187]
        unicode_pts_dict["alphabets_italian"] = expand_and_filter_to_characters(
            "alphabets_basic_latin", unicode_blocks, non_assigned_code_points
        ) + expand_unicode_range_to_characters(alphabets_italian)
        unicode_pts_dict["punctuations"] += [chr(i) for i in punctuation_italian]
    if "german" in user_input:
        alphabets_german = [196, 214, 220, 0x1E9E, 228, 246, 252, 223]
        punctuation_german = [160, 167, 168, 169, 171, 173, 174, 175, 182, 187]
        unicode_pts_dict["alphabets_german"] = expand_and_filter_to_characters(
            "alphabets_basic_latin", unicode_blocks, non_assigned_code_points
        ) + [chr(i) for i in alphabets_german]
        unicode_pts_dict["punctuations"] += [chr(i) for i in punctuation_german]
    if "french" in user_input:
        alphabets_french = [
            192, 194, 198, 199, (200, 203), 206, 207, 212, 338, 217, 219, 220, 376,
            224, 226, 230, 231, (232, 235), 238, 239, 244, 339, 630, 249, 251, 252, 255,
        ]
        punctuation_french = [160, 167, 168, 169, 171, 173, 174, 180, 183, 184, 187]
        unicode_pts_dict["alphabets_french"] = expand_and_filter_to_characters(
            "alphabets_basic_latin", unicode_blocks, non_assigned_code_points
        ) + expand_unicode_range_to_characters(alphabets_french)
        unicode_pts_dict["punctuations"] += [chr(i) for i in punctuation_french]
    if "hindi" in user_input:
        unicode_pts_dict["alphabets_hindi"] = expand_and_filter_to_characters(
            "devanagari", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["punctuations"] += [chr(i) for i in [160, 169, 174, 175]]
    if "thai" in user_input:
        unicode_pts_dict["alphabets_thai"] = expand_and_filter_to_characters(
            "thai", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["punctuations"] += [chr(i) for i in [160, 169, 174]]
    if "chinese" in user_input:
        unicode_pts_dict["alphabets_chinese"] = expand_and_filter_to_characters(
            "chinese", unicode_blocks, non_assigned_code_points
        )
        unicode_pts_dict["punctuations"] += [chr(i) for i in [160, 169, 171, 174, 175, 183, 187]]

    return unicode_pts_dict


def prune_vocab(
    orig_vocab: dict,
    unicode_vocab: dict,
    merges_list: list,
    tokenizer: Any,
    print_suppress: bool = True,
) -> tuple[dict, float]:
    """Prune the original vocabulary to tokens decodable from the unicode character set.

    Args:
        orig_vocab (dict): Original token -> id vocabulary.
        unicode_vocab (dict): Allowed unicode characters (e.g. from convert_to_unicode_vocab_dict).
        merges_list (list): BPE merge rules (list of [token1, token2] or "token1 token2" strings).
        tokenizer (Any): Hugging Face tokenizer.
        print_suppress (bool, optional): If False, print size and timing. Defaults to True.

    Returns:
        tuple[dict, float]: (pruned token->id dict, time taken in seconds).
    """
    start_time = time.time()
    orig_vocab_dict = {idx: token for token, idx in orig_vocab.items()}
    pruned_vocab = {}
    for c in unicode_vocab:
        encoded_ids = tokenizer.encode(c)
        filtered_ids = [i for i in encoded_ids if i < 128000]
        for idx in filtered_ids:
            if idx in orig_vocab_dict:
                pruned_vocab[orig_vocab_dict[idx]] = idx
    for merge in merges_list:
        if isinstance(merge, list):
            token1, token2 = merge[0], merge[1]
        else:
            token1, token2 = merge.split()
        if token1 in pruned_vocab and token2 in pruned_vocab:
            merged_token = "".join([token1, token2])
            if merged_token in orig_vocab:
                pruned_vocab[merged_token] = orig_vocab[merged_token]
    time_taken = time.time() - start_time
    if not print_suppress:
        print(f"Original vocab size: {len(orig_vocab)}")
        print(f"Unicode vocab size: {len(unicode_vocab)}")
        print(f"Pruned vocab size: {len(pruned_vocab)}")
        print(f"Vocabulary reduced to: {100 * len(pruned_vocab) / len(orig_vocab):.4f}%")
        print(f"Time taken: {time_taken} seconds")
    return pruned_vocab, time_taken


def generate_unicode_based_tokens(
    user_input: list[str],
    orig_vocab: dict,
    orig_merges: list,
    tokenizer: Any,
    extreme_compress: bool = True,
    save_to_json: bool = False,
    file_path: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Generate pruned vocabulary (token -> id) from unicode categories.

    Args:
        user_input (list[str]): Categories, e.g. ['english', 'chinese']. Allowed: code, math, english, spanish, portuguese, italian, german, french, hindi, thai, chinese.
        orig_vocab (dict): Original token -> id vocabulary.
        orig_merges (list): BPE merge rules.
        tokenizer (Any): Hugging Face tokenizer.
        extreme_compress (bool, optional): Passed to get_unicode_code_points_dict_from_user_inputs. Defaults to True.
        save_to_json (bool, optional): If True, write result to file_path. Defaults to False.
        file_path (str, optional): Output JSON path; required if save_to_json is True.
        verbose (bool, optional): Print progress. Defaults to True.

    Returns:
        dict: Pruned token -> id vocabulary.
    """
    if save_to_json and file_path is None:
        raise ValueError("file_path is required when save_to_json is True")
    user_input = [i.lower() for i in user_input]
    allowed = ["code", "math", "english", "spanish", "portuguese", "italian", "german", "french", "hindi", "thai", "chinese"]
    for i in user_input:
        if i not in allowed:
            raise ValueError(f"User input not allowed: {i}. Allowed: {allowed}")
    unicode_pts_dict = get_unicode_code_points_dict_from_user_inputs(
        user_input=user_input, extreme_compress=extreme_compress
    )
    unicode_vocab = convert_to_unicode_vocab_dict(unicode_pts_dict, verbose=verbose)
    pruned_vocab, _ = prune_vocab(
        orig_vocab=orig_vocab,
        unicode_vocab=unicode_vocab,
        merges_list=orig_merges,
        tokenizer=tokenizer,
        print_suppress=not verbose,
    )
    if save_to_json and file_path:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(pruned_vocab, f, indent=4)
    return pruned_vocab


__all__ = [
    "find_intersection",
    "convert_to_unicode_vocab_dict",
    "get_unicode_code_points_dict_from_user_inputs",
    "prune_vocab",
    "generate_unicode_based_tokens",
]
