# -- coding: utf-8 --**
import pandas as pd
import requests
import json
import argparse
import re
import numpy as np
import string
import subprocess
import time


def maybe_normalize_float(span: str):
    if (
        span
        and (
            re.match(r"^[+-][0-9]+[.]?[0-9]*$", span)
            or (re.match(r"^[0-9]*[.]?[0-9]*$", span))
        )
        and span != "."
    ):
        return str(float(span))
    else:
        return span


def maybe_normalize_number(text: str) -> str:
    units = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ]
    for index, unit in enumerate(units):
        if text == unit:
            return str(float(index))
    return text


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def check_overlap(str1, str2):
    str1 = remove_punc(str1.replace(" ", ""))
    str2 = remove_punc(str2.replace(" ", ""))
    if str1 in str2 or str2 in str1:
        return True
    count = 0
    for letter in str1:
        if letter != "0":
            if letter in str2:
                count += 1
    if len(str1) == 0 or len(str2) == 0:
        return True
    else:
        return True if count / len(str1) > 0.5 or count / len(str2) > 0.5 else False


def get_answer(pred):
    match = re.search(r"(The|the) answer is ([^\.]+)\.$", pred)
    if match:
        return match.group(2).strip('"'), True
    return pred, False


def eval_ex_match(pred, gold_result):
    pred = pred.lower()
    gold_result = str(gold_result).lower()
    compare_1 = pred.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_1) == 0:
        compare_1 = " "
    compare_2 = gold_result.replace(".", "").replace(",", "").replace("%", "")
    if len(compare_2) == 0:
        compare_2 = " "
    if compare_1[0] == "-":
        compare_1 = compare_1[1:]
    if compare_2[0] == "-":
        compare_2 = compare_2[1:]
    if (
        compare_1.isdigit() == True
        and compare_2.isdigit() == True
        and pred.count(".") < 2
        and gold_result != "-"
    ):
        if pred[-1] == ".":
            pred = pred[0 : len(pred) - 1]
        gold_result = gold_result.replace(",", "").replace("%", "")
        pred = pred.replace(",", "").replace("%", "")
        pred = abs(float(pred))
        gold_result = abs(float(gold_result))
        if abs(pred - gold_result) < 0.01:
            return True, str(pred), str(gold_result)
        else:
            return False, str(pred), str(gold_result)

    if " and " in pred and "|" in gold_result:
        pred = pred.replace(" and ", ", ")

    pred = [span.strip() for span in pred.split(", ")]

    if "|" in gold_result:
        gold_result = [span.strip() for span in gold_result.split("|")]
    else:
        gold_result = [span.strip() for span in gold_result.split(", ")]

    pred = [
        maybe_normalize_number(remove_punc(remove_articles(span.strip())))
        for span in pred
    ]
    gold_result = [
        maybe_normalize_number(remove_punc(remove_articles(span.strip())))
        for span in gold_result
    ]

    clean_float = True
    if clean_float:
        pred = [maybe_normalize_float(span) for span in pred]
        gold_result = [maybe_normalize_float(span) for span in gold_result]
    indicater = False
    for item in pred:
        if item in gold_result:
            indicater = True

    if sorted(pred) == sorted(gold_result):
        indicater = True
    return sorted(pred) == sorted(gold_result), sorted(pred), sorted(gold_result)


def match_all(data, option):
    if len(data["label"]) == len(data[option + " prediction"]):
        flag = True
        for i in range(len(data["label"])):
            if_match, pred, label = eval_ex_match(
                str(data["label"][i]), data[option + " prediction"][i]
            )
            if if_match == False:
                flag = False
        if flag == True:
            return True
    else:
        return False
